# -*- coding: utf-8 -*-
import json
import os
from glob import glob

import cv2
import numpy as np
from base.logger import MLogger
from tqdm import tqdm

import mediapipe as mp
from parts.config import DirName, FileName

logger = MLogger(__name__)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic


def execute(args):
    try:
        logger.info(
            "3D姿勢推定(Mediapipe)推定処理開始: {img_dir}",
            img_dir=args.img_dir,
            decoration=MLogger.DECORATION_BOX,
        )

        if not os.path.exists(args.img_dir):
            logger.error(
                "指定された処理用ディレクトリが存在しません。: {img_dir}",
                img_dir=args.img_dir,
                decoration=MLogger.DECORATION_BOX,
            )
            return False

        if not os.path.exists(os.path.join(args.img_dir, DirName.ALPHAPOSE.value)):
            logger.error(
                "指定された2D姿勢推定ディレクトリが存在しません。\n2D姿勢推定が完了していない可能性があります。: {img_dir}",
                img_dir=os.path.join(args.img_dir, DirName.ALPHAPOSE.value),
                decoration=MLogger.DECORATION_BOX,
            )
            return False

        output_dir_path = os.path.join(args.img_dir, DirName.MEDIAPIPE.value)
        os.makedirs(output_dir_path, exist_ok=True)

        with mp_holistic.Holistic(model_complexity=2, min_detection_confidence=0.3, min_tracking_confidence=0.3) as holistic:
            for persion_json_path in glob(os.path.join(args.img_dir, DirName.ALPHAPOSE.value, "*.json")):
                if FileName.ALPHAPOSE_RESULT.value in persion_json_path:
                    continue

                json_datas = {}
                with open(persion_json_path, "r") as f:
                    json_datas = json.load(f)

                pname, _ = os.path.splitext(os.path.basename(persion_json_path))

                logger.info(
                    "【No.{pname}】Mediapipe 推定開始",
                    pname=pname,
                    decoration=MLogger.DECORATION_LINE,
                )

                for n, (fidx, frame_json_data) in enumerate(
                    tqdm(
                        json_datas["estimation"].items(),
                        desc=f"No.{pname} ... ",
                    ),
                ):
                    if "bbox" not in frame_json_data:
                        continue

                    bbox_x = int(frame_json_data["bbox"]["x"])
                    bbox_y = int(frame_json_data["bbox"]["y"])
                    bbox_w = int(frame_json_data["bbox"]["width"])
                    bbox_h = int(frame_json_data["bbox"]["height"])

                    process_img_path = frame_json_data["image"]["path"]

                    if not os.path.exists(process_img_path):
                        continue

                    image = cv2.imread(process_img_path)
                    # bboxの範囲でトリミング
                    image_trim = image[bbox_y : bbox_y + bbox_h, bbox_x : bbox_x + bbox_w]

                    if not image_trim.any():
                        continue

                    image_trim2 = cv2.cvtColor(cv2.flip(image_trim, 1), cv2.COLOR_BGR2RGB)

                    # 一旦書き込み不可
                    image_trim2.flags.writeable = False
                    results = holistic.process(image_trim2)

                    if (
                        results.pose_landmarks
                        and results.pose_landmarks.landmark
                        and results.pose_world_landmarks
                        and results.pose_world_landmarks.landmark
                    ):
                        frame_json_data["mp_body_world_joints"] = {}

                        for landmark, world_landmark, output_name in zip(
                            results.pose_landmarks.landmark,
                            results.pose_world_landmarks.landmark,
                            POSE_LANDMARKS,
                        ):
                            frame_json_data["mp_body_world_joints"][output_name] = {
                                "x": -float(world_landmark.x) * 100,
                                "y": -float(world_landmark.y) * 100,
                                "z": float(world_landmark.z) * 100,
                                "score": float(landmark.visibility),
                            }

                        for jname in ADD_POSE_LANDMARKS:
                            frame_json_data["mp_body_world_joints"][jname] = {}

                        for axis in ["x", "y", "z", "score"]:
                            # 下半身
                            frame_json_data["mp_body_world_joints"]["Pelvis"][axis] = np.mean(
                                [
                                    frame_json_data["mp_body_world_joints"]["LHip"][axis],
                                    frame_json_data["mp_body_world_joints"]["RHip"][axis],
                                ]
                            )
                            # 下半身先
                            frame_json_data["mp_body_world_joints"]["Pelvis2"][axis] = np.mean(
                                [
                                    frame_json_data["mp_body_world_joints"]["LHip"][axis],
                                    frame_json_data["mp_body_world_joints"]["RHip"][axis],
                                    frame_json_data["mp_body_world_joints"]["LKnee"][axis],
                                    frame_json_data["mp_body_world_joints"]["RKnee"][axis],
                                ]
                            )
                            # 上半身
                            frame_json_data["mp_body_world_joints"]["Spine"][axis] = np.mean(
                                [
                                    frame_json_data["mp_body_world_joints"]["LHip"][axis],
                                    frame_json_data["mp_body_world_joints"]["RHip"][axis],
                                ]
                            )
                            # 首
                            frame_json_data["mp_body_world_joints"]["Neck"][axis] = np.mean(
                                [
                                    frame_json_data["mp_body_world_joints"]["LShoulder"][axis],
                                    frame_json_data["mp_body_world_joints"]["RShoulder"][axis],
                                ]
                            )
                            # 上半身2
                            frame_json_data["mp_body_world_joints"]["Spine2"][axis] = np.mean(
                                [
                                    frame_json_data["mp_body_world_joints"]["Neck"][axis],
                                    frame_json_data["mp_body_world_joints"]["Spine"][axis],
                                ]
                            )
                            # 左肩
                            frame_json_data["mp_body_world_joints"]["LCollar"][axis] = np.mean(
                                [
                                    frame_json_data["mp_body_world_joints"]["Neck"][axis],
                                    frame_json_data["mp_body_world_joints"]["LShoulder"][axis],
                                ]
                            )
                            # 右肩
                            frame_json_data["mp_body_world_joints"]["RCollar"][axis] = np.mean(
                                [
                                    frame_json_data["mp_body_world_joints"]["Neck"][axis],
                                    frame_json_data["mp_body_world_joints"]["RShoulder"][axis],
                                ]
                            )

                        if results.right_hand_landmarks:
                            frame_json_data["mp_left_hand_joints"] = {}
                            for landmark, output_name in zip(
                                results.right_hand_landmarks.landmark,
                                HAND_LANDMARKS,
                            ):
                                frame_json_data["mp_left_hand_joints"][output_name] = {
                                    "x": -float(landmark.x),
                                    "y": -float(landmark.y),
                                    "z": float(landmark.z),
                                }

                        if results.left_hand_landmarks:
                            frame_json_data["mp_right_hand_joints"] = {}
                            for landmark, output_name in zip(results.left_hand_landmarks.landmark, HAND_LANDMARKS):
                                frame_json_data["mp_right_hand_joints"][output_name] = {
                                    "x": -float(landmark.x),
                                    "y": -float(landmark.y),
                                    "z": float(landmark.z),
                                }

                        if results.face_landmarks:
                            frame_json_data["mp_face_joints"] = {}
                            for lidx, landmark in enumerate(results.face_landmarks.landmark):
                                frame_json_data["mp_face_joints"][lidx] = {
                                    "x": -float(landmark.x),
                                    "y": -float(landmark.y),
                                    "z": float(landmark.z),
                                }

                mediapipe_json_path = os.path.join(output_dir_path, os.path.basename(persion_json_path))

                logger.info(
                    "【No.{pname}】Mediapipe 推定結果出力",
                    pname=pname,
                    decoration=MLogger.DECORATION_LINE,
                )

                with open(mediapipe_json_path, "w") as f:
                    json.dump(json_datas, f, indent=4)

        logger.info(
            "3D姿勢推定(Mediapipe) 推定処理終了: {output_dir}",
            output_dir=output_dir_path,
            decoration=MLogger.DECORATION_BOX,
        )

        return True
    except Exception as e:
        logger.critical("Mediapipeで予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


# 左右逆
POSE_LANDMARKS = [
    "Nose",
    "REyeIn",
    "REye",
    "REyeOut",
    "LEyeIn",
    "LEye",
    "LEyeOut",
    "REar",
    "LEar",
    "RMouth",
    "LMouth",
    "RShoulder",
    "LShoulder",
    "RElbow",
    "LElbow",
    "RWrist",
    "LWrist",
    "RPinky",
    "LPinky",
    "RIndex",
    "LIndex",
    "RThumb",
    "LThumb",
    "RHip",
    "LHip",
    "RKnee",
    "LKnee",
    "RAnkle",
    "LAnkle",
    "RHeel",
    "LHeel",
    "RFootIndex",
    "LFootIndex",
]

ADD_POSE_LANDMARKS = [
    "Pelvis",
    "Pelvis2",
    "Spine",
    "Spine2",
    "Neck",
    "LCollar",
    "RCollar",
]

HAND_LANDMARKS = [
    "wrist",
    "thumb1",
    "thumb2",
    "thumb3",
    "thumb",
    "index1",
    "index2",
    "index3",
    "index",
    "middle1",
    "middle2",
    "middle3",
    "middle",
    "ring1",
    "ring2",
    "ring3",
    "ring",
    "pinky1",
    "pinky2",
    "pinky3",
    "pinky",
]
