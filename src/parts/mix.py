import json
import os
import re
from datetime import datetime
from glob import glob

import numpy as np
from base.exception import MApplicationException
from base.logger import MLogger
from base.math import MMatrix4x4, MQuaternion, MVector3D
from pykalman import KalmanFilter
from tqdm import tqdm

from parts.config import SMPL_JOINT_24, SMPL_JOINT_29, DirName

logger = MLogger(__name__)

# 身長158cmプラグインより
MIKU_CM = 0.1259496


def execute(args):
    logger.info(
        "推定結果合成 処理開始: {img_dir}",
        img_dir=args.img_dir,
        decoration=MLogger.DECORATION_BOX,
    )

    if not os.path.exists(args.img_dir):
        logger.error(
            "指定された処理用ディレクトリが存在しません。: {img_dir}",
            img_dir=args.img_dir,
            decoration=MLogger.DECORATION_BOX,
        )
        raise MApplicationException()

    if not os.path.exists(os.path.join(args.img_dir, DirName.POSETRIPLET.value)):
        logger.error(
            "指定されたPoseTripletディレクトリが存在しません。\nPoseTripletが完了していない可能性があります。: {img_dir}",
            img_dir=os.path.join(args.img_dir, DirName.POSETRIPLET.value),
            decoration=MLogger.DECORATION_BOX,
        )
        raise MApplicationException()

    try:
        output_dir_path = os.path.join(args.img_dir, DirName.MIX.value)

        if os.path.exists(output_dir_path):
            os.rename(output_dir_path, f"{output_dir_path}_{datetime.fromtimestamp(os.stat(output_dir_path).st_ctime).strftime('%Y%m%d_%H%M%S')}")

        os.makedirs(output_dir_path)

        logger.info(
            "推定結果合成 中央判定",
            decoration=MLogger.DECORATION_LINE,
        )

        target_pnames = []
        all_depths: dict[int, dict[int, float]] = {}
        for personal_json_path in tqdm(sorted(glob(os.path.join(args.img_dir, DirName.POSETRIPLET.value, "*.json")))):
            pname, _ = os.path.splitext(os.path.basename(personal_json_path))

            frame_joints = {}
            with open(personal_json_path, "r") as f:
                frame_joints = json.load(f)

            for sfno in frame_joints["estimation"].keys():
                if "pt-keypoints" in frame_joints["estimation"][sfno] and "depth" in frame_joints["estimation"][sfno]:
                    # Mediapipeの情報がある最初のキーフレを選択する
                    target_pnames.append(pname)
                    break

            if pname not in target_pnames:
                continue

            fno = int(sfno)
            if fno not in all_depths:
                all_depths[fno] = {}
            # メーターからcmに変換
            all_depths[fno][pname] = frame_joints["estimation"][sfno]["depth"] * 100

        start_fno = list(sorted(list(all_depths.keys())))[0]
        # Zは一番手前が0になるように
        root_depth = np.min(list(all_depths[start_fno].values()))

        for personal_json_path in sorted(glob(os.path.join(args.img_dir, DirName.POSETRIPLET.value, "*.json"))):
            pname, _ = os.path.splitext(os.path.basename(personal_json_path))

            if pname not in target_pnames:
                continue

            logger.info(
                "【No.{pname}】推定結果合成 準備開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            frame_joints = {}
            with open(personal_json_path, "r") as f:
                frame_joints = json.load(f)

            fnos = [int(sfno) for sfno in frame_joints["estimation"].keys()]

            joint_datas = {}
            for fidx, fno in tqdm(enumerate(fnos), desc=f"No.{pname} ... "):
                frame_json_data = frame_joints["estimation"][str(fno)]
                if "ap-3d-keypoints" in frame_json_data:
                    for jname, kps in zip(SMPL_JOINT_24.keys(), frame_json_data["ap-3d-keypoints"]):
                        for axis, kp in zip(("x", "y", "z"), kps):
                            if ("ap", jname, axis) not in joint_datas:
                                joint_datas[("ap", jname, axis)] = {}
                            # meterからcmに変換。Yは上がマイナスなので、符号反転
                            joint_datas[("ap", jname, axis)][fno] = float(kp) * 100 * (-1 if axis == "y" else 1)
                if "pt-keypoints" in frame_json_data:
                    for jname, jidx in zip(("Pelvis", "RAnkle", "LAnkle"), (0, 3, 6)):
                        # PoseTripletはzがY軸相当
                        for axis, kp in zip(
                            ("x", "z", "y"),
                            frame_json_data["pt-keypoints"][jidx],
                        ):
                            if ("pt", jname, axis) not in joint_datas:
                                joint_datas[("pt", jname, axis)] = {}
                            # meterからcmに変換。Xは右がマイナスなので、符号反転
                            joint_datas[("pt", jname, axis)][fno] = float(kp) * 100 * (-1 if axis in ["x", "z"] else 1)
                if "depth" in frame_json_data:
                    if ("mp", "depth", "d") not in joint_datas:
                        joint_datas[("mp", "depth", "d")] = {}
                    # meterからcmに変換
                    joint_datas[("mp", "depth", "d")][fno] = float(frame_json_data["depth"]) * 100

            logger.info(
                "【No.{pname}】推定結果合成 スムージング",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            smoothed_values = {}
            with tqdm(
                total=(len(SMPL_JOINT_24.keys()) + 3 + 1),
                desc=f"No.{pname} ... ",
            ) as pchar:
                kf = KalmanFilter(n_dim_state=3, n_dim_obs=3)
                for jname in SMPL_JOINT_24.keys():
                    values = np.vstack(
                        [
                            np.array(list(joint_datas[("ap", jname, "x")].values())),
                            np.array(list(joint_datas[("ap", jname, "y")].values())),
                            np.array(list(joint_datas[("ap", jname, "z")].values())),
                        ]
                    ).T
                    smoothed_values[("ap", jname)] = kf.em(values).smooth(values)[0]
                    smoothed_values[("ap", jname)][:10] = values[:10]
                    pchar.update(1)

                for jname in ("Pelvis", "RAnkle", "LAnkle"):
                    values = np.vstack(
                        [
                            np.array(list(joint_datas[("pt", jname, "x")].values())),
                            np.array(list(joint_datas[("pt", jname, "y")].values())),
                            np.array(list(joint_datas[("pt", jname, "z")].values())),
                        ]
                    ).T
                    smoothed_values[("pt", jname)] = kf.em(values).smooth(values)[0]
                    smoothed_values[("pt", jname)][:10] = values[:10]
                    pchar.update(1)

                kf = KalmanFilter(n_dim_state=1, n_dim_obs=1)
                values = np.array(list(joint_datas[("mp", "depth", "d")].values()))
                smoothed_values[("mp", "depth")] = kf.em(values).smooth(values)[0]
                smoothed_values[("mp", "depth")][:10] = np.array([values[:10]]).T
                pchar.update(1)

            logger.info(
                "【No.{pname}】推定結果合成 グルーブ補正",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            ys = []
            for jname in SMPL_JOINT_24.keys():
                ys.append(smoothed_values[("ap", jname)][:, 1])
            ys = np.array(ys)
            min_ys = np.min(ys, axis=0)

            for jname in SMPL_JOINT_24.keys():
                smoothed_values[("ap", jname)][:, 1] -= min_ys

            logger.info(
                "【No.{pname}】推定結果合成 合成開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            initial_nose_pos = MVector3D(0, 17.59783, -1.127905)
            initial_head_pos = MVector3D(0, 17.33944, 0.3088881)
            initial_neck_pos = MVector3D(0, 16.42476, 0.4232453)
            initial_direction: MVector3D = (initial_nose_pos - initial_head_pos).normalized()
            initial_up: MVector3D = (initial_head_pos - initial_neck_pos).normalized()
            initial_cross: MVector3D = initial_up.cross(initial_direction).normalized()
            initial_head_qq = MQuaternion.from_direction(initial_direction, initial_cross)

            initial_left_ear_pos = (MVector3D(1.147481, 17.91739, 0.4137991) - initial_nose_pos) / MIKU_CM
            initial_right_ear_pos = (MVector3D(-1.147481, 17.91739, 0.4137991) - initial_nose_pos) / MIKU_CM

            # PoseTriplet で測った場合のY座標（ジャンプしてるとその分上に行く）
            pt_leg_lengths = smoothed_values["pt", "Pelvis"][:, 1] * 0.8
            # AlphaPose で測った場合のY座標（もっともYが低い関節からの距離）
            ap_leg_lengths = smoothed_values["ap", "Pelvis"][:, 1]

            # ジャンプしてる場合はptの方が値が大きくなるので、＋のみ判定（接地までとする）
            # ややPoseTriplet を低めに見積もってるので、実値としてはかさ増しする
            adjust_foot_ys = np.where(pt_leg_lengths - ap_leg_lengths > 0, pt_leg_lengths - ap_leg_lengths, 0) * 1.5

            mix_joints = {"color": frame_joints["color"], "joints": {}}
            for fidx, fno in tqdm(
                enumerate(fnos),
                desc=f"No.{pname} ... ",
            ):
                mix_joints["joints"][fno] = {"body": {}, "2d": {}}
                sfno = str(fno)

                # 2Dの足情報を設定する
                mix_joints["joints"][fno]["2d"] = {
                    "LHip": {
                        "x": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["LHip"] * 3],
                        "y": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["LHip"] * 3 + 1],
                    },
                    "RHip": {
                        "x": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["RHip"] * 3],
                        "y": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["RHip"] * 3 + 1],
                    },
                    "LKnee": {
                        "x": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["LKnee"] * 3],
                        "y": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["LKnee"] * 3 + 1],
                    },
                    "RKnee": {
                        "x": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["RKnee"] * 3],
                        "y": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["RKnee"] * 3 + 1],
                    },
                    "LAnkle": {
                        "x": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["LAnkle"] * 3],
                        "y": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["LAnkle"] * 3 + 1],
                    },
                    "RAnkle": {
                        "x": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["RAnkle"] * 3],
                        "y": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["RAnkle"] * 3 + 1],
                    },
                    "LFoot": {
                        "x": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["LFoot"] * 3],
                        "y": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["LFoot"] * 3 + 1],
                    },
                    "RFoot": {
                        "x": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["RFoot"] * 3],
                        "y": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["RFoot"] * 3 + 1],
                    },
                    "LBigtoe": {
                        "x": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["LBigtoe"] * 3],
                        "y": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["LBigtoe"] * 3 + 1],
                    },
                    "RBigtoe": {
                        "x": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["RBigtoe"] * 3],
                        "y": frame_joints["estimation"][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["RBigtoe"] * 3 + 1],
                    },
                }

                for jname in SMPL_JOINT_24.keys():
                    mix_joints["joints"][fno]["body"][jname] = {
                        "x": float(smoothed_values[("ap", jname)][fidx, 0] + (smoothed_values[("pt", "Pelvis")][fidx, 0] * 1.2)),
                        "y": float(smoothed_values[("ap", jname)][fidx, 1] + adjust_foot_ys[fidx]),
                        "z": float(smoothed_values[("ap", jname)][fidx, 2] + ((smoothed_values[("mp", "depth")][fidx] - root_depth) * 1.3)),
                    }

                mix_joints["joints"][fno]["body"]["Pelvis2"] = {}
                mix_joints["joints"][fno]["body"]["Spine0"] = {}
                mix_joints["joints"][fno]["body"]["Neck"] = {}

                for axis in ("x", "y", "z"):
                    # 上半身
                    mix_joints["joints"][fno]["body"]["Spine0"][axis] = mix_joints["joints"][fno]["body"]["Pelvis"][axis]

                    # 下半身先
                    mix_joints["joints"][fno]["body"]["Pelvis2"][axis] = np.mean(
                        [
                            mix_joints["joints"][fno]["body"]["LHip"][axis],
                            mix_joints["joints"][fno]["body"]["RHip"][axis],
                        ]
                    )

                    # 首
                    mix_joints["joints"][fno]["body"]["Neck"][axis] = np.mean(
                        [
                            mix_joints["joints"][fno]["body"]["Head"][axis],
                            mix_joints["joints"][fno]["body"]["Spine3"][axis],
                        ]
                    )

                # 耳位置を暫定で求める
                head_pos = MVector3D(
                    mix_joints["joints"][fno]["body"]["Head"]["x"],
                    mix_joints["joints"][fno]["body"]["Head"]["y"],
                    mix_joints["joints"][fno]["body"]["Head"]["z"],
                )
                neck_pos = MVector3D(
                    mix_joints["joints"][fno]["body"]["Neck"]["x"],
                    mix_joints["joints"][fno]["body"]["Neck"]["y"],
                    mix_joints["joints"][fno]["body"]["Neck"]["z"],
                )
                nose_pos = MVector3D(
                    mix_joints["joints"][fno]["body"]["Nose"]["x"],
                    mix_joints["joints"][fno]["body"]["Nose"]["y"],
                    mix_joints["joints"][fno]["body"]["Nose"]["z"],
                )

                direction: MVector3D = (nose_pos - head_pos).normalized()
                up: MVector3D = (head_pos - neck_pos).normalized()
                cross: MVector3D = up.cross(direction).normalized()
                head_qq = MQuaternion.from_direction(direction, cross) * initial_head_qq.inverse()

                ear_mat = MMatrix4x4(identity=True)
                ear_mat.translate(nose_pos)
                ear_mat.rotate(head_qq)

                left_ear_pos = ear_mat * initial_left_ear_pos
                right_ear_pos = ear_mat * initial_right_ear_pos

                mix_joints["joints"][fno]["body"]["LEar"] = {
                    "x": float(left_ear_pos.x),
                    "y": float(left_ear_pos.y),
                    "z": float(left_ear_pos.z),
                }
                mix_joints["joints"][fno]["body"]["REar"] = {
                    "x": float(right_ear_pos.x),
                    "y": float(right_ear_pos.y),
                    "z": float(right_ear_pos.z),
                }

            logger.info(
                "【No.{pname}】推定結果合成 出力開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            with open(
                os.path.join(output_dir_path, f"{pname}.json"),
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(mix_joints, f, indent=4)

        logger.info(
            "推定結果合成 処理終了: {img_dir}",
            img_dir=args.img_dir,
            decoration=MLogger.DECORATION_BOX,
        )

        return True
    except Exception as e:
        logger.critical("推定結果合成で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        raise e
