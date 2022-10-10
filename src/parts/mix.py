import json
import os
import re
from glob import glob

import numpy as np
from base.logger import MLogger
from base.math import MVector3D
from scipy.signal import savgol_filter
from tqdm import tqdm

from parts.config import DirName
from parts.mediapipe import ADD_POSE_LANDMARKS, HAND_LANDMARKS, POSE_LANDMARKS

logger = MLogger(__name__, level=MLogger.DEBUG)


def execute(args):
    try:
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
            return False

        if not os.path.exists(os.path.join(args.img_dir, DirName.POSETRIPLET.value)):
            logger.error(
                "指定されたPoseTripletディレクトリが存在しません。\nPoseTripletが完了していない可能性があります。: {img_dir}",
                img_dir=os.path.join(args.img_dir, DirName.POSETRIPLET.value),
                decoration=MLogger.DECORATION_BOX,
            )
            return False

        os.makedirs(os.path.join(args.img_dir, DirName.MIX.value), exist_ok=True)

        logger.info(
            "推定結果合成 中央判定",
            decoration=MLogger.DECORATION_LINE,
        )

        target_pnames = []
        all_root_poses: dict[int, dict[str, MVector3D]] = {}
        for personal_json_path in tqdm(sorted(glob(os.path.join(args.img_dir, DirName.POSETRIPLET.value, "*.json")))):
            pname, _ = os.path.splitext(os.path.basename(personal_json_path))

            frame_joints = {}
            with open(personal_json_path, "r") as f:
                frame_joints = json.load(f)

            for fno in frame_joints["estimation"].keys():
                if "pt_joints" in frame_joints["estimation"][fno] and "mp_body_world_joints" in frame_joints["estimation"][fno]:
                    # Mediapipeの情報がある最初のキーフレを選択する
                    target_pnames.append(pname)
                    break

            if pname not in target_pnames:
                continue

            if fno not in all_root_poses:
                all_root_poses[fno] = {}
            all_root_poses[fno][pname] = MVector3D(
                frame_joints["estimation"][fno]["pt_joints"]["Pelvis"]["x"],
                frame_joints["estimation"][fno]["pt_joints"]["Pelvis"]["y"],
                frame_joints["estimation"][fno]["pt_joints"]["Pelvis"]["z"],
            )

        start_fno = list(sorted(list(all_root_poses.keys())))[0]
        xs = []
        zs = []
        for pname, rpos in all_root_poses[start_fno].items():
            xs.append(rpos.x)
            zs.append(rpos.z)
        # よりセンターに近い方がrootとなる
        all_root_pos = MVector3D()
        # Xは全体の中央
        min_root_x = np.min(xs)
        max_root_x = np.max(xs)
        all_root_pos.x = min_root_x + ((max_root_x - min_root_x) / 2)
        # Zは一番手前が0になるように
        all_root_pos.z = np.min(zs)

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

            max_fno = 0
            joint_datas = {}
            for fidx, frame_json_data in tqdm(frame_joints["estimation"].items(), desc=f"No.{pname} ... "):
                fno = int(fidx)
                for joint_type in ("pt_joints", "mp_body_world_joints", "mp_left_hand_joints", "mp_right_hand_joints", "mp_face_joints"):
                    if joint_type not in frame_json_data:
                        continue

                    for joint_name, jval in frame_json_data[joint_type].items():
                        for axis in ["x", "y", "z"]:
                            if (joint_type, joint_name, axis) not in joint_datas:
                                joint_datas[(joint_type, joint_name, axis)] = {}
                            joint_datas[(joint_type, joint_name, axis)][fno] = float(jval[axis])

                max_fno = fno
            max_fno += 1

            logger.info(
                "【No.{pname}】推定結果合成 スムージング",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            for key, joint_vals in tqdm(joint_datas.items()):
                # スムージング
                if len(joint_vals) > 5:
                    smoothed_joint_vals = savgol_filter(
                        list(joint_vals.values()),
                        window_length=5,
                        polyorder=2,
                    )
                else:
                    smoothed_joint_vals = joint_vals

                for fidx, fno in enumerate(joint_vals.keys()):
                    joint_datas[key][fno] = smoothed_joint_vals[fidx]

            logger.info(
                "【No.{pname}】推定結果合成 センター補正",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            mp_pelvis_ys = {}
            # mp_hip_ys = {"L": {}, "R": {}}
            # mp_knee_ys = {"L": {}, "R": {}}
            mp_ankle_ys = {"L": {}, "R": {}}
            pt_pelvis_ys = {}
            pt_ankle_ys = {"L": {}, "R": {}}
            # leg_lengths = {"L": {}, "R": {}}
            for direction in ("L", "R"):
                for fno in tqdm(joint_datas["pt_joints", f"{direction}Ankle", "y"].keys(), desc=f"No.{pname} ... "):
                    if fno in joint_datas["pt_joints", "Pelvis", "x"] and fno in joint_datas["pt_joints", f"{direction}Ankle", "x"]:
                        pt_pelvis_ys[fno] = joint_datas["pt_joints", "Pelvis", "y"][fno]
                        pt_ankle_ys[direction][fno] = joint_datas["pt_joints", f"{direction}Ankle", "y"][fno]

                for fidx, fno in tqdm(enumerate(joint_datas["mp_body_world_joints", f"{direction}Ankle", "y"].keys()), desc=f"No.{pname} ... "):
                    if (
                        fno in joint_datas["mp_body_world_joints", "Pelvis", "x"]
                        # and fno in joint_datas["mp_body_world_joints", f"{direction}Hip", "y"]
                        # and fno in joint_datas["mp_body_world_joints", f"{direction}Knee", "y"]
                        and fno in joint_datas["mp_body_world_joints", f"{direction}Ankle", "y"]
                    ):
                        if fidx > 0 and fno not in joint_datas["mp_body_world_joints", "Pelvis", "y"]:
                            prev_fno = (joint_datas["mp_body_world_joints", "Pelvis", "y"].keys())[fidx - 1]
                            mp_pelvis_ys[fno] = joint_datas["mp_body_world_joints", "Pelvis", "y"][prev_fno]
                        else:
                            mp_pelvis_ys[fno] = joint_datas["mp_body_world_joints", "Pelvis", "y"][fno]
                        # mp_hip_ys[direction][fno] = joint_datas["mp_body_world_joints", f"{direction}Hip", "y"][fno]
                        # mp_knee_ys[direction][fno] = joint_datas["mp_body_world_joints", f"{direction}Knee", "y"][fno]
                        mp_ankle_ys[direction][fno] = joint_datas["mp_body_world_joints", f"{direction}Ankle", "y"][fno]

                        # hip_pos = MVector3D(
                        #     joint_datas["mp_body_world_joints", f"{direction}Hip", "x"][fno],
                        #     joint_datas["mp_body_world_joints", f"{direction}Hip", "y"][fno],
                        #     joint_datas["mp_body_world_joints", f"{direction}Hip", "z"][fno],
                        # )
                        # knee_pos = MVector3D(
                        #     joint_datas["mp_body_world_joints", f"{direction}Knee", "x"][fno],
                        #     joint_datas["mp_body_world_joints", f"{direction}Knee", "y"][fno],
                        #     joint_datas["mp_body_world_joints", f"{direction}Knee", "z"][fno],
                        # )
                        # ankle_pos = MVector3D(
                        #     joint_datas["mp_body_world_joints", f"{direction}Ankle", "x"][fno],
                        #     joint_datas["mp_body_world_joints", f"{direction}Ankle", "y"][fno],
                        #     joint_datas["mp_body_world_joints", f"{direction}Ankle", "z"][fno],
                        # )

                        # leg_lengths[direction][fno] = hip_pos.distance(knee_pos) + knee_pos.distance(ankle_pos)

            # median_leg_length = np.median(list(leg_lengths["L"].values()) + list(leg_lengths["R"].values()))
            # median_ankle_height = abs(np.median([list(mp_ankle_ys["L"].values()), list(mp_ankle_ys["R"].values())]))

            logger.info(
                "【No.{pname}】推定結果合成 合成開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            mix_joints = {"color": frame_joints["color"], "joints": {}}
            for fno in tqdm(
                joint_datas[("mp_body_world_joints", "Pelvis", "x")].keys(),
            ):
                mix_joints["joints"][fno] = {"body": {}, "left_hand": {}, "right_hand": {}, "face": {}, "2d": {}}

                # 2Dの足情報を設定する
                mix_joints["joints"][fno]["2d"] = frame_joints["estimation"][str(fno)]["ap_joints"]

                if not (
                    ("mp_body_world_joints", "Pelvis", "x") in joint_datas
                    and fno in joint_datas[("mp_body_world_joints", "Pelvis", "x")]
                    # and ("mp_body_world_joints", "LHeel", "x") in joint_datas
                    # and fno in joint_datas[("mp_body_world_joints", "LHeel", "x")]
                    # and ("mp_body_world_joints", "RHeel", "x") in joint_datas
                    # and fno in joint_datas[("mp_body_world_joints", "RHeel", "x")]
                    # and fno in mp_ankle_ys["L"]
                    and ("pt_joints", "Pelvis", "x") in joint_datas
                    and fno in joint_datas[("pt_joints", "Pelvis", "x")]
                ):
                    continue

                median_ankle_y = np.median([list(pt_ankle_ys["L"].values()), list(pt_ankle_ys["R"].values())])
                pt_leg_length = pt_pelvis_ys[fno] - median_ankle_y
                mp_leg_length = mp_pelvis_ys[fno] - np.min([mp_ankle_ys["L"].get(fno, 0), mp_ankle_ys["R"].get(fno, 0)])

                # 腰からの足首Yの距離
                foot_y = joint_datas[("mp_body_world_joints", "Pelvis", "y")][fno] - min(
                    joint_datas[("mp_body_world_joints", "RFootIndex", "y")].get(fno, 0),
                    joint_datas[("mp_body_world_joints", "RHeel", "y")].get(fno, 0),
                    joint_datas[("mp_body_world_joints", "LFootIndex", "y")].get(fno, 0),
                    joint_datas[("mp_body_world_joints", "LHeel", "y")].get(fno, 0),
                )
                # ジャンプしてる場合はptの方が値が大きくなるので、＋のみ判定（接地までとする）
                foot_y += max(0, pt_leg_length - mp_leg_length)

                pelvis_x = joint_datas[("pt_joints", "Pelvis", "x")][fno] - all_root_pos.x
                pelvis_z = joint_datas[("pt_joints", "Pelvis", "z")][fno] - all_root_pos.z

                for joint_name in POSE_LANDMARKS + ADD_POSE_LANDMARKS:
                    if not (
                        ("mp_body_world_joints", joint_name, "x") in joint_datas
                        and ("mp_body_world_joints", joint_name, "y") in joint_datas
                        and ("mp_body_world_joints", joint_name, "z") in joint_datas
                        and fno in joint_datas[("mp_body_world_joints", joint_name, "x")]
                        and fno in joint_datas[("mp_body_world_joints", joint_name, "y")]
                        and fno in joint_datas[("mp_body_world_joints", joint_name, "z")]
                    ):
                        continue

                    mix_joints["joints"][fno]["body"][joint_name] = {
                        "x": joint_datas[("mp_body_world_joints", joint_name, "x")][fno] + pelvis_x,
                        "y": joint_datas[("mp_body_world_joints", joint_name, "y")][fno] + foot_y,
                        "z": joint_datas[("mp_body_world_joints", joint_name, "z")][fno] + pelvis_z,
                        "score": frame_joints["estimation"].get(str(fno), {}).get("mp_body_world_joints", {}).get(joint_name, {}).get("score", 1.0),
                    }

                    # if 10 < mix_joints["joints"][fno]["body"][joint_name]["y"] and (
                    #     "Ankle" in joint_name or "Heel" in joint_name or "FootIndex" in joint_name
                    # ):
                    #     mix_joints["joints"][fno]["body"][joint_name]["y"] *= 1.4

                for odd_direction, direction in (("L", "left"), ("R", "right")):
                    body_wrist_jname = f"{odd_direction}Wrist"
                    hand_jtype = f"mp_{direction}_hand_joints"
                    hand_output_jtype = f"{direction}_hand"

                    if not (
                        fno in mix_joints["joints"]
                        and body_wrist_jname in mix_joints["joints"][fno]["body"]
                        and (hand_jtype, "wrist", "x") in joint_datas
                        and (hand_jtype, "wrist", "y") in joint_datas
                        and (hand_jtype, "wrist", "z") in joint_datas
                        and fno in joint_datas[(hand_jtype, "wrist", "x")]
                        and fno in joint_datas[(hand_jtype, "wrist", "y")]
                        and fno in joint_datas[(hand_jtype, "wrist", "z")]
                    ):
                        continue

                    hand_root_vec = {}
                    for axis in ["x", "y", "z"]:
                        hand_root_vec[axis] = float(
                            joint_datas[(hand_jtype, "wrist", axis)][fno] - mix_joints["joints"][fno]["body"][body_wrist_jname][axis]
                        )

                    for joint_name in HAND_LANDMARKS:
                        if not (
                            (hand_jtype, joint_name, "x") in joint_datas
                            and (hand_jtype, joint_name, "y") in joint_datas
                            and (hand_jtype, joint_name, "z") in joint_datas
                            and fno in joint_datas[(hand_jtype, joint_name, "x")]
                            and fno in joint_datas[(hand_jtype, joint_name, "y")]
                            and fno in joint_datas[(hand_jtype, joint_name, "z")]
                        ):
                            continue

                        mix_joints["joints"][fno][hand_output_jtype][joint_name] = {
                            "x": joint_datas[(hand_jtype, joint_name, "x")][fno],
                            "y": joint_datas[(hand_jtype, joint_name, "y")][fno],
                            "z": joint_datas[(hand_jtype, joint_name, "z")][fno],
                        }

                    for joint_name in range(468):
                        if not (
                            ("mp_face_joints", joint_name, "x") in joint_datas
                            and ("mp_face_joints", joint_name, "y") in joint_datas
                            and ("mp_face_joints", joint_name, "z") in joint_datas
                            and fno in joint_datas[("mp_face_joints", joint_name, "x")]
                            and fno in joint_datas[("mp_face_joints", joint_name, "y")]
                            and fno in joint_datas[("mp_face_joints", joint_name, "z")]
                        ):
                            continue

                        mix_joints["joints"][fno]["face"][joint_name] = {
                            "x": joint_datas[("mp_face_joints", joint_name, "x")][fno],
                            "y": joint_datas[("mp_face_joints", joint_name, "y")][fno],
                            "z": joint_datas[("mp_face_joints", joint_name, "z")][fno],
                        }

            logger.info(
                "【No.{pname}】推定結果合成 出力開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            with open(
                os.path.join(args.img_dir, DirName.MIX.value, f"{pname}.json"),
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
        return False
