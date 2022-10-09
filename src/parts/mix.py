import json
import os
import re
from glob import glob

import numpy as np
from base.bezier import get_infections
from base.logger import MLogger
from base.math import MVector3D
from tqdm import tqdm

from parts.config import DirName
from parts.mediapipe import HAND_LANDMARKS
from parts.posetriplet import BODY_LANDMARKS

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
                    # PoseTripletの情報がある最初のキーフレを選択する
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

                if "mp_body_world_joints" in frame_json_data:
                    for joint_name, jval in frame_json_data["mp_body_world_joints"].items():
                        if ("mp_body_world_joints", joint_name, "score") not in joint_datas:
                            joint_datas[("mp_body_world_joints", joint_name, "score")] = {}
                        joint_datas[("mp_body_world_joints", joint_name, "score")][fno] = float(jval.get("score", 1.0))

                max_fno = fno
            max_fno += 1

            logger.info(
                "【No.{pname}】推定結果合成 センター補正",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            hip_ys = {"L": {}, "R": {}}
            knee_ys = {"L": {}, "R": {}}
            ankle_ys = {"L": {}, "R": {}}
            leg_lengths = {"L": {}, "R": {}}
            for direction in ("L", "R"):
                for fno in tqdm(joint_datas["pt_joints", f"{direction}Hip", "y"].keys(), desc=f"No.{pname} ... "):
                    if (
                        fno in joint_datas["pt_joints", "Pelvis", "x"]
                        and fno in joint_datas["pt_joints", f"{direction}Hip", "x"]
                        and fno in joint_datas["pt_joints", f"{direction}Knee", "x"]
                        and fno in joint_datas["pt_joints", f"{direction}Ankle", "x"]
                    ):
                        hip_ys[direction][fno] = joint_datas["pt_joints", f"{direction}Hip", "y"][fno]
                        knee_ys[direction][fno] = joint_datas["pt_joints", f"{direction}Knee", "y"][fno]
                        ankle_ys[direction][fno] = joint_datas["pt_joints", f"{direction}Ankle", "y"][fno]

                        hip_pos = MVector3D(
                            joint_datas["pt_joints", f"{direction}Hip", "x"][fno],
                            joint_datas["pt_joints", f"{direction}Hip", "y"][fno],
                            joint_datas["pt_joints", f"{direction}Hip", "z"][fno],
                        )
                        knee_pos = MVector3D(
                            joint_datas["pt_joints", f"{direction}Knee", "x"][fno],
                            joint_datas["pt_joints", f"{direction}Knee", "y"][fno],
                            joint_datas["pt_joints", f"{direction}Knee", "z"][fno],
                        )
                        ankle_pos = MVector3D(
                            joint_datas["pt_joints", f"{direction}Ankle", "x"][fno],
                            joint_datas["pt_joints", f"{direction}Ankle", "y"][fno],
                            joint_datas["pt_joints", f"{direction}Ankle", "z"][fno],
                        )

                        leg_lengths[direction][fno] = hip_pos.distance(knee_pos) + knee_pos.distance(ankle_pos)

            median_leg_length = np.median(list(leg_lengths["L"].values()) + list(leg_lengths["R"].values()))
            mean_ankle_y = np.mean(
                [
                    np.mean([list(ankle_ys["L"].values()), list(ankle_ys["R"].values())]),
                    np.median([list(ankle_ys["L"].values()), list(ankle_ys["R"].values())]),
                ]
            )

            logger.info(
                "【No.{pname}】推定結果合成 合成開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            mix_joints = {"color": frame_joints["color"], "joints": {}}
            for fno in tqdm(
                joint_datas[("pt_joints", "Pelvis", "x")].keys(),
            ):
                mix_joints["joints"][fno] = {"body": {}, "left_hand": {}, "right_hand": {}, "face": {}, "2d": {}}

                # 2Dの足情報を設定する
                mix_joints["joints"][fno]["2d"] = frame_joints["estimation"][str(fno)]["ap_joints"]

                if not (
                    ("mp_body_world_joints", "Pelvis", "x") in joint_datas
                    and fno in joint_datas[("mp_body_world_joints", "Pelvis", "x")]
                    and ("mp_body_world_joints", "LAnkle", "x") in joint_datas
                    and fno in joint_datas[("mp_body_world_joints", "LAnkle", "x")]
                    and ("mp_body_world_joints", "RAnkle", "x") in joint_datas
                    and fno in joint_datas[("mp_body_world_joints", "RAnkle", "x")]
                    and fno in hip_ys["L"]
                    and fno in knee_ys["L"]
                    and fno in ankle_ys["L"]
                ):
                    continue

                min_ankle_y = np.min([ankle_ys["L"][fno], ankle_ys["R"][fno]])
                max_hip_y = np.max([hip_ys["L"][fno], hip_ys["R"][fno]])
                adjust_y = mean_ankle_y
                if (max_hip_y - min(mean_ankle_y, min_ankle_y)) < median_leg_length:
                    # 足から足首中央までの長さ が足の長さの中央値一定より短い場合、どっちかの足は接地してるとみなす
                    if ankle_ys["R"][fno] <= ankle_ys["L"][fno]:
                        # 右足のが低い場合、右足で接地してると見なして、右足首のY位置に合わせる
                        adjust_y = ankle_ys["R"][fno]
                    elif ankle_ys["R"][fno] > ankle_ys["L"][fno]:
                        # 左足のが低い場合、左足で接地してると見なして、左足首のY位置に合わせる
                        adjust_y = ankle_ys["L"][fno]

                for joint_name in BODY_LANDMARKS:
                    if not (
                        ("pt_joints", joint_name, "x") in joint_datas
                        and ("pt_joints", joint_name, "y") in joint_datas
                        and ("pt_joints", joint_name, "z") in joint_datas
                        and fno in joint_datas[("pt_joints", joint_name, "x")]
                        and fno in joint_datas[("pt_joints", joint_name, "y")]
                        and fno in joint_datas[("pt_joints", joint_name, "z")]
                    ):
                        continue

                    mix_joints["joints"][fno]["body"][joint_name] = {
                        "x": joint_datas[("pt_joints", joint_name, "x")][fno] - all_root_pos.x,
                        "y": max(0, joint_datas[("pt_joints", joint_name, "y")][fno] - adjust_y),
                        "z": joint_datas[("pt_joints", joint_name, "z")][fno] - all_root_pos.z,
                    }

                    if 10 < mix_joints["joints"][fno]["body"][joint_name]["y"] and (
                        "Ankle" in joint_name or "Heel" in joint_name or "FootIndex" in joint_name
                    ):
                        mix_joints["joints"][fno]["body"][joint_name]["y"] *= 1.4

                for jname in [
                    "Pelvis2",
                    "Spine",
                    "LCollar",
                    "RCollar",
                ]:
                    mix_joints["joints"][fno]["body"][jname] = {}

                for axis in ["x", "y", "z"]:
                    # 下半身先
                    mix_joints["joints"][fno]["body"]["Pelvis2"][axis] = np.mean(
                        [
                            mix_joints["joints"][fno]["body"]["LHip"][axis],
                            mix_joints["joints"][fno]["body"]["RHip"][axis],
                            mix_joints["joints"][fno]["body"]["LKnee"][axis],
                            mix_joints["joints"][fno]["body"]["RKnee"][axis],
                        ]
                    )

                    # 上半身
                    mix_joints["joints"][fno]["body"]["Spine"][axis] = mix_joints["joints"][fno]["body"]["Pelvis"][axis]

                    # 左肩
                    mix_joints["joints"][fno]["body"]["LCollar"][axis] = np.mean(
                        [
                            mix_joints["joints"][fno]["body"]["Neck"][axis],
                            mix_joints["joints"][fno]["body"]["LShoulder"][axis],
                        ]
                    )

                    # 右肩
                    mix_joints["joints"][fno]["body"]["RCollar"][axis] = np.mean(
                        [
                            mix_joints["joints"][fno]["body"]["Neck"][axis],
                            mix_joints["joints"][fno]["body"]["RShoulder"][axis],
                        ]
                    )

                    for direction in ("L", "R"):
                        if (
                            fno in joint_datas[("mp_body_world_joints", f"{direction}Wrist", axis)]
                            and fno in joint_datas[("mp_body_world_joints", f"{direction}Pinky", axis)]
                            and fno in joint_datas[("mp_body_world_joints", f"{direction}Index", axis)]
                            and fno in joint_datas[("mp_body_world_joints", f"{direction}Thumb", axis)]
                            and fno in joint_datas[("mp_body_world_joints", f"{direction}Wrist", "score")]
                            and joint_datas[("mp_body_world_joints", f"{direction}Wrist", "score")][fno] > 0.9
                        ):
                            if f"{direction}Pinky" not in mix_joints["joints"][fno]["body"]:
                                mix_joints["joints"][fno]["body"][f"{direction}Pinky"] = {}
                                mix_joints["joints"][fno]["body"][f"{direction}Index"] = {}
                                mix_joints["joints"][fno]["body"][f"{direction}Thumb"] = {}

                            wrist_diff = (
                                joint_datas[("mp_body_world_joints", f"{direction}Wrist", axis)][fno]
                                - mix_joints["joints"][fno]["body"][f"{direction}Wrist"][axis]
                            )

                            # 小指
                            mix_joints["joints"][fno]["body"][f"{direction}Pinky"][axis] = (
                                joint_datas[("mp_body_world_joints", f"{direction}Pinky", axis)][fno] - wrist_diff
                            )

                            # 人差し指
                            mix_joints["joints"][fno]["body"][f"{direction}Index"][axis] = (
                                joint_datas[("mp_body_world_joints", f"{direction}Index", axis)][fno] - wrist_diff
                            )

                            # 親指
                            mix_joints["joints"][fno]["body"][f"{direction}Thumb"][axis] = (
                                joint_datas[("mp_body_world_joints", f"{direction}Thumb", axis)][fno] - wrist_diff
                            )

                        if (
                            fno in joint_datas[("mp_body_world_joints", f"{direction}Ankle", axis)]
                            and fno in joint_datas[("mp_body_world_joints", f"{direction}Heel", axis)]
                            and fno in joint_datas[("mp_body_world_joints", f"{direction}FootIndex", axis)]
                            and fno in joint_datas[("mp_body_world_joints", f"{direction}Ankle", "score")]
                            and joint_datas[("mp_body_world_joints", f"{direction}Ankle", "score")][fno] > 0.9
                        ):
                            if f"{direction}Heel" not in mix_joints["joints"][fno]["body"]:
                                mix_joints["joints"][fno]["body"][f"{direction}Heel"] = {}
                                mix_joints["joints"][fno]["body"][f"{direction}FootIndex"] = {}

                            ankle_diff = (
                                joint_datas[("mp_body_world_joints", f"{direction}Ankle", axis)][fno]
                                - mix_joints["joints"][fno]["body"][f"{direction}Ankle"][axis]
                            )

                            # 足かかと
                            mix_joints["joints"][fno]["body"][f"{direction}Heel"][axis] = (
                                joint_datas[("mp_body_world_joints", f"{direction}Heel", axis)][fno] - ankle_diff
                            )

                            # 足人差し指
                            mix_joints["joints"][fno]["body"][f"{direction}FootIndex"][axis] = (
                                joint_datas[("mp_body_world_joints", f"{direction}FootIndex", axis)][fno] - ankle_diff
                            )

                        if (
                            fno in joint_datas[("mp_body_world_joints", "Neck", axis)]
                            and fno in joint_datas[("mp_body_world_joints", "Nose", axis)]
                            and fno in joint_datas[("mp_body_world_joints", f"{direction}Ear", axis)]
                            and fno in joint_datas[("mp_body_world_joints", f"{direction}Eye", axis)]
                        ):
                            if "Nose" not in mix_joints["joints"][fno]["body"]:
                                mix_joints["joints"][fno]["body"]["Nose"] = {}

                            if f"{direction}Ear" not in mix_joints["joints"][fno]["body"]:
                                mix_joints["joints"][fno]["body"][f"{direction}Ear"] = {}
                                mix_joints["joints"][fno]["body"][f"{direction}Eye"] = {}

                            neck_diff = joint_datas[("mp_body_world_joints", "Neck", axis)][fno] - mix_joints["joints"][fno]["body"]["Neck"][axis]

                            # 鼻
                            mix_joints["joints"][fno]["body"]["Nose"][axis] = joint_datas[("mp_body_world_joints", "Nose", axis)][fno] - neck_diff

                            # 耳
                            mix_joints["joints"][fno]["body"][f"{direction}Ear"][axis] = (
                                joint_datas[("mp_body_world_joints", f"{direction}Ear", axis)][fno] - neck_diff
                            )

                            # 目
                            mix_joints["joints"][fno]["body"][f"{direction}Eye"][axis] = (
                                joint_datas[("mp_body_world_joints", f"{direction}Eye", axis)][fno] - neck_diff
                            )

                pt_leg_ys = [
                    mix_joints["joints"][fno]["body"].get("LAnkle", {}).get("y", 99999),
                    mix_joints["joints"][fno]["body"].get("RAnkle", {}).get("y", 99999),
                    mix_joints["joints"][fno]["body"].get("LHeel", {}).get("y", 99999),
                    mix_joints["joints"][fno]["body"].get("RHeel", {}).get("y", 99999),
                    mix_joints["joints"][fno]["body"].get("LFootIndex", {}).get("y", 99999),
                    mix_joints["joints"][fno]["body"].get("RFootIndex", {}).get("y", 99999),
                ]
                pt_leg_y = min(0, np.min(pt_leg_ys))

                for jname, jvals in mix_joints["joints"][fno]["body"].items():
                    jvals["y"] -= pt_leg_y

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
