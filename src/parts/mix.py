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

            for fno in frame_joints.keys():
                if "pt_joints" in frame_joints[fno] and "mp_body_world_joints" in frame_joints[fno]:
                    # PoseTripletの情報がある最初のキーフレを選択する
                    target_pnames.append(pname)
                    break

            if pname not in target_pnames:
                continue

            if fno not in all_root_poses:
                all_root_poses[fno] = {}
            all_root_poses[fno][pname] = MVector3D(
                frame_joints[fno]["pt_joints"]["Pelvis"]["x"],
                frame_joints[fno]["pt_joints"]["Pelvis"]["y"],
                frame_joints[fno]["pt_joints"]["Pelvis"]["z"],
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
            for fidx, frame_json_data in tqdm(frame_joints.items(), desc=f"No.{pname} ... "):
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

            # 踵の位置を推定

            left_ankle_ys = {}
            for fno, fval in tqdm(joint_datas["pt_joints", "LAnkle", "y"].items(), desc=f"No.{pname} ... "):
                left_ankle_ys[fno] = fval

            right_ankle_ys = {}
            for fno, fval in tqdm(joint_datas["pt_joints", "RAnkle", "y"].items(), desc=f"No.{pname} ... "):
                right_ankle_ys[fno] = fval

            left_ankle_infections = get_infections(list(left_ankle_ys.values()), 0.1, 1)
            right_ankle_infections = get_infections(list(right_ankle_ys.values()), 0.1, 1)
            median_ankle_y = (
                np.median(
                    np.concatenate(
                        [
                            np.array(list(left_ankle_ys.values()))[left_ankle_infections],
                            np.array(list(right_ankle_ys.values()))[right_ankle_infections],
                        ]
                    )
                )
                + 15
            )

            #     if len(joints) > 5:
            #         smooth_vals = savgol_filter(list(joints.values()), window_length=5, polyorder=4)
            #         for fno, fval in zip(joints.keys(), smooth_vals):
            #             smooth_joint_datas[(joint_type, joint_name, axis)][fno] = fval
            #     else:
            #         smooth_joint_datas[(joint_type, joint_name, axis)] = joints

            # logger.info(
            #     "【No.{pname}】関節距離計測",
            #     pname=pname,
            #     decoration=MLogger.DECORATION_LINE,
            # )

            # # 足の全体の長さと個々の関節の長さから直立を見つける
            # mp_leg_foot_lengths = {}
            # mp_leg_ankle_lengths = {}
            # mp_joint_lengths = {}
            # for fno in tqdm(smooth_joint_datas[("pt_joints", "Pelvis", "x")].keys()):
            #     is_target = True
            #     for joint_type, joint_name in (
            #         ("pt_joints", "Pelvis"),
            #         ("mp_body_world_joints", "Pelvis"),
            #         ("mp_body_world_joints", "RHip"),
            #         ("mp_body_world_joints", "RKnee"),
            #         ("mp_body_world_joints", "RAnkle"),
            #         ("mp_body_world_joints", "RHeel"),
            #         ("mp_body_world_joints", "RFootIndex"),
            #         ("mp_body_world_joints", "LHip"),
            #         ("mp_body_world_joints", "LKnee"),
            #         ("mp_body_world_joints", "LAnkle"),
            #         ("mp_body_world_joints", "LHeel"),
            #         ("mp_body_world_joints", "LFootIndex"),
            #     ):
            #         if not (
            #             (joint_type, joint_name, "x") in smooth_joint_datas
            #             and (joint_type, joint_name, "y") in smooth_joint_datas
            #             and (joint_type, joint_name, "z") in smooth_joint_datas
            #             and fno in smooth_joint_datas[(joint_type, joint_name, "x")]
            #             and fno in smooth_joint_datas[(joint_type, joint_name, "y")]
            #             and fno in smooth_joint_datas[(joint_type, joint_name, "z")]
            #         ):
            #             # 計測対象が揃ってない場合、スルー
            #             is_target = False
            #             break

            #     if not is_target:
            #         continue

            #     mp_joints = {}
            #     for joint_name in ("LHip", "LKnee", "LAnkle", "LHeel", "RHip", "RKnee", "RAnkle", "RHeel"):
            #         mp_joints[joint_name] = MVector3D(
            #             smooth_joint_datas[("mp_body_world_joints", joint_name, "x")][fno],
            #             smooth_joint_datas[("mp_body_world_joints", joint_name, "y")][fno],
            #             smooth_joint_datas[("mp_body_world_joints", joint_name, "z")][fno],
            #         )
            #     # 足全体の距離（しゃがんでたら短くなる）
            #     mp_leg_ankle_lengths[fno] = np.sum([mp_joints["LHip"].distance(mp_joints["LHeel"]), mp_joints["RHip"].distance(mp_joints["RHeel"])])
            #     # 足関節自体の距離（しゃがんでも変わらないはず）
            #     mp_joint_lengths[fno] = np.sum(
            #         [
            #             (
            #                 mp_joints["LHip"].distance(mp_joints["LKnee"])
            #                 + mp_joints["LKnee"].distance(mp_joints["LAnkle"])
            #                 + mp_joints["LAnkle"].distance(mp_joints["LHeel"])
            #             ),
            #             (
            #                 mp_joints["RHip"].distance(mp_joints["RKnee"])
            #                 + mp_joints["RKnee"].distance(mp_joints["RAnkle"])
            #                 + mp_joints["RAnkle"].distance(mp_joints["RHeel"])
            #             ),
            #         ]
            #     )
            #     # 足指のY位置（＝腰からの足指の距離）
            #     mp_leg_foot_lengths[fno] = np.sum(
            #         [
            #             smooth_joint_datas[("mp_body_world_joints", "LHeel", "y")][fno],
            #             smooth_joint_datas[("mp_body_world_joints", "RHeel", "y")][fno],
            #             smooth_joint_datas[("mp_body_world_joints", "LFootIndex", "y")][fno],
            #             smooth_joint_datas[("mp_body_world_joints", "RFootIndex", "y")][fno],
            #         ]
            #     )

            # # 地面に対して直立しているキーフレのINDEXを取得する
            # leg_straight_idxs = []
            # for count in range(10, max_fno, 10):
            #     # できるだけ関節距離の合計と足-足指間の距離の合計が等しいキーフレ
            #     leg_ankle_straight_idxs = np.argsort(
            #         np.abs(np.array(list(mp_joint_lengths.values())) - np.array(list(mp_leg_ankle_lengths.values())))
            #     )[:count]
            #     # できるだけ足の長さが長い（直立）キーフレ
            #     leg_ground_straight_idxs = np.argsort(list(mp_leg_foot_lengths.values()))[:count]
            #     leg_straight_idxs = list(set(leg_ankle_straight_idxs) & set(leg_ground_straight_idxs))
            #     if leg_straight_idxs:
            #         leg_straight_idxs = np.array(leg_straight_idxs)
            #         break
            # # 中央値を直立とみなす
            # leg_most_straight_idx = np.argsort(np.array(list(mp_joint_lengths.values()))[leg_straight_idxs])[len(leg_straight_idxs) // 2]
            # # 足の長さが直立に近いキーフレ
            # leg_most_straight_fno = np.array(list(mp_joint_lengths.keys()))[leg_straight_idxs[leg_most_straight_idx]]
            # # 直立時のかかとの位置（＝足を真っ直ぐに伸ばした状態での足の長さ）
            # leg_most_straight_length = abs(
            #     np.min(
            #         [
            #             smooth_joint_datas[("mp_body_world_joints", "LHeel", "y")].get(leg_most_straight_fno, 0),
            #             smooth_joint_datas[("mp_body_world_joints", "RHeel", "y")].get(leg_most_straight_fno, 0),
            #         ]
            #     )
            #     - smooth_joint_datas[("mp_body_world_joints", "Pelvis", "y")].get(leg_most_straight_fno, 0)
            # )
            # leg_most_straight_pelvis_y = smooth_joint_datas[("pt_joints", "Pelvis", "y")][leg_most_straight_fno]

            logger.info(
                "【No.{pname}】推定結果合成 合成開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            mix_joints = {"joints": {}}
            for fno in tqdm(
                joint_datas[("pt_joints", "Pelvis", "x")].keys(),
            ):
                mix_joints["joints"][fno] = {"body": {}, "left_hand": {}, "right_hand": {}, "face": {}, "2d": {}}

                # 2Dの足情報を設定する
                mix_joints["joints"][fno]["2d"] = frame_joints[str(fno)]["ap_joints"]

                if not (
                    ("mp_body_world_joints", "Pelvis", "x") in joint_datas
                    and ("mp_body_world_joints", "Pelvis", "y") in joint_datas
                    and ("mp_body_world_joints", "Pelvis", "z") in joint_datas
                    and fno in joint_datas[("mp_body_world_joints", "Pelvis", "x")]
                    and fno in joint_datas[("mp_body_world_joints", "Pelvis", "y")]
                    and fno in joint_datas[("mp_body_world_joints", "Pelvis", "z")]
                    and ("mp_body_world_joints", "LAnkle", "x") in joint_datas
                    and ("mp_body_world_joints", "LAnkle", "y") in joint_datas
                    and ("mp_body_world_joints", "LAnkle", "z") in joint_datas
                    and fno in joint_datas[("mp_body_world_joints", "LAnkle", "x")]
                    and fno in joint_datas[("mp_body_world_joints", "LAnkle", "y")]
                    and fno in joint_datas[("mp_body_world_joints", "LAnkle", "z")]
                    and ("mp_body_world_joints", "RAnkle", "x") in joint_datas
                    and ("mp_body_world_joints", "RAnkle", "y") in joint_datas
                    and ("mp_body_world_joints", "RAnkle", "z") in joint_datas
                    and fno in joint_datas[("mp_body_world_joints", "RAnkle", "x")]
                    and fno in joint_datas[("mp_body_world_joints", "RAnkle", "y")]
                    and fno in joint_datas[("mp_body_world_joints", "RAnkle", "z")]
                ):
                    continue

                # min(
                #     leg_most_straight_length,
                #     leg_length,
                # ) + min(0, smooth_joint_datas[("pt_joints", "Pelvis", "y")][fno] - leg_most_straight_pelvis_y)

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
                        "y": joint_datas[("pt_joints", joint_name, "y")][fno] - median_ankle_y,
                        "z": joint_datas[("pt_joints", joint_name, "z")][fno] - all_root_pos.z,
                    }

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
                            and joint_datas[("mp_body_world_joints", f"{direction}Wrist", "score")][fno] > 0.8
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
                            and joint_datas[("mp_body_world_joints", f"{direction}Ankle", "score")][fno] > 0.8
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