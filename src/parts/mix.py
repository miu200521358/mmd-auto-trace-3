import json
import os
import re
from glob import glob

import numpy as np
from base.logger import MLogger
from base.math import MVector3D
from scipy.signal import savgol_filter
from tqdm import tqdm

from parts.config import SMPL_JOINT_24, DirName

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
        all_depths: dict[int, dict[str, float]] = {}
        for personal_json_path in tqdm(sorted(glob(os.path.join(args.img_dir, DirName.POSETRIPLET.value, "*.json")))):
            pname, _ = os.path.splitext(os.path.basename(personal_json_path))

            frame_joints = {}
            with open(personal_json_path, "r") as f:
                frame_joints = json.load(f)

            for fno in frame_joints["estimation"].keys():
                if "pt-keypoints" in frame_joints["estimation"][fno] and "depth" in frame_joints["estimation"][fno]:
                    # Mediapipeの情報がある最初のキーフレを選択する
                    target_pnames.append(pname)
                    break

            if pname not in target_pnames:
                continue

            if fno not in all_depths:
                all_depths[fno] = {}
            # メーターからcmに変換
            all_depths[fno][pname] = frame_joints["estimation"][fno]["depth"] * 100

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
                "【No.{pname}】推定結果合成 グルーブ補正",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            for fno in tqdm(joint_datas["ap", "Pelvis", "y"].keys(), desc=f"No.{pname} ... "):
                # 全身でもっとも低いところを探す(逆立ちとかも可能性としてはあるので、全部舐める)
                min_y = np.min([joint_datas["ap", jname, "y"][fno] for jname in SMPL_JOINT_24.keys()])

                for jname in SMPL_JOINT_24.keys():
                    joint_datas["ap", jname, "y"][fno] -= min_y

            logger.info(
                "【No.{pname}】推定結果合成 合成開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            # PoseTriplet で測った場合のY座標（ジャンプしてるとその分上に行く）
            pt_median_ankle_y = np.median(list(joint_datas["pt", "RAnkle", "y"].values()) + list(joint_datas["pt", "LAnkle", "y"].values()))

            mix_joints = {"color": frame_joints["color"], "joints": {}}
            for fidx, fno in tqdm(
                enumerate(joint_datas[("ap", "Pelvis", "x")].keys()),
            ):
                mix_joints["joints"][fno] = {"body": {}, "2d": {}}

                # # 2Dの足情報を設定する
                # mix_joints["joints"][fno]["2d"] = frame_joints["estimation"][str(fno)]["ap"]

                if not (("ap", "Pelvis", "x") in joint_datas and fno in joint_datas[("ap", "Pelvis", "x")]):
                    continue

                pt_leg_length = abs(joint_datas["pt", "Pelvis", "y"][fno] - pt_median_ankle_y) + 7
                ap_leg_length = abs(
                    joint_datas["ap", "Pelvis", "y"][fno]
                    - np.min(
                        [
                            joint_datas["ap", "RAnkle", "y"][fno],
                            joint_datas["ap", "LAnkle", "y"][fno],
                        ]
                    )
                )

                # ジャンプしてる場合はptの方が値が大きくなるので、＋のみ判定（接地までとする）
                # ややPoseTriplet を低めに見積もってるので、実値としてはかさ増しする
                adjust_foot_y = max(0, pt_leg_length - ap_leg_length) * 1.3
                logger.debug(
                    "[{fno}] adjust_foot_y: {adjust_foot_y}, pt_leg_length: {pt_leg_length}, ap_leg_length: {ap_leg_length}",
                    fno=fno,
                    adjust_foot_y=adjust_foot_y,
                    pt_leg_length=pt_leg_length,
                    ap_leg_length=ap_leg_length,
                )

                for jname in SMPL_JOINT_24.keys():
                    if not (
                        ("ap", jname, "x") in joint_datas
                        and fno in joint_datas[("ap", jname, "x")]
                        and ("pt", "Pelvis", "x") in joint_datas
                        and fno in joint_datas[("pt", "Pelvis", "x")]
                        and ("mp", "depth", "d") in joint_datas
                        and fno in joint_datas[("mp", "depth", "d")]
                    ):
                        continue

                    mix_joints["joints"][fno]["body"][jname] = {
                        "x": joint_datas[("ap", jname, "x")][fno] + (joint_datas[("pt", "Pelvis", "x")][fno] * 1.2),
                        "y": joint_datas[("ap", jname, "y")][fno] + adjust_foot_y,
                        "z": joint_datas[("ap", jname, "z")][fno] + (joint_datas[("mp", "depth", "d")][fno] - root_depth * 1.2),
                    }

                mix_joints["joints"][fno]["body"]["Pelvis2"] = {}
                mix_joints["joints"][fno]["body"]["Spine0"] = {}
                mix_joints["joints"][fno]["body"]["Spine4"] = {}

                for axis in ("x", "y", "z"):
                    # 上半身0
                    mix_joints["joints"][fno]["body"]["Spine0"][axis] = mix_joints["joints"][fno]["body"]["Pelvis"][axis]

                    # 下半身先
                    mix_joints["joints"][fno]["body"]["Pelvis2"][axis] = np.mean(
                        [
                            mix_joints["joints"][fno]["body"]["LHip"][axis],
                            mix_joints["joints"][fno]["body"]["RHip"][axis],
                        ]
                    )

                    # 首根元
                    mix_joints["joints"][fno]["body"]["Spine4"][axis] = np.mean(
                        [
                            mix_joints["joints"][fno]["body"]["Neck"][axis],
                            mix_joints["joints"][fno]["body"]["Spine3"][axis],
                        ]
                    )

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
