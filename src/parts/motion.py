import json
import os
from datetime import datetime
from glob import glob

import numpy as np
from base.bezier import create_interpolation, get_infections, get_y_infections
from base.logger import MLogger
from base.math import MMatrix4x4, MQuaternion, MVector3D
from mmd.pmx.reader import PmxReader
from mmd.vmd.collection import VmdMotion
from mmd.vmd.part import VmdBoneFrame
from mmd.vmd.writer import VmdWriter
from tqdm import tqdm

from parts.config import DirName

logger = MLogger(__name__)

# 身長158cmプラグインより
MIKU_CM = 0.1259496


def execute(args):
    try:
        logger.info(
            "モーション生成処理開始: {img_dir}",
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

        if not os.path.exists(os.path.join(args.img_dir, DirName.MIX.value)):
            logger.error(
                "指定されたMixディレクトリが存在しません。\nMixが完了していない可能性があります。: {img_dir}",
                img_dir=os.path.join(args.img_dir, DirName.MIX.value),
                decoration=MLogger.DECORATION_BOX,
            )
            return False

        motion_dir_path = os.path.join(args.img_dir, DirName.MOTION.value)
        os.makedirs(motion_dir_path, exist_ok=True)

        # トレース用モデルを読み込む
        trace_model = PmxReader().read_by_filepath(args.trace_rot_model_config)

        for person_file_path in sorted(glob(os.path.join(args.img_dir, DirName.MIX.value, "*.json"))):
            pname, _ = os.path.splitext(os.path.basename(person_file_path))

            logger.info(
                "【No.{pname}】モーション結果位置計算開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            trace_abs_mov_motion = VmdMotion()
            trace_2d_motion = VmdMotion()
            trace_abs_hand_mov_motion = VmdMotion()

            mix_data = {}
            with open(person_file_path, "r", encoding="utf-8") as f:
                mix_data = json.load(f)

            if "joints" not in mix_data:
                continue

            start_fno = end_fno = 0
            for iidx, (fidx, frames) in enumerate(tqdm(mix_data["joints"].items(), desc=f"No.{pname} ... ")):
                fno = int(fidx)
                if iidx == 0:
                    start_fno = fno

                for jname, joint in frames["body"].items():
                    if jname not in PMX_CONNECTIONS:
                        continue
                    bf = VmdBoneFrame(name=PMX_CONNECTIONS[jname], index=fno)
                    bf.position = MVector3D(
                        float(joint["x"]) * MIKU_CM,
                        float(joint["y"]) * MIKU_CM,
                        float(joint["z"]) * MIKU_CM,
                    )
                    trace_abs_mov_motion.bones.append(bf)

                for jname, joint in frames["2d"].items():
                    if jname not in PMX_CONNECTIONS:
                        continue
                    bf = VmdBoneFrame(name=PMX_CONNECTIONS[jname], index=fno)
                    bf.position = MVector3D(
                        float(joint["x"]),
                        float(joint["y"]),
                        0,
                    )
                    trace_2d_motion.bones.append(bf)

                if args.hand_motion:
                    for direction, jp_direction in (("left", "左"), ("right", "右")):
                        for jname, joint in frames[f"{direction}_hand"].items():
                            if jname not in PMX_HAND_CONNECTIONS:
                                continue
                            bf = VmdBoneFrame(name=PMX_HAND_CONNECTIONS[jname].format(direction=jp_direction), index=fno)
                            bf.position = MVector3D(
                                float(joint["x"]),
                                float(joint["y"]),
                                float(joint["z"]),
                            )
                            trace_abs_hand_mov_motion.bones.append(bf)

                if fno > end_fno:
                    end_fno = fno

            trace_org_motion = VmdMotion()

            logger.info(
                "【No.{pname}】モーション(回転)計算開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            with tqdm(
                total=((len(VMD_CONNECTIONS) + (len(VMD_HAND_CONNECTIONS) if args.hand_motion else 0)) * (end_fno - start_fno)),
                desc=f"No.{pname} ... ",
            ) as pchar:

                dest_connections = [(VMD_CONNECTIONS, trace_abs_mov_motion)]
                if args.hand_motion:
                    dest_connections.append((VMD_HAND_CONNECTIONS, trace_abs_hand_mov_motion))

                for target_connections, dest_motion in dest_connections:
                    for target_bone_name, vmd_params in target_connections.items():
                        if "direction" not in vmd_params:
                            continue
                        direction_from_name = vmd_params["direction"][0]
                        direction_to_name = vmd_params["direction"][1]
                        up_from_name = vmd_params["up"][0]
                        up_to_name = vmd_params["up"][1]
                        cross_from_name = vmd_params["cross"][0] if "cross" in vmd_params else vmd_params["direction"][0]
                        cross_to_name = vmd_params["cross"][1] if "cross" in vmd_params else vmd_params["direction"][1]
                        cancel_names = vmd_params["cancel"]

                        for mov_bf in dest_motion.bones[target_bone_name]:
                            if mov_bf.index not in dest_motion.bones[direction_from_name] or mov_bf.index not in dest_motion.bones[direction_to_name]:
                                # キーがない場合、スルーする
                                pchar.update(1)
                                continue

                            bone_direction = (
                                trace_model.bones[direction_to_name].position - trace_model.bones[direction_from_name].position
                            ).normalized()

                            bone_up = (trace_model.bones[up_to_name].position - trace_model.bones[up_from_name].position).normalized()

                            bone_cross = (trace_model.bones[cross_to_name].position - trace_model.bones[cross_from_name].position).normalized()

                            bone_cross_vec: MVector3D = bone_up.cross(bone_cross).normalized()

                            initial_qq = MQuaternion.from_direction(bone_direction, bone_cross_vec)

                            direction_from_abs_pos = dest_motion.bones[direction_from_name][mov_bf.index].position

                            direction_to_abs_pos = dest_motion.bones[direction_to_name][mov_bf.index].position

                            direction: MVector3D = (direction_to_abs_pos - direction_from_abs_pos).normalized()

                            up_from_abs_pos = dest_motion.bones[up_from_name][mov_bf.index].position

                            up_to_abs_pos = dest_motion.bones[up_to_name][mov_bf.index].position

                            up: MVector3D = (up_to_abs_pos - up_from_abs_pos).normalized()

                            cross_from_abs_pos = dest_motion.bones[cross_from_name][mov_bf.index].position

                            cross_to_abs_pos = dest_motion.bones[cross_to_name][mov_bf.index].position

                            cross: MVector3D = (cross_to_abs_pos - cross_from_abs_pos).normalized()

                            motion_cross_vec: MVector3D = up.cross(cross).normalized()

                            motion_qq = MQuaternion.from_direction(direction, motion_cross_vec)

                            cancel_qq = MQuaternion()
                            for cancel_name in cancel_names:
                                cancel_qq *= trace_org_motion.bones[cancel_name][mov_bf.index].rotation

                            bf = VmdBoneFrame(name=target_bone_name, index=mov_bf.index)
                            bf.rotation = cancel_qq.inverse() * motion_qq * initial_qq.inverse()
                            trace_org_motion.bones.append(bf)

                            pchar.update(1)

            logger.info(
                "【No.{pname}】モーション(IK)計算開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            bone_abs_poses = {}

            with tqdm(
                total=(2 * (end_fno - start_fno)),
                desc=f"No.{pname} ... ",
            ) as pchar:
                for direction in ["左", "右"]:

                    leg_ik_bone_name = f"{direction}足ＩＫ"
                    leg_bone_name = f"{direction}足"
                    knee_bone_name = f"{direction}ひざ"
                    ankle_bone_name = f"{direction}足首"
                    toe_bone_name = f"{direction}つま先"
                    heel_bone_name = f"{direction}かかと"

                    for bname in [ankle_bone_name, toe_bone_name, heel_bone_name]:
                        bone_abs_poses[bname] = {}

                    toe_link_names = [
                        "全ての親",
                        "センター",
                        "グルーブ",
                        "腰",
                        "下半身",
                        f"腰キャンセル{direction}",
                        leg_bone_name,
                        knee_bone_name,
                        ankle_bone_name,
                        toe_bone_name,
                    ]

                    heel_link_names = [
                        "全ての親",
                        "センター",
                        "グルーブ",
                        "腰",
                        "下半身",
                        f"腰キャンセル{direction}",
                        leg_bone_name,
                        knee_bone_name,
                        ankle_bone_name,
                        heel_bone_name,
                    ]

                    leg_bone_direction = (trace_model.bones[ankle_bone_name].position - trace_model.bones[knee_bone_name].position).normalized()

                    leg_bone_up = (trace_model.bones[toe_bone_name].position - trace_model.bones[ankle_bone_name].position).normalized()

                    leg_bone_cross_vec: MVector3D = leg_bone_up.cross(leg_bone_direction).normalized()

                    leg_initial_qq = MQuaternion.from_direction(leg_bone_direction, leg_bone_cross_vec)

                    for lower_bf in trace_org_motion.bones["下半身"]:
                        fno = lower_bf.index

                        mats = {}
                        for bidx, bname in enumerate(toe_link_names):
                            mat = MMatrix4x4(identity=True)
                            bf = trace_org_motion.bones[bname][fno]
                            # キーフレの相対位置
                            relative_pos = trace_model.bones[bname].position + bf.position
                            if bidx > 0:
                                # 子ボーンの場合、親の位置をさっぴく
                                relative_pos -= trace_model.bones[toe_link_names[bidx - 1]].position

                            mat.translate(relative_pos)
                            mat.rotate(bf.rotation)
                            mats[bname] = mats[toe_link_names[bidx - 1]] * mat if bidx > 0 else mat

                        # 回転から求めた当該モデルの絶対位置
                        knee_abs_pos = mats[knee_bone_name] * MVector3D()
                        ankle_abs_pos = mats[ankle_bone_name] * MVector3D()
                        toe_abs_pos = mats[toe_bone_name] * MVector3D()

                        for bidx, bname in enumerate(heel_link_names):
                            mat = MMatrix4x4(identity=True)
                            bf = trace_org_motion.bones[bname][fno]
                            # キーフレの相対位置
                            relative_pos = trace_model.bones[bname].position + bf.position
                            if bidx > 0:
                                # 子ボーンの場合、親の位置をさっぴく
                                relative_pos -= trace_model.bones[toe_link_names[bidx - 1]].position

                            mat.translate(relative_pos)
                            mat.rotate(bf.rotation)
                            mats[bname] = mats[toe_link_names[bidx - 1]] * mat if bidx > 0 else mat

                        # 回転から求めた当該モデルの絶対位置
                        heel_abs_pos = mats[heel_bone_name] * MVector3D()

                        bone_abs_poses[ankle_bone_name][fno] = ankle_abs_pos
                        bone_abs_poses[toe_bone_name][fno] = toe_abs_pos
                        bone_abs_poses[heel_bone_name][fno] = heel_abs_pos

                        leg_direction_pos = (ankle_abs_pos - knee_abs_pos).normalized()
                        leg_up_pos = (toe_abs_pos - ankle_abs_pos).normalized()
                        leg_cross_pos: MVector3D = leg_up_pos.cross(leg_direction_pos).normalized()

                        leg_ik_qq = MQuaternion.from_direction(leg_direction_pos, leg_cross_pos) * leg_initial_qq.inverse()

                        leg_ik_bf = VmdBoneFrame(name=leg_ik_bone_name, index=fno)
                        leg_ik_bf.position = ankle_abs_pos - trace_model.bones[ankle_bone_name].position
                        leg_ik_bf.rotation = leg_ik_qq
                        trace_org_motion.bones.append(leg_ik_bf)

                        pchar.update(1)

            logger.info(
                "【No.{pname}】モーション(センター)計算開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            for lower_bf in tqdm(trace_abs_mov_motion.bones["下半身"]):
                fno = lower_bf.index

                center_abs_pos: MVector3D = lower_bf.position
                left_ankle_abs_pos: MVector3D = trace_abs_mov_motion.bones["左足首"][lower_bf.index].position
                right_ankle_abs_pos: MVector3D = trace_abs_mov_motion.bones["右足首"][lower_bf.index].position
                left_heel_abs_pos: MVector3D = trace_abs_mov_motion.bones["左かかと"][lower_bf.index].position
                right_heel_abs_pos: MVector3D = trace_abs_mov_motion.bones["右かかと"][lower_bf.index].position
                left_toe_abs_pos: MVector3D = trace_abs_mov_motion.bones["左つま先"][lower_bf.index].position
                right_toe_abs_pos: MVector3D = trace_abs_mov_motion.bones["右つま先"][lower_bf.index].position
                botton_leg_y = np.min([left_heel_abs_pos.y, right_heel_abs_pos.y, left_toe_abs_pos.y, right_toe_abs_pos.y])
                botton_leg_name = str(
                    np.array(["左足首", "右足首", "左かかと", "右かかと", "左つま先", "右つま先"])[
                        np.argmin(
                            [
                                left_ankle_abs_pos.y,
                                right_ankle_abs_pos.y,
                                left_heel_abs_pos.y,
                                right_heel_abs_pos.y,
                                left_toe_abs_pos.y,
                                right_toe_abs_pos.y,
                            ]
                        )
                    ]
                )

                bottom_adjust_y = botton_leg_y - bone_abs_poses[botton_leg_name].get(fno, MVector3D()).y
                # if "足首" not in botton_leg_name:
                #     bottom_adjust_y -= trace_org_motion.bones[f"{botton_leg_name[0]}足ＩＫ"][fno].position.y

                center_relative_pos: MVector3D = (
                    center_abs_pos - trace_model.bones["下半身"].position + (trace_model.bones["グルーブ"].position - trace_model.bones["センター"].position)
                )

                center_pos: MVector3D = center_relative_pos.copy()
                center_pos.y = 0

                center_bf = VmdBoneFrame(name="センター", index=lower_bf.index)
                center_bf.position = center_pos
                trace_org_motion.bones.append(center_bf)

                trace_org_motion.bones["左足ＩＫ"][fno].position += center_pos
                trace_org_motion.bones["右足ＩＫ"][fno].position += center_pos

                groove_pos: MVector3D = center_relative_pos.copy()
                groove_pos.x = 0
                groove_pos.z = 0

                groove_bf = VmdBoneFrame(name="グルーブ", index=lower_bf.index)
                groove_bf.position = groove_pos
                trace_org_motion.bones.append(groove_bf)

                trace_org_motion.bones["左足ＩＫ"][fno].position.y = max(0, trace_org_motion.bones["左足ＩＫ"][fno].position.y + bottom_adjust_y)
                trace_org_motion.bones["右足ＩＫ"][fno].position.y = max(0, trace_org_motion.bones["右足ＩＫ"][fno].position.y + bottom_adjust_y)
                trace_org_motion.bones["グルーブ"][fno].position.y += botton_leg_y

            trace_org_motion_path = os.path.join(motion_dir_path, f"trace_no{pname}_original.vmd")
            logger.info(
                "【No.{pname}】モーション(回転)生成開始【{path}】",
                pname=pname,
                path=os.path.basename(trace_org_motion_path),
                decoration=MLogger.DECORATION_LINE,
            )
            VmdWriter.write(trace_model.name, trace_org_motion, trace_org_motion_path)

            # logger.info(
            #     "【No.{pname}】モーション スムージング準備",
            #     pname=pname,
            #     decoration=MLogger.DECORATION_LINE,
            # )

            # joint_datas = {}
            # joint_params = {}

            # all_connections = [VMD_CONNECTIONS]
            # if args.hand_motion:
            #     all_connections.append(VMD_HAND_CONNECTIONS)

            # # スムージング
            # for fno in tqdm(range(start_fno, end_fno), desc=f"No.{pname} ... "):
            #     # left_leg_ik_bf = trace_org_motion.bones["左足ＩＫ"][fno]
            #     # right_leg_ik_bf = trace_org_motion.bones["右足ＩＫ"][fno]

            #     # # とりあえずマイナスは打ち消す
            #     # min_y = min(left_leg_ik_bf.position.y, right_leg_ik_bf.position.y)
            #     # if min_y < 0:
            #     #     groove_bf.position.y -= min_y

            #     # left_leg_ik_bf.position.y = max(0, left_leg_ik_bf.position.y)
            #     # right_leg_ik_bf.position.y = max(0, right_leg_ik_bf.position.y)

            #     for bone_name in ["センター", "グルーブ", "左足ＩＫ", "右足ＩＫ"]:
            #         if (bone_name, "mov", "x") not in joint_datas:
            #             joint_datas[(bone_name, "mov", "x")] = []
            #             joint_datas[(bone_name, "mov", "y")] = []
            #             joint_datas[(bone_name, "mov", "z")] = []

            #         pos = trace_org_motion.bones[bone_name][fno].position
            #         joint_datas[(bone_name, "mov", "x")].append(pos.x)
            #         joint_datas[(bone_name, "mov", "y")].append(pos.y)
            #         joint_datas[(bone_name, "mov", "z")].append(pos.z)

            #         joint_params[(bone_name, "mov")] = {
            #             "window_lengt": VMD_CONNECTIONS[bone_name]["window_lengt"],
            #             "polyorder": VMD_CONNECTIONS[bone_name]["polyorder"],
            #         }

            #     for target_connections in all_connections:
            #         for bone_name in target_connections.keys():
            #             if (bone_name, "rot", "x") not in joint_datas:
            #                 joint_datas[(bone_name, "rot", "x")] = []
            #                 joint_datas[(bone_name, "rot", "y")] = []
            #                 joint_datas[(bone_name, "rot", "z")] = []
            #                 joint_datas[(bone_name, "rot", "scalar")] = []

            #             rot = trace_org_motion.bones[bone_name][fno].rotation
            #             joint_datas[(bone_name, "rot", "x")].append(rot.x)
            #             joint_datas[(bone_name, "rot", "y")].append(rot.y)
            #             joint_datas[(bone_name, "rot", "z")].append(rot.z)
            #             joint_datas[(bone_name, "rot", "scalar")].append(rot.scalar)

            #             joint_params[(bone_name, "rot")] = {
            #                 "window_lengt": target_connections[bone_name]["window_lengt"],
            #                 "polyorder": target_connections[bone_name]["polyorder"],
            #             }

            # logger.info(
            #     "【No.{pname}】モーション スムージング開始",
            #     pname=pname,
            #     decoration=MLogger.DECORATION_LINE,
            # )

            # # スムージング
            # smooth_joint_datas = {}
            # for (jname, jtype, axis), joints in tqdm(joint_datas.items(), desc=f"No.{pname} ... "):
            #     smooth_joint_datas[(jname, jtype, axis)] = {}
            #     if len(joints) > joint_params[(jname, jtype)]["window_lengt"]:
            #         smooth_joint_datas[(jname, jtype, axis)] = savgol_filter(
            #             joints, window_length=joint_params[(jname, jtype)]["window_lengt"], polyorder=joint_params[(jname, jtype)]["polyorder"]
            #         )
            #     else:
            #         smooth_joint_datas[(jname, jtype, axis)] = joints

            # for jname in ("左足ＩＫ", "右足ＩＫ"):
            #     num = 5
            #     ik_fix_axis_idxs = {}
            #     for axis in ("x", "y", "z"):
            #         avgs = np.convolve(smooth_joint_datas[(jname, "mov", axis)], np.ones(num) / num, "valid")
            #         # 移動平均でデータ数が減るため、前と後ろに同じ値を繰り返しで補填する
            #         fore_n = int((num - 1) / 2)
            #         back_n = num - 1 - fore_n
            #         ik_avgs = np.hstack((np.tile([avgs[0]], fore_n), avgs, np.tile([avgs[-1]], back_n)))

            #     ik_fix_idxs = list(ik_fix_axis_idxs["x"] & ik_fix_axis_idxs["y"] & ik_fix_axis_idxs["z"])

            # logger.info(
            #     "【No.{pname}】モーション スムージング設定",
            #     pname=pname,
            #     decoration=MLogger.DECORATION_LINE,
            # )

            # trace_smooth_motion = VmdMotion()

            # for (jname, jtype, axis), joints in tqdm(smooth_joint_datas.items(), desc=f"No.{pname} ... "):
            #     for fidx, jvalue in enumerate(joints):
            #         fno = start_fno + fidx
            #         bf = trace_smooth_motion.bones[jname][fno]
            #         if jtype == "mov":
            #             if axis == "x":
            #                 bf.position.x = jvalue
            #             elif axis == "y":
            #                 bf.position.y = jvalue
            #             else:
            #                 bf.position.z = jvalue
            #         else:
            #             if axis == "x":
            #                 bf.rotation.x = jvalue
            #             elif axis == "y":
            #                 bf.rotation.y = jvalue
            #             elif axis == "z":
            #                 bf.rotation.z = jvalue
            #             else:
            #                 bf.rotation.scalar = jvalue
            #                 bf.rotation.normalize()
            #         trace_smooth_motion.bones.append(bf)

            # trace_smooth_motion_path = os.path.join(motion_dir_path, f"trace_no{pname}_smooth.vmd")
            # logger.info(
            #     "【No.{pname}】モーション(スムージング)生成開始【{path}】",
            #     pname=pname,
            #     path=os.path.basename(trace_smooth_motion_path),
            #     decoration=MLogger.DECORATION_LINE,
            # )
            # VmdWriter.write(trace_model.name, trace_smooth_motion, trace_smooth_motion_path)

            logger.info(
                "【No.{pname}】モーション 間引き準備",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            trace_thining_motion = VmdMotion()

            # 間引き
            for bone_name in tqdm(trace_org_motion.bones.names()):
                mx_values = []
                my_values = []
                mz_values = []
                rot_values = []
                rot_y_values = []
                for fno in range(start_fno, end_fno):
                    fno = start_fno + fno
                    pos = trace_org_motion.bones[bone_name][fno].position
                    mx_values.append(pos.x)
                    my_values.append(pos.y)
                    mz_values.append(pos.z)
                    rot = trace_org_motion.bones[bone_name][fno].rotation
                    rot_values.append(MQuaternion().dot(rot))
                    rot_y_values.append(rot.to_euler_degrees().y)

                mx_infections = get_infections(mx_values, 0.05, 1)
                my_infections = get_infections(my_values, 0.05, 1)
                mz_infections = get_infections(mz_values, 0.1, 1)
                rot_infections = get_infections(rot_values, 0.001, 3)
                # 体幹はY軸のオイラー角の変位（180度のフリップ）を検出する
                rot_y_infections = get_infections(rot_y_values, 0.1, 1) if bone_name in ["上半身", "下半身"] else []
                # 回転変動も検出する(180度だけだとどっち向きの回転か分からないので)
                rot_y2_infections = get_y_infections(rot_y_values, 110) if bone_name in ["上半身", "下半身"] else []

                infections = list(
                    sorted(
                        list(
                            {start_fno, end_fno}
                            | set(mx_infections)
                            | set(my_infections)
                            | set(mz_infections)
                            | set(rot_infections)
                            | set(rot_y_infections)
                            | set(rot_y2_infections)
                        )
                    )
                )

                for sfno, efno in zip(infections[:-1], infections[1:]):
                    if sfno == infections[0]:
                        start_bf = trace_org_motion.bones[bone_name][sfno]
                    else:
                        start_bf = trace_thining_motion.bones[bone_name][sfno]
                    end_bf = trace_org_motion.bones[bone_name][efno]
                    end_bf.interpolations.translation_x = create_interpolation(mx_values[sfno:efno])
                    end_bf.interpolations.translation_y = create_interpolation(my_values[sfno:efno])
                    end_bf.interpolations.translation_z = create_interpolation(mz_values[sfno:efno])
                    end_bf.interpolations.rotation = create_interpolation(rot_values[sfno:efno])
                    trace_thining_motion.bones.append(start_bf)
                    trace_thining_motion.bones.append(end_bf)

            trace_thining_motion_path = os.path.join(motion_dir_path, f"trace_no{pname}_thining.vmd")
            logger.info(
                "【No.{pname}】モーション(間引き)生成開始【{path}】",
                pname=pname,
                path=os.path.basename(trace_thining_motion_path),
                decoration=MLogger.DECORATION_LINE,
            )
            VmdWriter.write(trace_model.name, trace_thining_motion, trace_thining_motion_path)

        logger.info(
            "モーション結果保存完了: {motion_dir_path}",
            motion_dir_path=motion_dir_path,
            decoration=MLogger.DECORATION_BOX,
        )

        return True
    except Exception as e:
        logger.critical("モーション生成で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


PMX_CONNECTIONS = {
    "Spine": "上半身",
    "Spine2": "上半身2",
    "Neck": "首",
    "Nose": "鼻",
    "Head": "頭",
    "REye": "右目",
    "LEye": "左目",
    "REar": "右耳",
    "LEar": "左耳",
    "Pelvis": "下半身",
    "Pelvis2": "下半身2",
    "LHip": "左足",
    "RHip": "右足",
    "LKnee": "左ひざ",
    "RKnee": "右ひざ",
    "LAnkle": "左足首",
    "RAnkle": "右足首",
    "LFootIndex": "左つま先",
    "RFootIndex": "右つま先",
    "LHeel": "左かかと",
    "RHeel": "右かかと",
    "LCollar": "左肩",
    "RCollar": "右肩",
    "LShoulder": "左腕",
    "RShoulder": "右腕",
    "LElbow": "左ひじ",
    "RElbow": "右ひじ",
    "LWrist": "左手首",
    "RWrist": "右手首",
    "RPinky": "右小指１",
    "LPinky": "左小指１",
    "RIndex": "右人指１",
    "LIndex": "左人指１",
    "RThumb": "右親指０",
    "LThumb": "左親指０",
}


PMX_HAND_CONNECTIONS = {
    "wrist": "{direction}手首",
    # "thumb1": "{direction}親指０",
    "thumb2": "{direction}親指１",
    "thumb3": "{direction}親指２",
    "thumb": "{direction}親指先",
    "index1": "{direction}人指１",
    "index2": "{direction}人指２",
    "index3": "{direction}人指３",
    "index": "{direction}人指先",
    "middle1": "{direction}中指１",
    "middle2": "{direction}中指２",
    "middle3": "{direction}中指３",
    "middle": "{direction}中指先",
    "ring1": "{direction}薬指１",
    "ring2": "{direction}薬指２",
    "ring3": "{direction}薬指３",
    "ring": "{direction}薬指先",
    "pinky1": "{direction}小指１",
    "pinky2": "{direction}小指２",
    "pinky3": "{direction}小指３",
    "pinky": "{direction}小指先",
}

VMD_CONNECTIONS = {
    "センター": {"window_lengt": 7, "polyorder": 2},
    "グルーブ": {"window_lengt": 5, "polyorder": 2},
    "右足ＩＫ": {"window_lengt": 5, "polyorder": 2},
    "左足ＩＫ": {"window_lengt": 5, "polyorder": 2},
    "下半身": {
        "direction": ("下半身", "下半身2"),
        "up": ("左足", "右足"),
        "cancel": (),
        "window_lengt": 7,
        "polyorder": 2,
    },
    "上半身": {
        "direction": ("上半身", "上半身2"),
        "up": ("左腕", "右腕"),
        "cancel": (),
        "window_lengt": 7,
        "polyorder": 2,
    },
    "上半身2": {
        "direction": ("上半身2", "首"),
        "up": ("左腕", "右腕"),
        "cancel": ("上半身",),
        "window_lengt": 7,
        "polyorder": 2,
    },
    "首": {
        "direction": ("首", "頭"),
        "up": ("左腕", "右腕"),
        "cancel": (
            "上半身",
            "上半身2",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "頭": {
        "direction": ("首", "頭"),
        "up": ("左耳", "右耳"),
        "cancel": (
            "上半身",
            "上半身2",
            "首",
        ),
        "window_lengt": 5,
        "polyorder": 4,
    },
    "左肩": {
        "direction": ("左肩", "左腕"),
        "up": ("上半身", "上半身2", "首"),
        "cancel": ("上半身",),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左腕": {
        "direction": ("左腕", "左ひじ"),
        "up": ("左肩", "左腕"),
        "cancel": (
            "上半身",
            "上半身2",
            "左肩",
        ),
        "window_lengt": 5,
        "polyorder": 4,
    },
    "左ひじ": {
        "direction": ("左ひじ", "左手首"),
        "up": ("左腕", "左ひじ"),
        "cancel": (
            "上半身",
            "上半身2",
            "左肩",
            "左腕",
        ),
        "window_lengt": 5,
        "polyorder": 4,
    },
    "左手首": {
        "direction": ("左手首", "左人指１"),
        "up": ("左ひじ", "左手首"),
        "cancel": (
            "上半身",
            "上半身2",
            "左肩",
            "左腕",
            "左ひじ",
        ),
        "window_lengt": 5,
        "polyorder": 4,
    },
    "右肩": {
        "direction": ("右肩", "右腕"),
        "up": ("上半身", "首"),
        "cancel": (
            "上半身",
            "上半身2",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右腕": {
        "direction": ("右腕", "右ひじ"),
        "up": ("右肩", "右腕"),
        "cancel": (
            "上半身",
            "上半身2",
            "右肩",
        ),
        "window_lengt": 5,
        "polyorder": 4,
    },
    "右ひじ": {
        "direction": ("右ひじ", "右手首"),
        "up": ("右腕", "右ひじ"),
        "cancel": (
            "上半身",
            "上半身2",
            "右肩",
            "右腕",
        ),
        "window_lengt": 5,
        "polyorder": 4,
    },
    "右手首": {
        "direction": ("右手首", "右人指１"),
        "up": ("右ひじ", "右手首"),
        "cancel": (
            "上半身",
            "上半身2",
            "右肩",
            "右腕",
            "右ひじ",
        ),
        "window_lengt": 5,
        "polyorder": 4,
    },
    "左足": {
        "direction": ("左足", "左ひざ"),
        "up": ("左足", "右足"),
        "cancel": ("下半身",),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左ひざ": {
        "direction": ("左ひざ", "左足首"),
        "up": ("左足", "左ひざ"),
        "cancel": (
            "下半身",
            "左足",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左足首": {
        "direction": ("左足首", "左つま先"),
        "up": ("左ひざ", "左足首"),
        "cancel": (
            "下半身",
            "左足",
            "左ひざ",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右足": {
        "direction": ("右足", "右ひざ"),
        "up": ("右足", "左足"),
        "cancel": ("下半身",),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右ひざ": {
        "direction": ("右ひざ", "右足首"),
        "up": ("右足", "右ひざ"),
        "cancel": (
            "下半身",
            "右足",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右足首": {
        "direction": ("右足首", "右つま先"),
        "up": ("右ひざ", "右足首"),
        "cancel": (
            "下半身",
            "右足",
            "右ひざ",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
}

VMD_HAND_CONNECTIONS = {
    # "左親指０": {
    #     "direction": ("左親指０", "左親指１"),
    #     "up": ("左手首", "左親指０"),
    #     "cancel": (
    #         "上半身",
    #         "左肩",
    #         "左腕",
    #         "左ひじ",
    #         "左手首",
    #     ),
    #     "window_lengt": 5,
    #     "polyorder": 2,
    # },
    "左親指１": {
        "direction": ("左親指１", "左親指２"),
        "up": ("左手首", "左親指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左親指２": {
        "direction": ("左親指２", "左親指先"),
        "up": ("左人指１", "左小指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
            "左親指１",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左人指１": {
        "direction": ("左人指１", "左人指２"),
        "up": ("左手首", "左人指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左人指２": {
        "direction": ("左人指２", "左人指３"),
        "up": ("左人指１", "左小指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
            "左人指１",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左人指３": {
        "direction": ("左人指３", "左人指先"),
        "up": ("左人指１", "左小指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
            "左人指１",
            "左人指２",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左中指１": {
        "direction": ("左中指１", "左中指２"),
        "up": ("左手首", "左中指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左中指２": {
        "direction": ("左中指２", "左中指３"),
        "up": ("左人指１", "左小指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
            "左中指１",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左中指３": {
        "direction": ("左中指３", "左中指先"),
        "up": ("左人指１", "左小指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
            "左中指１",
            "左中指２",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左薬指１": {
        "direction": ("左薬指１", "左薬指２"),
        "up": ("左手首", "左薬指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左薬指２": {
        "direction": ("左薬指２", "左薬指３"),
        "up": ("左人指１", "左小指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
            "左薬指１",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左薬指３": {
        "direction": ("左薬指３", "左薬指先"),
        "up": ("左人指１", "左小指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
            "左薬指１",
            "左薬指２",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左小指１": {
        "direction": ("左小指１", "左小指２"),
        "up": ("左手首", "左小指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左小指２": {
        "direction": ("左小指２", "左小指３"),
        "up": ("左人指１", "左小指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
            "左小指１",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "左小指３": {
        "direction": ("左小指３", "左小指先"),
        "up": ("左人指１", "左小指１"),
        "cancel": (
            "上半身",
            "左肩",
            "左腕",
            "左ひじ",
            "左手首",
            "左小指１",
            "左小指２",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右親指１": {
        "direction": ("右親指１", "右親指２"),
        "up": ("右手首", "右親指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右親指２": {
        "direction": ("右親指２", "右親指先"),
        "up": ("右人指１", "右小指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
            "右親指１",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右人指１": {
        "direction": ("右人指１", "右人指２"),
        "up": ("右手首", "右人指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右人指２": {
        "direction": ("右人指２", "右人指３"),
        "up": ("右人指１", "右小指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
            "右人指１",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右人指３": {
        "direction": ("右人指３", "右人指先"),
        "up": ("右人指１", "右小指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
            "右人指１",
            "右人指２",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右中指１": {
        "direction": ("右中指１", "右中指２"),
        "up": ("右手首", "右中指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右中指２": {
        "direction": ("右中指２", "右中指３"),
        "up": ("右人指１", "右小指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
            "右中指１",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右中指３": {
        "direction": ("右中指３", "右中指先"),
        "up": ("右人指１", "右小指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
            "右中指１",
            "右中指２",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右薬指１": {
        "direction": ("右薬指１", "右薬指２"),
        "up": ("右手首", "右薬指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右薬指２": {
        "direction": ("右薬指２", "右薬指３"),
        "up": ("右人指１", "右小指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
            "右薬指１",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右薬指３": {
        "direction": ("右薬指３", "右薬指先"),
        "up": ("右人指１", "右小指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
            "右薬指１",
            "右薬指２",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右小指１": {
        "direction": ("右小指１", "右小指２"),
        "up": ("右手首", "右小指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右小指２": {
        "direction": ("右小指２", "右小指３"),
        "up": ("右人指１", "右小指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
            "右小指１",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
    "右小指３": {
        "direction": ("右小指３", "右小指先"),
        "up": ("右人指１", "右小指１"),
        "cancel": (
            "上半身",
            "右肩",
            "右腕",
            "右ひじ",
            "右手首",
            "右小指１",
            "右小指２",
        ),
        "window_lengt": 5,
        "polyorder": 2,
    },
}
