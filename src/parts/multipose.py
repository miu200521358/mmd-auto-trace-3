# -*- coding: utf-8 -*-
import json
import os
from datetime import datetime
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
from base.logger import MLogger
from MultiPose.lib.models import networktcn
from TorchSUL import Model as M
from tqdm import tqdm

from parts.config import SMPL_JOINT_29, DirName, FileName

logger = MLogger(__name__)


def execute(args):
    try:
        logger.info(
            "MultiPose 推定処理開始: {img_dir}",
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

        output_dir_path = os.path.join(args.img_dir, DirName.MULTIPOSE.value)

        if os.path.exists(output_dir_path):
            os.rename(output_dir_path, f"{output_dir_path}_{datetime.fromtimestamp(os.stat(output_dir_path).st_ctime).strftime('%Y%m%d_%H%M%S')}")

        os.makedirs(output_dir_path)

        seq_len = 243
        nettcn = networktcn.Refine2dNet(17, seq_len, input_dimension=2, output_dimension=1, output_pts=1)
        x_dumb = torch.zeros(2, 243, 17 * 2)
        nettcn(x_dumb)
        M.Saver(nettcn).restore("../data/3d-multi-pose-pose/ckpts/model_root/")
        nettcn.cuda()
        nettcn.eval()

        for persion_json_path in glob(os.path.join(args.img_dir, DirName.ALPHAPOSE.value, "*.json")):
            if FileName.ALPHAPOSE_RESULT.value in persion_json_path:
                continue

            json_datas = {}
            with open(persion_json_path, "r") as f:
                json_datas = json.load(f)

            pname, _ = os.path.splitext(os.path.basename(persion_json_path))

            logger.info(
                "【No.{pname}】MultiPose 推定開始",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            # キーフレ番号
            fnos = [int(sfno) for sfno in json_datas["estimation"].keys()]
            # 関節の位置
            fp2ds = []

            for fno in tqdm(
                fnos,
                desc=f"No.{pname} ... ",
            ):
                frame_json_data = json_datas["estimation"][str(fno)]

                kps = np.array(frame_json_data["ap-2d-keypoints"]).reshape(-1, 3)
                # SMPL -> H36M
                kps2 = kps[
                    (
                        SMPL_JOINT_29["Pelvis"],
                        SMPL_JOINT_29["LHip"],
                        SMPL_JOINT_29["LKnee"],
                        SMPL_JOINT_29["LAnkle"],
                        SMPL_JOINT_29["RHip"],
                        SMPL_JOINT_29["RKnee"],
                        SMPL_JOINT_29["RAnkle"],
                        SMPL_JOINT_29["Spine1"],
                        SMPL_JOINT_29["Neck"],
                        SMPL_JOINT_29["Jaw"],
                        SMPL_JOINT_29["Head"],
                        SMPL_JOINT_29["LShoulder"],
                        SMPL_JOINT_29["LElbow"],
                        SMPL_JOINT_29["LWrist"],
                        SMPL_JOINT_29["RShoulder"],
                        SMPL_JOINT_29["RElbow"],
                        SMPL_JOINT_29["RWrist"],
                    ),
                    :,
                ]

                fp2ds.append(kps2[:, :2])

            p2d = np.array(fp2ds, dtype=np.float32)
            p2d = torch.from_numpy(p2d).cuda() / 915

            with torch.no_grad():
                p2d = p2d.unsqueeze(0).unsqueeze(0)
                p2d = F.pad(p2d, (0, 0, 0, 0, seq_len // 2, seq_len // 2), mode="replicate")
                p2d = p2d.squeeze()
                pred = nettcn.evaluate(p2d)
                pred = pred.cpu().numpy()

            for fidx, fno in tqdm(
                enumerate(fnos),
                desc=f"No.{pname} ... ",
            ):
                json_datas["estimation"][str(fno)]["depth"] = float(pred[fidx])

            mediapipe_json_path = os.path.join(output_dir_path, os.path.basename(persion_json_path))

            logger.info(
                "【No.{pname}】MultiPose 推定結果出力",
                pname=pname,
                decoration=MLogger.DECORATION_LINE,
            )

            with open(mediapipe_json_path, "w") as f:
                json.dump(json_datas, f, indent=4)

        logger.info(
            "MultiPose 推定処理終了: {output_dir}",
            output_dir=output_dir_path,
            decoration=MLogger.DECORATION_BOX,
        )

        return True
    except Exception as e:
        logger.critical("MultiPose で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False
