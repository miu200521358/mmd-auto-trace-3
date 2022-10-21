import argparse
import json
import os
import platform
import shutil
import sys
import time
from datetime import datetime
from glob import glob

sys.path.append(os.path.abspath(os.path.join(__file__, "../../AlphaPose")))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml  # type: ignore
from AlphaPose.alphapose.models import builder
from AlphaPose.alphapose.utils.detector import DetectionLoader
from AlphaPose.alphapose.utils.writer_smpl import DataWriterSMPL
from AlphaPose.detector.yolox_api import YOLOXDetector
from AlphaPose.detector.yolox_cfg import cfg as ycfg
from AlphaPose.trackers import track
from AlphaPose.trackers.tracker_api import Tracker
from AlphaPose.trackers.tracker_cfg import cfg as tcfg
from base.exception import MApplicationException
from base.logger import MLogger
from easydict import EasyDict as edict
from PIL import Image
from tqdm import tqdm

from parts.config import SMPL_JOINT_29, DirName, FileName

# from AlphaPose.detector.yolo_api import YOLODetector
# from AlphaPose.detector.yolo_cfg import cfg as ycfg


logger = MLogger(__name__)


def execute(args):
    logger.info(
        "AlphaPose 開始: {img_dir}",
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

    try:
        parser = get_args_parser()
        argv = parser.parse_args(args=[])
        cfg = update_config(argv.cfg)

        if platform.system() == "Windows":
            argv.qsize = 1024
            argv.sp = True
            argv.detbatch = 10
            argv.posebatch = 64

        argv.inputpath = os.path.join(args.img_dir, DirName.FRAMES.value)
        argv.outputpath = os.path.join(args.img_dir, DirName.ALPHAPOSE.value)

        if os.path.exists(argv.outputpath):
            rename_dir_path = f"{argv.outputpath}_{datetime.fromtimestamp(os.stat(argv.outputpath).st_ctime).strftime('%Y%m%d_%H%M%S')}"
            os.rename(argv.outputpath, rename_dir_path)
            os.makedirs(argv.outputpath)
            if os.path.exists(os.path.join(rename_dir_path, FileName.ALPHAPOSE_RESULT.value)):
                shutil.copy(
                    os.path.join(rename_dir_path, FileName.ALPHAPOSE_RESULT.value), os.path.join(argv.outputpath, FileName.ALPHAPOSE_RESULT.value)
                )
        else:
            os.makedirs(argv.outputpath)

        argv.gpus = [int(i) for i in argv.gpus.split(",")] if torch.cuda.device_count() >= 1 else [-1]
        argv.device = torch.device("cuda:" + str(argv.gpus[0]) if argv.gpus[0] >= 0 else "cpu")
        argv.detbatch = argv.detbatch * len(argv.gpus)
        argv.posebatch = argv.posebatch * len(argv.gpus)
        argv.tracking = argv.pose_track or argv.pose_flow or argv.detector == "tracker"

        if not argv.sp:
            torch.multiprocessing.set_start_method("forkserver", force=True)
            torch.multiprocessing.set_sharing_strategy("file_system")

        input_source = [os.path.basename(file_path) for file_path in sorted(glob(os.path.join(argv.inputpath, "*.png")))]
        result_path = os.path.join(argv.outputpath, FileName.ALPHAPOSE_RESULT.value)

        if not os.path.exists(result_path):
            logger.info(
                "学習モデル準備開始: {checkpoint}",
                checkpoint=argv.checkpoint,
                decoration=MLogger.DECORATION_LINE,
            )

            ycfg.MODEL_NAME = "yolox-x"
            ycfg.MODEL_WEIGHTS = "../data/alphapose/detector/yolox/yolox_x.pth"
            detector = YOLOXDetector(ycfg, argv)
            # ycfg.CONFIG = "AlphaPose/detector/yolo/cfg/yolov3-spp.cfg"
            # ycfg.WEIGHTS = "../data/alphapose/detector/yolo/yolov3-spp.weights"
            # detector = YOLODetector(ycfg, argv)
            det_loader = DetectionLoader(input_source, detector, cfg, argv, batchSize=argv.detbatch, mode="image", queueSize=argv.qsize)
            det_loader.start()

            # Load pose model
            pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
            pose_model.load_state_dict(torch.load(argv.checkpoint, map_location=argv.device))
            tcfg.loadmodel = (
                "../data/alphapose/tracker/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
            )
            tracker = Tracker(tcfg, argv)

            if len(argv.gpus) > 1:
                pose_model = torch.nn.DataParallel(pose_model, device_ids=argv.gpus).to(argv.device)
            else:
                pose_model.to(argv.device)
            pose_model.eval()

            writer = DataWriterSMPL(cfg, argv, save_video=False, queueSize=argv.qsize).start()

            logger.info(
                "AlphaPose 開始",
                decoration=MLogger.DECORATION_LINE,
            )

            with torch.no_grad():
                for _ in tqdm(input_source, dynamic_ncols=True):
                    (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()
                    if orig_img is None:
                        break
                    if boxes is None or boxes.nelement() == 0:
                        writer.save(None, None, None, None, None, orig_img, im_name)
                        continue
                    # Pose Estimation
                    inps = inps.to(argv.device)
                    pose_output = pose_model(inps)

                    old_ids = torch.arange(boxes.shape[0]).long()
                    _, _, ids, new_ids, _ = track(tracker, argv, orig_img, inps, boxes, old_ids, cropped_boxes, im_name, scores)
                    new_ids = new_ids.long()

                    boxes = boxes[new_ids]
                    cropped_boxes = cropped_boxes[new_ids]
                    scores = scores[new_ids]

                    smpl_output = {
                        "pred_uvd_jts": pose_output.pred_uvd_jts.cpu()[new_ids],
                        "maxvals": pose_output.maxvals.cpu()[new_ids],
                        "transl": pose_output.transl.cpu()[new_ids],
                        "pred_vertices": pose_output.pred_vertices.cpu()[new_ids],
                        "pred_xyz_jts_24": pose_output.pred_xyz_jts_24_struct.cpu()[new_ids] * 2,  # convert to meters
                    }

                    writer.save(boxes, scores, ids, smpl_output, cropped_boxes, orig_img, im_name)

            running_str = "."
            while writer.running():
                time.sleep(1)
                logger.info(
                    "Rendering {running}",
                    running=running_str,
                    decoration=MLogger.DECORATION_LINE,
                )
                running_str += "."

            writer.stop()
            det_loader.stop()

        logger.info(
            "AlphaPose 結果分類準備",
            decoration=MLogger.DECORATION_LINE,
        )

        json_datas = {}
        with open(result_path, "r") as f:
            json_datas = json.load(f)

        max_fno = 0
        personal_datas = {}

        logger.info(
            "AlphaPose 結果分類",
            decoration=MLogger.DECORATION_LINE,
        )

        img = Image.open(sorted(glob(os.path.join(argv.inputpath, "*.png")))[0])

        all_bbox_areas = {}
        for json_data in tqdm(json_datas):
            # 人物INDEX別に保持
            person_idx = int(json_data["idx"])
            fno = int(json_data["image_id"].replace(".png", ""))

            if person_idx not in personal_datas:
                personal_datas[person_idx] = {}
            if person_idx not in all_bbox_areas:
                all_bbox_areas[person_idx] = []

            personal_datas[person_idx][fno] = {
                "image": {
                    "path": os.path.join(argv.inputpath, json_data["image_id"]),
                },
                "bbox": {
                    "x": json_data["box"][0],
                    "y": json_data["box"][1],
                    "width": json_data["box"][2],
                    "height": json_data["box"][3],
                },
                "ap-2d-keypoints": json_data["keypoints"],
                "ap-3d-keypoints": json_data["pred_xyz_jts"],
            }

            all_bbox_areas[person_idx].append(float(json_data["box"][2]) * float(json_data["box"][3]))

            if fno > max_fno:
                max_fno = fno

        # 明らかに遠景の人物を除外する
        threshold_bbox_area = (img.size[0] / 10) * (img.size[1] / 10)

        all_extras_bboxs = {}
        for person_idx, personal_data in tqdm(personal_datas.items()):
            # 連続していないキーフレのペアリスト
            extract_fnos = np.array(
                sorted(
                    np.concatenate(
                        [
                            [list(personal_data.keys())[0]],
                            np.array(list(personal_data.keys()))[np.where(np.diff(np.array(list(personal_data.keys()))) > 1)[0]],
                            np.array(list(personal_data.keys()))[np.where(np.diff(np.array(list(personal_data.keys()))) > 1)[0] + 1],
                            [list(personal_data.keys())[-1]],
                        ]
                    ).tolist()
                )
            )
            for sfno, efno in zip(extract_fnos[:-1:2], extract_fnos[1::2]):
                if (sfno, efno) not in all_extras_bboxs:
                    all_extras_bboxs[(sfno, efno)] = {}
                all_extras_bboxs[(sfno, efno)][person_idx] = []
                for fno in range(sfno, efno + 1):
                    if fno in personal_data:
                        all_extras_bboxs[(sfno, efno)][person_idx].append(personal_data[fno]["bbox"]["width"] * personal_data[fno]["bbox"]["height"])

        output_datas = {}
        for sfno, efno in tqdm(sorted(all_extras_bboxs.keys())):
            for pidx, extras_bboxs in all_extras_bboxs[sfno, efno].items():
                if np.median(extras_bboxs) > threshold_bbox_area:
                    person_idx = -1
                    if sfno == 0:
                        # 最初はそのまま登録
                        person_idx = len(output_datas)
                        output_datas[person_idx] = {}
                    else:
                        # 移植対象のBBOXの中心を求める
                        px = personal_datas[pidx][sfno]["bbox"]["x"]
                        py = personal_datas[pidx][sfno]["bbox"]["y"]
                        pw = personal_datas[pidx][sfno]["bbox"]["width"]
                        ph = personal_datas[pidx][sfno]["bbox"]["height"]
                        pcenter = np.array([px, py]) + np.array([pw, ph]) / 2

                        # Pelvisの位置
                        phx = personal_datas[pidx][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["Pelvis"] * 3]
                        phy = personal_datas[pidx][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["Pelvis"] * 3 + 1]
                        phip = np.array([phx, phy])

                        # LAnkleの位置
                        plax = personal_datas[pidx][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["LAnkle"] * 3]
                        play = personal_datas[pidx][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["LAnkle"] * 3 + 1]
                        plaip = np.array([plax, play])

                        # RAnkleの位置
                        prax = personal_datas[pidx][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["RAnkle"] * 3]
                        pray = personal_datas[pidx][sfno]["ap-2d-keypoints"][SMPL_JOINT_29["RAnkle"] * 3 + 1]
                        praip = np.array([prax, pray])

                        ocenters = {}
                        ohips = {}
                        olaips = {}
                        oraips = {}
                        ofnos = {}
                        for ppidx, odata in output_datas.items():
                            oefno = list(odata.keys())[-1]
                            for n in range(1, 10):
                                # 続きの場合、少し前のキーで終わってるブロックがあるか確認する
                                if sfno - n <= oefno:
                                    # 最後のキーフレがひとつ前のキーで終わっている場合
                                    ox = odata[oefno]["bbox"]["x"]
                                    oy = odata[oefno]["bbox"]["y"]
                                    ow = odata[oefno]["bbox"]["width"]
                                    oh = odata[oefno]["bbox"]["height"]
                                    ocenter = np.array([ox, oy]) + np.array([ow, oh]) / 2

                                    # Pelvisの位置
                                    ohx = odata[oefno]["ap-2d-keypoints"][SMPL_JOINT_29["Pelvis"] * 3]
                                    ohy = odata[oefno]["ap-2d-keypoints"][SMPL_JOINT_29["Pelvis"] * 3 + 1]
                                    ohip = np.array([ohx, ohy])

                                    # LAnkleの位置
                                    olax = odata[oefno]["ap-2d-keypoints"][SMPL_JOINT_29["LAnkle"] * 3]
                                    olay = odata[oefno]["ap-2d-keypoints"][SMPL_JOINT_29["LAnkle"] * 3 + 1]
                                    olaip = np.array([olax, olay])

                                    # RAnkleの位置
                                    orax = odata[oefno]["ap-2d-keypoints"][SMPL_JOINT_29["RAnkle"] * 3]
                                    oray = odata[oefno]["ap-2d-keypoints"][SMPL_JOINT_29["RAnkle"] * 3 + 1]
                                    oraip = np.array([orax, oray])

                                    offset = 5 * n

                                    if (
                                        np.isclose(pcenter, ocenter, atol=np.array([64 + offset, 36 + offset])).all()
                                        or np.isclose(phip, ohip, atol=np.array([64 + offset, 36 + offset])).all()
                                        or np.isclose(plaip, olaip, atol=np.array([64 + offset, 36 + offset])).all()
                                        or np.isclose(praip, oraip, atol=np.array([64 + offset, 36 + offset])).all()
                                    ):
                                        # 大体同じ位置にあるBBOXがあったら検討対象
                                        ocenters[ppidx] = ocenter
                                        ohips[ppidx] = ohip
                                        olaips[ppidx] = olaip
                                        oraips[ppidx] = oraip
                                        ofnos[ppidx] = np.array([oefno, oefno])
                                        break

                        if ocenters:
                            # BBOXの中央とHipの位置から最も近いppidxを選ぶ
                            person_idx = np.array(list(ocenters.keys()))[
                                np.argmin(
                                    np.sum(
                                        np.hstack(
                                            [
                                                np.abs(np.array(list(ocenters.values())) - pcenter),
                                                np.abs(np.array(list(ohips.values())) - phip),
                                                np.abs(np.array(list(olaips.values())) - plaip),
                                                np.abs(np.array(list(oraips.values())) - praip),
                                                np.abs(np.array(list(ofnos.values())) - sfno) * 100,
                                            ]
                                        ),
                                        axis=1,
                                    )
                                )
                            ]

                    if 0 > person_idx or (person_idx in output_datas and sfno in output_datas[person_idx]):
                        # 最終的に求められなかった場合、もしくは既に割り当て済みの場合、新規に求める
                        person_idx = len(output_datas)
                        output_datas[person_idx] = {}

                    for fno in range(sfno, efno + 1):
                        if fno in personal_datas[pidx]:
                            output_datas[person_idx][fno] = personal_datas[pidx][fno]

        logger.info(
            "AlphaPose 結果保存",
            decoration=MLogger.DECORATION_LINE,
        )

        # 一定以上ある場合はOK
        result_datas = {}
        for pidx, odatas in output_datas.items():
            if len(odatas) > 5:
                result_datas[len(result_datas)] = odatas

        # 追跡画像用色生成
        cmap = plt.get_cmap("gist_rainbow")
        pid_colors = np.array([cmap(i) for i in np.linspace(0, 1, len(result_datas))])
        idxs = np.arange(len(result_datas))
        # 適当にばらけさせる
        cidxs = np.concatenate(
            [
                np.where(idxs % 5 == 0)[0],
                np.where(idxs % 5 == 1)[0][::-1],
                np.where(idxs % 5 == 2)[0],
                np.where(idxs % 5 == 3)[0][::-1],
                np.where(idxs % 5 == 4)[0],
            ]
        )
        pid_colors_opencv = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in pid_colors[cidxs]]

        output_frames = {}
        for person_idx, result_data in tqdm(result_datas.items()):
            json_data = {
                "color": [
                    float(pid_colors_opencv[person_idx][2]) / 255,
                    float(pid_colors_opencv[person_idx][1]) / 255,
                    float(pid_colors_opencv[person_idx][0]) / 255,
                ],
                "image": {
                    "width": img.size[0],
                    "height": img.size[1],
                },
                "estimation": result_data,
            }

            with open(os.path.join(args.img_dir, DirName.ALPHAPOSE.value, f"{(person_idx + 1):03d}.json"), "w") as f:
                json.dump(json_data, f, indent=4)

            for fno, rdata in result_data.items():
                if fno not in output_frames:
                    output_frames[fno] = []
                output_frames[fno].append(
                    {
                        "color": pid_colors_opencv[person_idx],
                        "person_idx": person_idx,
                        "image": rdata["image"],
                        "bbox": rdata["bbox"],
                        "ap-2d-keypoints": rdata["ap-2d-keypoints"],
                    }
                )

        logger.info(
            "AlphaPose 映像出力",
            decoration=MLogger.DECORATION_LINE,
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(args.img_dir, DirName.ALPHAPOSE.value, FileName.ALPHAPOSE_VIDEO.value),
            fourcc,
            30.0,
            (img.size[0], img.size[1]),
        )

        for file_path in tqdm(sorted(glob(os.path.join(argv.inputpath, "*.png")))):
            fno = int(os.path.basename(file_path).split(".")[0])

            if fno in output_frames:
                # キーフレ内に人物が検出されている場合、描画
                img = save_2d_image(
                    cv2.imread(file_path),
                    fno,
                    output_frames[fno],
                )
            else:
                # 人物がいない場合、そのまま読み込み
                img = cv2.imread(file_path)

            # 書き込み出力
            out.write(img)

        out.release()
        cv2.destroyAllWindows()

        logger.info(
            "AlphaPose 結果保存完了: {outputpath}",
            outputpath=argv.outputpath,
            decoration=MLogger.DECORATION_BOX,
        )

        return True
    except Exception as e:
        logger.critical("AlphaPose で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        raise e


def save_2d_image(img, fno: int, person_frames: list):
    # alphapose\models\layers\smpl\SMPL.py
    SKELETONS = [
        (SMPL_JOINT_29["Pelvis"], SMPL_JOINT_29["Spine1"]),
        (SMPL_JOINT_29["Spine1"], SMPL_JOINT_29["Spine2"]),
        (SMPL_JOINT_29["Spine2"], SMPL_JOINT_29["Spine3"]),
        (SMPL_JOINT_29["Spine3"], SMPL_JOINT_29["Neck"]),
        (SMPL_JOINT_29["Neck"], SMPL_JOINT_29["Jaw"]),
        (SMPL_JOINT_29["Jaw"], SMPL_JOINT_29["Head"]),
        (SMPL_JOINT_29["Pelvis"], SMPL_JOINT_29["LHip"]),
        (SMPL_JOINT_29["LHip"], SMPL_JOINT_29["LKnee"]),
        (SMPL_JOINT_29["LKnee"], SMPL_JOINT_29["LAnkle"]),
        (SMPL_JOINT_29["LAnkle"], SMPL_JOINT_29["LFoot"]),
        (SMPL_JOINT_29["LAnkle"], SMPL_JOINT_29["LBigtoe"]),
        (SMPL_JOINT_29["Spine3"], SMPL_JOINT_29["LCollar"]),
        (SMPL_JOINT_29["LCollar"], SMPL_JOINT_29["LShoulder"]),
        (SMPL_JOINT_29["LShoulder"], SMPL_JOINT_29["LElbow"]),
        (SMPL_JOINT_29["LElbow"], SMPL_JOINT_29["LWrist"]),
        (SMPL_JOINT_29["LWrist"], SMPL_JOINT_29["LThumb"]),
        (SMPL_JOINT_29["LWrist"], SMPL_JOINT_29["LMiddle"]),
        (SMPL_JOINT_29["Pelvis"], SMPL_JOINT_29["RHip"]),
        (SMPL_JOINT_29["RHip"], SMPL_JOINT_29["RKnee"]),
        (SMPL_JOINT_29["RKnee"], SMPL_JOINT_29["RAnkle"]),
        (SMPL_JOINT_29["RAnkle"], SMPL_JOINT_29["RFoot"]),
        (SMPL_JOINT_29["RAnkle"], SMPL_JOINT_29["RBigtoe"]),
        (SMPL_JOINT_29["Spine3"], SMPL_JOINT_29["RCollar"]),
        (SMPL_JOINT_29["RCollar"], SMPL_JOINT_29["RShoulder"]),
        (SMPL_JOINT_29["RShoulder"], SMPL_JOINT_29["RElbow"]),
        (SMPL_JOINT_29["RElbow"], SMPL_JOINT_29["RWrist"]),
        (SMPL_JOINT_29["RWrist"], SMPL_JOINT_29["RThumb"]),
        (SMPL_JOINT_29["RWrist"], SMPL_JOINT_29["RMiddle"]),
    ]

    for person_data in person_frames:
        person_idx = person_data["person_idx"]
        keypoints = person_data["ap-2d-keypoints"]
        bbox = person_data["bbox"]
        pid_color = person_data["color"]

        kps = np.array(keypoints).reshape(-1, 3)

        for j1, j2 in SKELETONS:
            joint1_x = int(kps[j1, 0])
            joint1_y = int(kps[j1, 1])
            joint2_x = int(kps[j2, 0])
            joint2_y = int(kps[j2, 1])

            t = 2
            r = 6
            cv2.line(
                img,
                (joint1_x, joint1_y),
                (joint2_x, joint2_y),
                color=tuple(pid_color),
                thickness=t,
            )
            cv2.circle(
                img,
                thickness=-1,
                center=(joint1_x, joint1_y),
                radius=r,
                color=tuple(pid_color),
            )
            cv2.circle(
                img,
                thickness=-1,
                center=(joint2_x, joint2_y),
                radius=r,
                color=tuple(pid_color),
            )

        bbox_x = int(bbox["x"])
        bbox_y = int(bbox["y"])
        bbox_w = int(bbox["width"])
        bbox_h = int(bbox["height"])
        bbx_thick = 3
        cv2.line(
            img,
            (bbox_x, bbox_y),
            (bbox_x + bbox_w, bbox_y),
            color=tuple(pid_color),
            thickness=bbx_thick,
        )
        cv2.line(
            img,
            (bbox_x, bbox_y),
            (bbox_x, bbox_y + bbox_h),
            color=tuple(pid_color),
            thickness=bbx_thick,
        )
        cv2.line(
            img,
            (bbox_x + bbox_w, bbox_y),
            (bbox_x + bbox_w, bbox_y + bbox_h),
            color=tuple(pid_color),
            thickness=bbx_thick,
        )
        cv2.line(
            img,
            (bbox_x, bbox_y + bbox_h),
            (bbox_x + bbox_w, bbox_y + bbox_h),
            color=tuple(pid_color),
            thickness=bbx_thick,
        )

        cv2.putText(
            img,
            f"{(person_idx + 1):03d}",
            (bbox_x + bbox_w // 3, bbox_y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color=tuple(pid_color),
            thickness=bbx_thick,
        )

        cv2.putText(
            img,
            f"{(person_idx + 1):03d}",
            ((bbox_x + bbox_w) + 5, (bbox_y + bbox_h) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color=tuple(pid_color),
            thickness=bbx_thick,
        )

        cv2.putText(
            img,
            f"{fno:6d}F",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color=(182, 0, 182),
            thickness=bbx_thick,
        )

    return img


def get_args_parser():

    """----------------------------- Demo options -----------------------------"""
    parser = argparse.ArgumentParser(description="AlphaPose Demo")
    parser.add_argument(
        "--cfg",
        type=str,
        default="../data/alphapose/config/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml",
        help="experiment configure file name",
    )
    parser.add_argument("--checkpoint", type=str, default="../data/alphapose/checkpoint/pretrained_w_cam.pth", help="checkpoint file name")

    parser.add_argument("--sp", default=False, action="store_true", help="Use single process for pytorch")
    parser.add_argument("--detector", dest="detector", help="detector name", default="yolox")
    parser.add_argument("--detfile", dest="detfile", help="detection result file", default="")
    parser.add_argument("--indir", dest="inputpath", help="image-directory", default="")
    parser.add_argument("--list", dest="inputlist", help="image-list", default="")
    parser.add_argument("--image", dest="inputimg", help="image-name", default="")
    parser.add_argument("--outdir", dest="outputpath", help="output-directory", default="examples/res/")
    parser.add_argument("--save_img", default=False, action="store_true", help="save result as image")
    parser.add_argument("--vis", default=False, action="store_true", help="visualize image")
    parser.add_argument("--showbox", default=False, action="store_true", help="visualize human bbox")
    parser.add_argument("--show_skeleton", default=True, action="store_true", help="visualize 3d human skeleton")
    parser.add_argument("--profile", default=False, action="store_true", help="add speed profiling at screen output")
    parser.add_argument("--format", type=str, help="save in the format of cmu or coco or openpose, option: coco/cmu/open")
    parser.add_argument("--min_box_area", type=int, default=0, help="min box area to filter out")
    parser.add_argument("--detbatch", type=int, default=30, help="detection batch size PER GPU")
    parser.add_argument("--posebatch", type=int, default=256, help="pose estimation maximum batch size PER GPU")
    parser.add_argument(
        "--eval",
        dest="eval",
        default=False,
        action="store_true",
        help="save the result json as coco format, using image index(int) instead of image name(str)",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        dest="gpus",
        default="0",
        help="choose which cuda device to use by index and input comma to use multi gpus, e.g. 0,1,2,3. (input -1 for cpu only)",
    )
    parser.add_argument(
        "--qsize", type=int, dest="qsize", default=4, help="the length of result buffer, where reducing it will lower requirement of cpu memory"
    )
    parser.add_argument("--flip", default=False, action="store_true", help="enable flip testing")
    parser.add_argument("--debug", default=False, action="store_true", help="print detail information")
    """----------------------------- Video options -----------------------------"""
    parser.add_argument("--video", dest="video", help="video-name", default="")
    parser.add_argument("--webcam", dest="webcam", type=int, help="webcam number", default=-1)
    parser.add_argument("--save_video", dest="save_video", help="whether to save rendered video", default=False, action="store_true")
    parser.add_argument("--vis_fast", dest="vis_fast", help="use fast rendering", action="store_true", default=False)
    """----------------------------- Tracking options -----------------------------"""
    parser.add_argument("--pose_flow", dest="pose_flow", help="track humans in video with PoseFlow", action="store_true", default=False)
    parser.add_argument("--pose_track", dest="pose_track", help="track humans in video with reid", action="store_true", default=True)

    return parser


def update_config(config_file):
    with open(config_file) as f:
        config = edict(yaml.load(f, Loader=yaml.FullLoader))
        return config
