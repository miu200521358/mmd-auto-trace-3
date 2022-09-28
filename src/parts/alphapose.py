import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from glob import glob

sys.path.append(os.path.abspath(os.path.join(__file__, "../../AlphaPose")))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml  # type: ignore
from AlphaPose.alphapose.models import builder
from AlphaPose.alphapose.utils.detector import DetectionLoader
from AlphaPose.alphapose.utils.writer import DataWriter
from AlphaPose.detector.apis import get_detector
from AlphaPose.detector.tracker_api import Tracker as DTracker
from AlphaPose.detector.tracker_cfg import cfg as dcfg
from AlphaPose.trackers import track
from AlphaPose.trackers.tracker_api import Tracker
from AlphaPose.trackers.tracker_cfg import cfg as tcfg
from base.logger import MLogger
from easydict import EasyDict as edict
from PIL import Image
from tqdm import tqdm

from parts.config import DirName, FileName

logger = MLogger(__name__)


def execute(args):
    try:
        logger.info(
            "2D姿勢推定 開始: {img_dir}",
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

        parser = get_args_parser()
        argv = parser.parse_args(args=[])
        cfg = update_config(argv.cfg)

        if platform.system() == "Windows":
            argv.sp = True

        argv.inputpath = os.path.join(args.img_dir, DirName.FRAMES.value)
        argv.outputpath = os.path.join(args.img_dir, DirName.ALPHAPOSE.value)
        os.makedirs(argv.outputpath, exist_ok=True)

        argv.gpus = [int(i) for i in argv.gpus.split(",")] if torch.cuda.device_count() >= 1 else [-1]
        argv.device = torch.device("cuda:" + str(argv.gpus[0]) if argv.gpus[0] >= 0 else "cpu")
        argv.detbatch = argv.detbatch * len(argv.gpus)
        argv.posebatch = argv.posebatch * len(argv.gpus)
        argv.tracking = argv.pose_track or argv.pose_flow or argv.detector == "tracker"

        if not argv.sp:
            torch.multiprocessing.set_start_method("forkserver", force=True)
            torch.multiprocessing.set_sharing_strategy("file_system")

        input_source = [os.path.basename(file_path) for file_path in glob(os.path.join(argv.inputpath, "*.png"))]

        logger.info(
            "学習モデル準備開始: {checkpoint}",
            checkpoint=argv.checkpoint,
            decoration=MLogger.DECORATION_LINE,
        )

        dcfg.CONFIG = "AlphaPose/detector/tracker/cfg/yolov3.cfg"
        dcfg.WEIGHTS = "../data/alphapose/pretrained_models/jde.1088x608.uncertainty.pt"
        det_loader = DetectionLoader(input_source, DTracker(dcfg, argv), cfg, argv, batchSize=argv.detbatch, mode="image", queueSize=argv.qsize)
        det_loader.start()

        # Load pose model
        pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
        pose_model.load_state_dict(torch.load(argv.checkpoint, map_location=argv.device))
        tcfg.loadmodel = (
            "../data/alphapose/pretrained_models/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
        )
        tracker = Tracker(tcfg, argv)

        if len(argv.gpus) > 1:
            pose_model = torch.nn.DataParallel(pose_model, device_ids=argv.gpus).to(argv.device)
        else:
            pose_model.to(argv.device)
        pose_model.eval()

        writer = DataWriter(cfg, argv, save_video=False, queueSize=argv.qsize).start()

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
                datalen = inps.size(0)
                leftover = 0
                if (datalen) % argv.posebatch:
                    leftover = 1
                num_batches = datalen // argv.posebatch + leftover
                hm = []
                for j in range(num_batches):
                    inps_j = inps[j * argv.posebatch : min((j + 1) * argv.posebatch, datalen)]
                    hm_j = pose_model(inps_j)
                    hm.append(hm_j)
                hm = torch.cat(hm)
                boxes, scores, ids, hm, cropped_boxes = track(tracker, argv, orig_img, inps, boxes, hm, cropped_boxes, im_name, scores)
                hm = hm.cpu()
                writer.save(boxes, scores, ids, hm, cropped_boxes, orig_img, im_name)

        running_str = "."
        while writer.running():
            time.sleep(1)
            logger.info(
                "Rendering {running}",
                running=running_str,
                decoration=MLogger.DECORATION_LINE,
            )
            running_str += "."

        logger.info(
            "AlphaPose 結果分類準備",
            decoration=MLogger.DECORATION_LINE,
        )

        writer.stop()
        det_loader.stop()

        json_datas = {}
        with open(os.path.join(argv.outputpath, FileName.ALPHAPOSE_RESULT.value), "r") as f:
            json_datas = json.load(f)

        # person_idx を数える
        person_idxs = []
        for json_data in tqdm(json_datas):
            person_idxs.append(json_data["idx"])

        # 追跡画像用色生成
        random.seed(13)
        pids = list(set(person_idxs))
        cmap = plt.get_cmap("rainbow")
        pid_colors = [cmap(i) for i in np.linspace(0, 1, len(pids))]
        random.shuffle(pid_colors)
        pid_colors_opencv = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in pid_colors]

        target_image_path = os.path.join(argv.outputpath, FileName.ALPHAPOSE_IMAGE.value)
        prev_image_id = ""
        max_fno = 0
        personal_datas = {}

        img = Image.open(glob(os.path.join(argv.inputpath, "*.png"))[0])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(args.img_dir, DirName.ALPHAPOSE.value, FileName.ALPHAPOSE_VIDEO.value),
            fourcc,
            30.0,
            (img.size[0], img.size[1]),
        )

        logger.info(
            "AlphaPose 結果分類",
            decoration=MLogger.DECORATION_LINE,
        )

        for json_data in tqdm(json_datas):
            # 人物INDEX別に保持
            person_idx = int(json_data["idx"])
            if person_idx not in personal_datas:
                personal_datas[person_idx] = {}

            kps = json_data["keypoints"]
            kps = np.array(kps).reshape(-1, 3)[:17, :2]
            fno = int(json_data["image_id"].replace(".png", ""))
            personal_datas[person_idx][fno] = {
                "image_id": json_data["image_id"],
                "2d-keypoints": json_data["keypoints"],
                "bbox": json_data["box"],
            }

            if prev_image_id != json_data["image_id"]:
                # 前の画像IDが入ってる場合、動画出力
                if prev_image_id:
                    out.write(cv2.imread(target_image_path))
                # 前と画像が違う場合、1枚だけコピー
                shutil.copy(os.path.join(argv.inputpath, json_data["image_id"]), target_image_path)

            save_2d_image(
                target_image_path,
                person_idx,
                fno,
                json_data["keypoints"],
                json_data["box"],
                pid_colors_opencv[person_idx - 1],
            )

            prev_image_id = json_data["image_id"]
            if fno > max_fno:
                max_fno = fno

        # 最後の1枚を出力
        out.write(cv2.imread(target_image_path))

        out.release()
        cv2.destroyAllWindows()

        logger.info(
            "AlphaPose 結果保存",
            decoration=MLogger.DECORATION_LINE,
        )

        for person_idx, personal_data in personal_datas.items():
            with open(os.path.join(args.img_dir, DirName.ALPHAPOSE.value, f"{person_idx:03d}.json"), "w") as f:
                json.dump(personal_data, f, indent=4)

        logger.info(
            "2D姿勢推定 結果保存完了: {outputpath}",
            outputpath=argv.outputpath,
            decoration=MLogger.DECORATION_BOX,
        )

        return True
    except Exception as e:
        logger.critical("2D姿勢推定で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


def save_2d_image(image_path: str, person_idx: int, fno: int, keypoints: list, bbox: list, pid_color: tuple):
    img = cv2.imread(image_path)
    kps = np.array(keypoints).reshape(-1, 3)

    # https://github.com/Fang-Haoshu/Halpe-FullBody
    SKELETONS = [
        (19, 11),  # Hip, LHip
        (19, 12),  # Hip, RHip
        (19, 18),  # Hip, Neck
        (18, 5),  # Neck, LShoulder
        (18, 6),  # Neck, RShoulder
        (18, 0),  # Neck, Nose
        (0, 1),  # Nose, LEye
        (0, 2),  # Nose, REye
        (0, 17),  # Nose, Head
        (1, 3),  # LEye, LEar
        (2, 4),  # REye, REar
        (5, 7),  # LShoulder, LElbow
        (7, 9),  # LElbow, LWrist
        (6, 8),  # RShoulder, RElbow
        (8, 10),  # RElbow, RWrist
        (11, 13),  # LHip, LKnee
        (13, 15),  # LKnee, LAnkle
        (12, 14),  # RHip, Rknee
        (14, 16),  # Rknee, RAnkle
        (15, 20),  # LAnkle, LBigToe
        (15, 22),  # LAnkle, LSmallToe
        (15, 24),  # LAnkle, LHeel
        (16, 21),  # RAnkle, RBigToe
        (16, 23),  # RAnkle, RSmallToe
        (16, 25),  # RAnkle, RHeel
    ]

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

    bbox_x = int(bbox[0])
    bbox_y = int(bbox[1])
    bbox_w = int(bbox[2])
    bbox_h = int(bbox[3])
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
        f"{person_idx:03d}",
        (bbox_x + bbox_w // 3, bbox_y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        color=tuple(pid_color),
        thickness=bbx_thick,
    )

    cv2.putText(
        img,
        f"{person_idx:03d}",
        ((bbox_x + bbox_w) + bbox_w // 3, (bbox_y + bbox_h) - 5),
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
        thickness=2,
    )

    # 同じファイルに上書き
    cv2.imwrite(image_path, img)


def get_args_parser():

    """----------------------------- Demo options -----------------------------"""
    parser = argparse.ArgumentParser(description="AlphaPose Demo")
    parser.add_argument(
        "--cfg", type=str, default="AlphaPose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml", help="experiment configure file name"
    )
    parser.add_argument(
        "--checkpoint", type=str, default="../data/alphapose/pretrained_models/halpe26_fast_res50_256x192.pth", help="checkpoint file name"
    )
    parser.add_argument("--sp", default=False, action="store_true", help="Use single process for pytorch")
    parser.add_argument("--detector", dest="detector", help="detector name", default="tracker")
    parser.add_argument("--detfile", dest="detfile", help="detection result file", default="")
    parser.add_argument("--indir", dest="inputpath", help="image-directory", default="")
    parser.add_argument("--list", dest="inputlist", help="image-list", default="")
    parser.add_argument("--image", dest="inputimg", help="image-name", default="")
    parser.add_argument("--outdir", dest="outputpath", help="output-directory", default="examples/res/")
    parser.add_argument("--save_img", default=False, action="store_true", help="save result as image")
    parser.add_argument("--vis", default=False, action="store_true", help="visualize image")
    parser.add_argument("--showbox", default=False, action="store_true", help="visualize human bbox")
    parser.add_argument("--profile", default=False, action="store_true", help="add speed profiling at screen output")
    parser.add_argument("--format", type=str, help="save in the format of cmu or coco or openpose, option: coco/cmu/open")
    parser.add_argument("--min_box_area", type=int, default=0, help="min box area to filter out")
    parser.add_argument("--detbatch", type=int, default=5, help="detection batch size PER GPU")
    parser.add_argument("--posebatch", type=int, default=64, help="pose estimation maximum batch size PER GPU")
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
        "--qsize", type=int, dest="qsize", default=128, help="the length of result buffer, where reducing it will lower requirement of cpu memory"
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