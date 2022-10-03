import argparse
import json
import os
import platform
import random
import shutil
import sys
import time
from glob import glob
from unittest import result

sys.path.append(os.path.abspath(os.path.join(__file__, "../../AlphaPose")))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml  # type: ignore
from AlphaPose.alphapose.models import builder
from AlphaPose.alphapose.utils.detector import DetectionLoader
from AlphaPose.alphapose.utils.writer import DataWriter
# from AlphaPose.detector.yolo_api import YOLODetector
# from AlphaPose.detector.yolo_cfg import cfg as ycfg
from AlphaPose.detector.yolox_api import YOLOXDetector
from AlphaPose.detector.yolox_cfg import cfg as ycfg
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

        max_fno = 0
        personal_datas = {}

        logger.info(
            "AlphaPose 結果分類",
            decoration=MLogger.DECORATION_LINE,
        )

        img = Image.open(glob(os.path.join(argv.inputpath, "*.png"))[0])

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
                    "width": img.size[0],
                    "height": img.size[1],
                },
                "bbox": {
                    "x": json_data["box"][0],
                    "y": json_data["box"][1],
                    "width": json_data["box"][2],
                    "height": json_data["box"][3],
                },
                "2d-keypoints": json_data["keypoints"],
            }

            all_bbox_areas[person_idx].append(float(json_data["box"][2]) * float(json_data["box"][3]))

            if fno > max_fno:
                max_fno = fno

        # BBOX閾値計算
        max_bbox_area_max = np.max([np.max(bbox_areas) for _, bbox_areas in all_bbox_areas.items()])
        min_bbox_area_min = np.min([np.min(bbox_areas) for _, bbox_areas in all_bbox_areas.items()])
        threshold_bbox_area = min_bbox_area_min + (max_bbox_area_max - min_bbox_area_min) * 0.2

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

                        for ppidx, odata in output_datas.items():
                            oefno = list(odata.keys())[-1]
                            for n in range(1, 11):
                                # 続きの場合、少し前のキーで終わってるブロックがあるか確認する
                                if sfno - n <= oefno:
                                    # 最後のキーフレがひとつ前のキーで終わっている場合
                                    ox = odata[oefno]["bbox"]["x"]
                                    oy = odata[oefno]["bbox"]["y"]
                                    ow = odata[oefno]["bbox"]["width"]
                                    oh = odata[oefno]["bbox"]["height"]
                                    ocenter = np.array([ox, oy]) + np.array([ow, oh]) / 2

                                    if np.isclose(pcenter, ocenter, atol=np.array([80, 80])).all():
                                        # 大体同じ中央にあるBBOXがあったら追加（キーフレの拡張はしない。後ろで移動したら大体切り替わる）
                                        person_idx = ppidx
                                        break
                            if 0 <= person_idx:
                                break

                    if 0 > person_idx:
                        # 最終的に求められなかった場合、新規に求める
                        person_idx = len(output_datas)
                        output_datas[person_idx] = {}

                    for fno in range(sfno, efno + 1):
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
        cidxs = np.concatenate([np.where(idxs % 3 == 0)[0], np.where(idxs % 3 == 1)[0], np.where(idxs % 3 == 2)[0]])
        pid_colors_opencv = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in pid_colors[cidxs]]

        output_frames = {}
        for person_idx, result_data in tqdm(result_datas.items()):
            json_data = {
                "color": [
                    float(pid_colors_opencv[person_idx][2]) / 255,
                    float(pid_colors_opencv[person_idx][1]) / 255,
                    float(pid_colors_opencv[person_idx][0]) / 255,
                ],
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
                        "2d-keypoints": rdata["2d-keypoints"],
                    }
                )

        logger.info(
            "AlphaPose 映像出力",
            decoration=MLogger.DECORATION_LINE,
        )

        target_image_path = os.path.join(argv.outputpath, FileName.ALPHAPOSE_IMAGE.value)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(args.img_dir, DirName.ALPHAPOSE.value, FileName.ALPHAPOSE_VIDEO.value),
            fourcc,
            30.0,
            (img.size[0], img.size[1]),
        )

        for fno in tqdm(sorted(output_frames.keys())):
            person_frames = output_frames[fno]
            shutil.copy(os.path.join(argv.inputpath, person_frames[0]["image"]["path"]), target_image_path)

            for person_frame in person_frames:
                # 人物INDEX別に画像に書き込み

                save_2d_image(
                    target_image_path,
                    person_frame["person_idx"],
                    fno,
                    person_frame["2d-keypoints"],
                    person_frame["bbox"],
                    person_frame["color"],
                )

            # 書き込み出力
            out.write(cv2.imread(target_image_path))

        out.release()
        cv2.destroyAllWindows()

        logger.info(
            "2D姿勢推定 結果保存完了: {outputpath}",
            outputpath=argv.outputpath,
            decoration=MLogger.DECORATION_BOX,
        )

        return True
    except Exception as e:
        logger.critical("2D姿勢推定で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


def save_2d_image(image_path: str, person_idx: int, fno: int, keypoints: list, bbox: dict, pid_color: tuple):
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

    # 同じファイルに上書き
    cv2.imwrite(image_path, img)


def get_args_parser():

    """----------------------------- Demo options -----------------------------"""
    parser = argparse.ArgumentParser(description="AlphaPose Demo")
    parser.add_argument(
        "--cfg", type=str, default="AlphaPose/configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml", help="experiment configure file name"
    )
    parser.add_argument("--checkpoint", type=str, default="../data/alphapose/checkpoint/halpe26_fast_res50_256x192.pth", help="checkpoint file name")
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
