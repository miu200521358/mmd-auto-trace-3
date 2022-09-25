import argparse
import json
import os
import platform
import random
import sys
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
from AlphaPose.detector.yolo_api import YOLODetector
from AlphaPose.detector.yolo_cfg import cfg as ycfg
from AlphaPose.trackers import track
from AlphaPose.trackers.tracker_api import Tracker
from AlphaPose.trackers.tracker_cfg import cfg as tcfg
from base.logger import MLogger
from easydict import EasyDict as edict
from PIL import Image
from tqdm import tqdm

logger = MLogger(__name__)


def execute(args):
    try:
        logger.info(
            "2D人物姿勢推定開始: {img_dir}",
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

        argv.inputpath = os.path.join(args.img_dir, "01_frames")
        argv.outputpath = os.path.join(args.img_dir, "02_alphapose")
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

        ycfg.CONFIG = "AlphaPose/detector/yolo/cfg/yolov3-spp.cfg"
        ycfg.WEIGHTS = "../data/alphapose/yolo/yolov3-spp.weights"
        det_loader = DetectionLoader(input_source, YOLODetector(ycfg, argv), cfg, argv, batchSize=argv.detbatch, mode="image", queueSize=argv.qsize)
        det_worker = det_loader.start()

        # Load pose model
        pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

        logger.info(
            "学習モデル準備開始: {checkpoint}",
            checkpoint=argv.checkpoint,
            decoration=MLogger.DECORATION_LINE,
        )
        pose_model.load_state_dict(torch.load(argv.checkpoint, map_location=argv.device))
        tcfg.loadmodel = (
            "../data/alphapose/trackers/weights/osnet_ain_x1_0_msmt17_256x128_amsgrad_ep50_lr0.0015_coslr_b64_fb10_softmax_labsmth_flip_jitter.pth"
        )
        tracker = Tracker(tcfg, argv)

        if len(argv.gpus) > 1:
            pose_model = torch.nn.DataParallel(pose_model, device_ids=argv.gpus).to(argv.device)
        else:
            pose_model.to(argv.device)
        pose_model.eval()

        writer = DataWriter(cfg, argv, save_video=False, queueSize=argv.qsize).start()

        logger.info(
            "AlphaPose開始",
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

        writer.stop()
        det_loader.stop()

        # model, criterion, postprocessors = build_model(argv)
        # if argv.resume and os.path.exists(argv.resume):
        #     checkpoint = torch.load(argv.resume, map_location="cpu")
        #     model.load_state_dict(checkpoint["model"])
        # else:
        #     logger.error(
        #         "指定された学習モデルが存在しません\n{resume}",
        #         resume=argv.resume,
        #         decoration=MLogger.DECORATION_BOX,
        #     )
        #     return False
        # model.eval()

        # logger.info(
        #     "学習モデル準備完了: {resume}",
        #     resume=argv.resume,
        #     decoration=MLogger.DECORATION_LINE,
        # )

        # for frame_dir in sorted(glob(os.path.join(args.img_dir, "frames", "*"))):
        #     dir_name = os.path.basename(frame_dir)
        #     argv.data_dir = frame_dir
        #     argv.outdir = os.path.join(args.img_dir, "snipper", os.path.basename(frame_dir))
        #     os.makedirs(argv.outdir, exist_ok=True)

        #     logger.info(
        #         "【No.{dir_name}】snipper姿勢推定開始",
        #         dir_name=dir_name,
        #         decoration=MLogger.DECORATION_LINE,
        #     )

        #     all_samples, frame_indices, all_filenames = get_all_samples(argv)  # snippet of images

        #     results = []
        #     with torch.set_grad_enabled(False):  # deactivate autograd to reduce memory usage
        #         for samples in tqdm(all_samples):
        #             imgs = samples["imgs"].to(device).unsqueeze(dim=0)  # argv.posebatch = 1
        #             input_size = samples["input_size"].to(device).unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, 2]
        #             outputs, _ = model(imgs)

        #             max_depth = argv.max_depth
        #             bs, num_queries = outputs["pred_logits"].shape[:2]
        #             for i in range(bs):
        #                 human_prob = outputs["pred_logits"][i].softmax(-1)[..., 1]

        #                 _out_kepts_depth = outputs["pred_depth"][i]  # n x T x num_kpts x 1
        #                 # root + displacement
        #                 _out_kepts_depth[:, :, 1:, :] = _out_kepts_depth[:, :, 0:1, :] + _out_kepts_depth[:, :, 1:, :] / max_depth
        #                 out_kepts_depth = max_depth * _out_kepts_depth  # scale to original depth

        #                 out_score = outputs["pred_kpts2d"][i, :, :, :, 2:3]  # n x T x num_kpts x 1
        #                 out_kepts2d = outputs["pred_kpts2d"][i, :, :, :, 0:2]  # n x T x num_kpts x 2
        #                 # root + displacement
        #                 out_kepts2d[:, :, 1:, :] = out_kepts2d[:, :, :1, :] + out_kepts2d[:, :, 1:, :]
        #                 out_kepts2d = out_kepts2d * input_size  # scale to original image size

        #                 inv_trans = samples["inv_trans"]
        #                 input_size = samples["input_size"]
        #                 img_size = samples["img_size"]
        #                 filenames = samples["filenames"]
        #                 results.append(
        #                     {
        #                         "human_score": human_prob.cpu().numpy(),  # [n]
        #                         "pred_kpt_scores": out_score.cpu().numpy(),  # [n, T, num_joints, 1]
        #                         "pred_kpts": out_kepts2d.cpu().numpy(),  # [n, T, num_kpts, 2]
        #                         "pred_depth": out_kepts_depth.cpu().numpy(),  # [n, T, num_kpts, 1]
        #                         "inv_trans": inv_trans.cpu().numpy(),  # [2, 3]
        #                         "filenames": filenames,  # [filename_{t}, filename_{t+gap}, ...]
        #                         "input_size": input_size.cpu().numpy(),  # (w, h)
        #                         "img_size": img_size.cpu().numpy(),  # (w, h)
        #                     }
        #                 )

        #     logger.info(
        #         "【No.{dir_name}】姿勢推定の関連付け",
        #         dir_name=dir_name,
        #         decoration=MLogger.DECORATION_LINE,
        #     )

        #     all_frames_results, max_pid = associate_snippets(results, frame_indices, all_filenames, argv)

        #     logger.info(
        #         "【No.{dir_name}】姿勢推定結果保存(3D)",
        #         dir_name=dir_name,
        #         decoration=MLogger.DECORATION_LINE,
        #     )

        #     save_results_3d(
        #         all_frames_results,
        #         all_filenames,
        #         argv.data_dir,
        #         argv.outdir,
        #         max_pid,
        #         argv.max_depth,
        #         argv.seq_gap,
        #     )

        # logger.info(
        #     "姿勢推定結果検証",
        #     decoration=MLogger.DECORATION_LINE,
        # )

        # mix_outdir = os.path.join(args.img_dir, "snipper", "mix")
        # os.makedirs(mix_outdir, exist_ok=True)

        # mix_output_json_dir = os.path.join(mix_outdir, "json")
        # os.makedirs(mix_output_json_dir, exist_ok=True)

        # mix_output_track2d_dir = os.path.join(mix_outdir, "track2d")
        # os.makedirs(mix_output_track2d_dir, exist_ok=True)

        # frame_count = 0
        # for outdir in sorted(glob(os.path.join(args.img_dir, "snipper", "*"))):
        #     frame_count += len(glob(os.path.join(outdir, "json", "*.json"))) * 1000

        # all_json_datas = {}
        # with tqdm(
        #     total=frame_count,
        # ) as pchar:

        #     for oidx, outdir in enumerate(sorted(glob(os.path.join(args.img_dir, "snipper", "*")))):
        #         for json_path in list(sorted(glob(os.path.join(outdir, "json", "*.json")))):
        #             file_name = os.path.basename(json_path)
        #             person_idx, _ = file_name.split(".")
        #             json_datas = {}
        #             with open(json_path, "r") as f:
        #                 json_datas = json.load(f)

        #             if oidx == 0:
        #                 # 最初はそのままコピー
        #                 all_json_datas[person_idx] = json_datas
        #                 continue

        #             start_matchs = {}
        #             for (
        #                 target_person_idx,
        #                 person_json_datas,
        #             ) in all_json_datas.items():
        #                 start_matchs[target_person_idx] = {}

        #                 for sidx in list(json_datas.keys())[:200]:
        #                     start_matchs[target_person_idx][sidx] = 9999999999

        #                     bbox = json_datas[sidx]["snipper"]["bbox"]

        #                     bbox_x = int(bbox["x"])
        #                     bbox_y = int(bbox["y"])
        #                     bbox_w = int(bbox["width"])
        #                     bbox_h = int(bbox["height"])

        #                     if sidx not in person_json_datas:
        #                         continue

        #                     pbbox = person_json_datas[sidx]["snipper"]["bbox"]

        #                     pbbox_x = int(pbbox["x"])
        #                     pbbox_y = int(pbbox["y"])
        #                     pbbox_w = int(pbbox["width"])
        #                     pbbox_h = int(pbbox["height"])

        #                     # bboxの差異を図る
        #                     start_matchs[target_person_idx][sidx] = (
        #                         abs(pbbox_x - bbox_x) + abs(pbbox_y - bbox_y) + abs(pbbox_w - bbox_w) + abs(pbbox_h - bbox_h)
        #                     )

        #             match_idxs = {}
        #             for pidx, start_match in start_matchs.items():
        #                 match_idxs[pidx] = np.mean(list(start_match.values()))

        #             match_person_idx = list(match_idxs.keys())[np.argmin(list(match_idxs.values()))]
        #             # マッチしたのでも差異が大きければ新しくINDEX付与
        #             if match_idxs[match_person_idx] > 120:
        #                 match_person_idx = f"{(int(list(all_json_datas.keys())[-1]) + 1):03d}"
        #                 all_json_datas[match_person_idx] = {}

        #             for sidx, json_data in json_datas.items():
        #                 all_json_datas[match_person_idx][sidx] = json_data

        #                 pchar.update(1)

        # random.seed(13)
        # pids = list(all_json_datas.keys())
        # cmap = plt.get_cmap("rainbow")
        # pid_colors = [cmap(i) for i in np.linspace(0, 1, len(pids))]
        # random.shuffle(pid_colors)
        # pid_colors_opencv = [(np.array((c[2], c[1], c[0])) * 255).astype(int).tolist() for c in pid_colors]

        # logger.info("姿勢推定結果保存(2D)", decoration=MLogger.DECORATION_LINE)

        # with tqdm(
        #     total=frame_count,
        # ) as pchar:
        #     image_paths = {}
        #     for pidx, (pid, json_datas) in enumerate(all_json_datas.items()):
        #         for fidx, json_data in json_datas.items():
        #             image_path = json_data["snipper"]["image"]["path"]
        #             if image_path in image_paths:
        #                 image_path = image_paths[image_path]

        #             process_path = os.path.join(mix_output_track2d_dir, os.path.basename(image_path))
        #             save_visual_results_2d(
        #                 image_path,
        #                 process_path,
        #                 pidx,
        #                 pid,
        #                 pid_colors_opencv,
        #                 json_data["snipper"]["joints"],
        #                 json_data["snipper"]["bbox"],
        #             )

        #             image_paths[image_path] = process_path

        #             pchar.update(1)

        #         with open(os.path.join(mix_output_json_dir, f"{pid}.json"), mode="w") as f:
        #             json.dump(json_datas, f, indent=4)

        # logger.info("姿勢推定結果保存(動画)", decoration=MLogger.DECORATION_LINE)

        # frames = glob(os.path.join(mix_output_track2d_dir, "*.*"))
        # img = Image.open(frames[0])

        # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # out = cv2.VideoWriter(
        #     os.path.join(mix_outdir, "snipper.mp4"),
        #     fourcc,
        #     30.0,
        #     (img.size[0], img.size[1]),
        # )

        # for process_img_path in tqdm(frames):
        #     # トラッキングmp4合成
        #     out.write(cv2.imread(process_img_path))

        # out.release()
        # cv2.destroyAllWindows()

        logger.info(
            "2D姿勢推定結果保存完了: {outputpath}",
            outputpath=argv.outputpath,
            decoration=MLogger.DECORATION_BOX,
        )

        return True
    except Exception as e:
        logger.critical("2D姿勢推定で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False


# def save_visual_results_2d(
#     image_path: str,
#     process_path: str,
#     pidx: int,
#     pid: str,
#     pid_colors_opencv: list,
#     joints: dict,
#     bbox: dict,
# ):
#     img = cv2.imread(image_path)

#     SKELETONS = [
#         ("root", "left_hip"),
#         ("root", "right_hip"),
#         ("root", "head_bottom"),
#         ("head_bottom", "left_shoulder"),
#         ("head_bottom", "right_shoulder"),
#         ("head_bottom", "nose"),
#         ("left_shoulder", "left_elbow"),
#         ("left_elbow", "left_wrist"),
#         ("right_shoulder", "right_elbow"),
#         ("right_elbow", "right_wrist"),
#         ("left_hip", "left_knee"),
#         ("left_knee", "left_ankle"),
#         ("right_hip", "right_knee"),
#         ("right_knee", "right_ankle"),
#     ]

#     for l, (j1, j2) in enumerate(SKELETONS):
#         joint1 = joints[j1]
#         joint2 = joints[j2]

#         joint1_x = int(joint1["x"])
#         joint1_y = int(joint1["y"])
#         joint2_x = int(joint2["x"])
#         joint2_y = int(joint2["y"])

#         if joint1["z"] > 0 and joint2["z"] > 0:
#             t = 4
#             r = 8
#             cv2.line(
#                 img,
#                 (joint1_x, joint1_y),
#                 (joint2_x, joint2_y),
#                 color=tuple(pid_colors_opencv[pidx]),
#                 # color=tuple(sks_colors[l]),
#                 thickness=t,
#             )
#             cv2.circle(
#                 img,
#                 thickness=-1,
#                 center=(joint1_x, joint1_y),
#                 radius=r,
#                 color=tuple(pid_colors_opencv[pidx]),
#             )
#             cv2.circle(
#                 img,
#                 thickness=-1,
#                 center=(joint2_x, joint2_y),
#                 radius=r,
#                 color=tuple(pid_colors_opencv[pidx]),
#             )

#     bbox_x = int(bbox["x"])
#     bbox_y = int(bbox["y"])
#     bbox_w = int(bbox["width"])
#     bbox_h = int(bbox["height"])
#     bbx_thick = 3
#     cv2.line(
#         img,
#         (bbox_x, bbox_y),
#         (bbox_x + bbox_w, bbox_y),
#         color=tuple(pid_colors_opencv[pidx]),
#         thickness=bbx_thick,
#     )
#     cv2.line(
#         img,
#         (bbox_x, bbox_y),
#         (bbox_x, bbox_y + bbox_h),
#         color=tuple(pid_colors_opencv[pidx]),
#         thickness=bbx_thick,
#     )
#     cv2.line(
#         img,
#         (bbox_x + bbox_w, bbox_y),
#         (bbox_x + bbox_w, bbox_y + bbox_h),
#         color=tuple(pid_colors_opencv[pidx]),
#         thickness=bbx_thick,
#     )
#     cv2.line(
#         img,
#         (bbox_x, bbox_y + bbox_h),
#         (bbox_x + bbox_w, bbox_y + bbox_h),
#         color=tuple(pid_colors_opencv[pidx]),
#         thickness=bbx_thick,
#     )

#     cv2.putText(
#         img,
#         pid,
#         (bbox_x + bbox_w // 3, bbox_y - 5),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         color=tuple(pid_colors_opencv[pidx]),
#         thickness=bbx_thick,
#     )

#     cv2.imwrite(process_path, img)


# def save_results_3d(all_frames_results, all_filenames, data_dir, save_dir, max_pid, max_depth, gap):

#     result_dir = os.path.join(save_dir, "json")
#     os.makedirs(result_dir, exist_ok=True)

#     json_datas = {}
#     for frame_idx in tqdm(all_frames_results.keys()):
#         filename = all_filenames[frame_idx]
#         # ファイル名をそのままフレーム番号として扱う
#         fno = int(filename.split(".")[0])
#         img = cv2.imread(os.path.join(data_dir, filename))
#         h, w, _ = img.shape

#         pids, poses = all_frames_results[frame_idx]
#         for p, pid in enumerate(pids):
#             kpt_3d = poses[p]
#             if pid not in json_datas:
#                 json_datas[pid] = {}

#             bbx = bbox_2d_padded(kpt_3d, 0.3, 0.3)

#             json_datas[int(pid)][fno] = {
#                 "snipper": {
#                     "image": {
#                         "path": os.path.join(data_dir, filename),
#                         "width": float(w),
#                         "height": float(h),
#                     },
#                     "bbox": {
#                         "x": float(bbx[0]),
#                         "y": float(bbx[1]),
#                         "width": float(bbx[2]),
#                         "height": float(bbx[3]),
#                     },
#                     "joints": {},
#                 },
#             }
#             for n, (x, y, z, score) in enumerate(kpt_3d):
#                 json_datas[int(pid)][fno]["snipper"]["joints"][Joint.NAMES[n]] = {
#                     "x": float(x),
#                     "y": float(y),
#                     "z": float(z),
#                     "score": float(score),
#                 }

#     for pid, json_data in json_datas.items():
#         with open(os.path.join(result_dir, f"{pid:03d}.json"), mode="w") as f:
#             json.dump(json_data, f, indent=4)


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
    parser.add_argument("--detector", dest="detector", help="detector name", default="yolo")
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
