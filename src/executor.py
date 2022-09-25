import argparse
import os
import time

import parts
from base.logger import MLogger

logger = MLogger(__name__)


def show_worked_time(elapsed_time):
    # 経過秒数を時分秒に変換
    td_m, td_s = divmod(elapsed_time, 60)
    td_h, td_m = divmod(td_m, 60)

    if td_m == 0:
        worked_time = "00:00:{0:02d}".format(int(td_s))
    elif td_h == 0:
        worked_time = "00:{0:02d}:{1:02d}".format(int(td_m), int(td_s))
    else:
        worked_time = "{0:02d}:{1:02d}:{2:02d}".format(int(td_h), int(td_m), int(td_s))

    return worked_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-file", type=str, dest="video_file", default="", help="Video file path")
    parser.add_argument(
        "--parent-dir",
        type=str,
        dest="parent_dir",
        default="",
        help="Process parent dir path",
    )
    parser.add_argument("--process", type=str, dest="process", default="", help="Process to be executed")
    parser.add_argument(
        "--img-dir",
        type=str,
        dest="img_dir",
        default="",
        help="Prepared image directory",
    )
    parser.add_argument("--audio-file", type=str, dest="audio_file", default="", help="Audio file path")
    parser.add_argument(
        "--trace-mov-model-config",
        type=str,
        dest="trace_mov_model_config",
        default=os.path.abspath(os.path.join(__file__, "../../data/pmx/trace_mov_model.pmx")),
        help="MMD Model Bone pmx",
    )
    parser.add_argument(
        "--trace-rot-model-config",
        type=str,
        dest="trace_rot_model_config",
        default=os.path.abspath(os.path.join(__file__, "../../data/pmx/trace_model.pmx")),
        help="MMD Model Bone pmx",
    )
    parser.add_argument("--verbose", type=int, dest="verbose", default=20, help="Log level")
    parser.add_argument("--log-mode", type=int, dest="log_mode", default=0, help="Log output mode")
    parser.add_argument("--lang", type=str, dest="lang", default="en", help="Language")

    args = parser.parse_args()
    MLogger.initialize(level=args.verbose, mode=args.log_mode, lang=args.lang)
    result = True

    start = time.time()

    logger.info(
        "MMD自動トレース開始\n　処理対象映像ファイル: {video_file}\n　処理内容: {process}",
        video_file=args.video_file,
        process=args.process,
        decoration=MLogger.DECORATION_BOX,
    )

    if "prepare" in args.process:
        # 準備
        from parts.prepare import execute

        result, args.img_dir = execute(args)

    if result and "alphapose" in args.process:
        # alphaposeによる2D人物推定
        from parts.alphapose import execute

        result = execute(args)

    # if result and "mediapipe" in args.process:
    #     # mediapipeによる人物推定
    #     from parts.mediapipe import execute

    #     result = execute(args)

    # if result and "smooth" in args.process:
    #     # 人物スムージング
    #     from parts.smooth import execute

    #     result = execute(args)

    # if result and "motion" in args.process:
    #     # モーション生成
    #     from parts.motion import execute

    #     result = execute(args)

    elapsed_time = time.time() - start

    logger.info(
        "MMD自動トレース終了\n　処理対象映像ファイル: {video_file}\n　処理内容: {process}\n　トレース結果: {img_dir}\n　処理時間: {elapsed_time}",
        video_file=args.video_file,
        process=args.process,
        img_dir=args.img_dir,
        elapsed_time=show_worked_time(elapsed_time),
        decoration=MLogger.DECORATION_BOX,
    )

    # 終了音を鳴らす
    if os.name == "nt":
        # Windows
        try:
            import winsound

            winsound.PlaySound("SystemAsterisk", winsound.SND_ALIAS)
        except Exception:
            pass
