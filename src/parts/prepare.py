import datetime
import os
import pathlib
import shutil
import warnings

import cv2
import numpy as np
from base.logger import MLogger
from PIL import Image
from skimage import img_as_ubyte
from tqdm import tqdm

logger = MLogger(__name__)


def execute(args):
    try:
        logger.info("動画準備開始", decoration=MLogger.DECORATION_BOX)

        if not os.path.exists(args.video_file):
            logger.error(
                "指定されたファイルパスが存在しません。\n{video_file}",
                video_file=args.video_file,
                decoration=MLogger.DECORATION_BOX,
            )
            return False, None

        # 親パス(指定がなければ動画のある場所。Colabはローカルで作成するので指定あり想定)
        base_path = str(pathlib.Path(args.video_file).parent) if not args.parent_dir else args.parent_dir
        video = cv2.VideoCapture(args.video_file)

        # 幅
        W = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 高さ
        H = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 総フレーム数
        count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # fps
        fps = video.get(cv2.CAP_PROP_FPS)

        logger.info(
            "【初回チェック】\n　ファイル名: {video_file}, ファイルサイズ: {size}, 横: {W}, 縦: {H}, フレーム数: {count}, fps: {fps}",
            video_file=args.video_file,
            size=os.path.getsize(args.video_file),
            W=W,
            H=H,
            count=count,
            fps=round(fps, 5),
            decoration=MLogger.DECORATION_BOX,
        )

        # 縮尺を調整
        width = int(1280) if W > H else int(720)

        if len(args.parent_dir) > 0:
            process_img_dir = base_path
        else:
            process_img_dir = os.path.join(
                base_path,
                "{0}_{1:%Y%m%d_%H%M%S}".format(
                    os.path.basename(args.video_file).replace(".", "_"),
                    datetime.datetime.now(),
                ),
            )

        # 既存は削除
        if os.path.exists(process_img_dir):
            shutil.rmtree(process_img_dir)

        # フォルダ生成
        os.makedirs(process_img_dir)
        os.makedirs(os.path.join(process_img_dir, "resize"), exist_ok=True)
        os.makedirs(os.path.join(process_img_dir, "01_frames"), exist_ok=True)

        # リサイズpng出力先
        resize_img_path = os.path.join(process_img_dir, "resize", "{0:012}.png")
        # 補間png出力先
        process_img_path = os.path.join(process_img_dir, "01_frames", "{0:012}.png")

        # 縮尺
        scale = width / W

        # 縮尺後の高さ
        org_height = int(H * scale)
        height = int(org_height + (org_height % 40))

        try:
            # 入力ファイル
            cap = cv2.VideoCapture(args.video_file)

            logger.info("元動画読み込み開始", decoration=MLogger.DECORATION_BOX)

            if width % 40 != 0 or org_height % 40 != 0:
                logger.warning(
                    "入力動画のサイズが調整後に40で割り切れません。調整前({W}x{H}) -> 調整後({width}x{org_height})\n適切なサイズ({height})になるまで上辺を塗りつぶします。\n{video_file}",
                    W=W,
                    H=H,
                    width=width,
                    org_height=org_height,
                    height=height,
                    video_file=args.video_file,
                    decoration=MLogger.DECORATION_BOX,
                )

            for n in tqdm(range(int(count))):
                # 動画から1枚キャプチャして読み込む
                flag, org_img = cap.read()  # Capture frame-by-frame

                # 動画が終わっていたら終了
                if flag == False:
                    break

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    try:
                        # 画像に再変換
                        org_img = Image.fromarray(org_img)

                        # 画像の縦横を指定サイズに変形
                        org_img = org_img.resize((width, org_height), Image.ANTIALIAS)

                        img = Image.new(org_img.mode, (width, height), (0, 0, 0))

                        img.paste(org_img, (0, height - org_height))

                    except Exception as e:
                        # エラーするようなら無視
                        logger.error(e)

                    # opencv用に変換
                    out_frame = img_as_ubyte(img)

                    # PNG出力
                    cv2.imwrite(resize_img_path.format(n), out_frame)

            # 補間 --------------------------

            logger.info("補間生成開始", decoration=MLogger.DECORATION_BOX)

            # 元のフレームを30fpsで計算し直した場合の1Fごとの該当フレーム数
            interpolations = np.arange(0, count + 1, fps / 30)

            for kidx, k in enumerate(tqdm(interpolations)):
                # コピー対象の画像パス
                target_path = resize_img_path.format(round(k))

                if not os.path.exists(target_path):
                    # 最終フレームとかで対象パスがない場合、ひとつ手前
                    target_path = resize_img_path.format(round(k) - 1)

                process_path = process_img_path.format(kidx)
                if not os.path.exists(target_path):
                    # 最終フレームとかで対象パスがない場合、ひとつ手前
                    target_path = process_img_path.format(kidx - 1)

                os.makedirs(os.path.dirname(process_path), exist_ok=True)

                # 該当フレーム番号の画像をコピー
                shutil.copy(target_path, process_path)

            # 終わったら開放
            cap.release()

            logger.info(
                "【再チェック】\n　準備フォルダ: {process_img_dir}, 横: {width}, 縦: {height}, フレーム数: {last}, fps: {fps}",
                process_img_dir=process_img_dir,
                width=width,
                height=height,
                last=round(interpolations[-1]),
                fps=30,
            )
        except Exception as e:
            logger.error("再エンコード失敗", e)
            return False, None

        cv2.destroyAllWindows()

        # resizeは削除
        shutil.rmtree(os.path.join(process_img_dir, "resize"))

        logger.info(
            "動画準備完了: {process_img_dir}",
            process_img_dir=process_img_dir,
            decoration=MLogger.DECORATION_BOX,
        )

        return True, process_img_dir
    except Exception as e:
        logger.critical("動画準備で予期せぬエラーが発生しました。", e, decoration=MLogger.DECORATION_BOX)
        return False, None
