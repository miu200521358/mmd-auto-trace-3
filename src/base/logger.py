import gettext
import logging
import os
import re
import traceback
from datetime import datetime
from enum import IntEnum

import numpy as np

from base.exception import MLibException


class LoggingMode(IntEnum):
    # 翻訳モード
    # 読み取り専用：翻訳リストにない文字列は入力文字列をそのまま出力する
    MODE_READONLY = 0
    # 更新あり：翻訳リストにない文字列は出力する
    MODE_UPDATE = 1


class MLogger:

    DECORATION_IN_BOX = "in_box"
    DECORATION_BOX = "box"
    DECORATION_LINE = "line"
    DEFAULT_FORMAT = "%(message)s [%(funcName)s][P-%(process)s](%(asctime)s)"

    DEBUG_FULL = 2
    TEST = 5
    TIMER = 12
    FULL = 15
    INFO_DEBUG = 22
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL

    # システム全体のロギングレベル
    total_level = logging.INFO
    # システム全体の開始出力日時
    outout_datetime = ""

    # LoggingMode
    mode = LoggingMode.MODE_READONLY
    # デフォルトログファイルパス
    default_out_path = ""
    # デフォルト翻訳言語
    lang = "en"
    translater = None

    logger = None
    re_break = re.compile(r"\n")
    stream_handler = None
    file_handler = None

    def __init__(self, module_name, level=logging.INFO, out_path=None):
        self.module_name = module_name
        self.default_level = level

        # ロガー
        self.logger = logging.getLogger("auto-trace-3").getChild(self.module_name)

        if not out_path:
            # クラス単位の出力パスがない場合、デフォルトパス
            out_path = self.default_out_path

        if out_path:
            # ファイル出力ハンドラ
            self.file_handler = logging.FileHandler(out_path)
            self.file_handler.setLevel(self.default_level)
            self.file_handler.setFormatter(logging.Formatter(self.DEFAULT_FORMAT))

        self.stream_handler = logging.StreamHandler()

    def time(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}

        kwargs["level"] = self.TIMER
        self.print_logger(msg, *args, **kwargs)

    def info_debug(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}

        kwargs["level"] = self.INFO_DEBUG
        self.print_logger(msg, *args, **kwargs)

    def test(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}

        kwargs["level"] = self.TEST
        self.print_logger(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}

        kwargs["level"] = logging.DEBUG
        self.print_logger(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}

        kwargs["level"] = logging.INFO
        self.print_logger(msg, *args, **kwargs)

    # ログレベルカウント
    def count(self, msg, fno, fnos, *args, **kwargs):
        last_fno = 0

        if fnos and len(fnos) > 0 and fnos[-1] > 0:
            last_fno = fnos[-1]

        if not fnos and kwargs and "last_fno" in kwargs and kwargs["last_fno"] > 0:
            last_fno = kwargs["last_fno"]

        if last_fno > 0:
            if not kwargs:
                kwargs = {}

            kwargs["level"] = logging.INFO
            kwargs["fno"] = fno
            kwargs["per"] = round((fno / last_fno) * 100, 3)
            kwargs["msg"] = msg
            log_msg = "-- {fno}フレーム目:終了({per}％){msg}"
            self.print_logger(log_msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}

        kwargs["level"] = logging.WARNING
        self.print_logger(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}

        kwargs["level"] = logging.ERROR
        self.print_logger(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if not kwargs:
            kwargs = {}

        kwargs["level"] = logging.CRITICAL
        self.print_logger(msg, *args, **kwargs)

    def quit(self):
        # 終了ログ
        with open("../log/quit.log", "w") as f:
            f.write("quit")

    # 実際に出力する実態
    def print_logger(self, msg, *args, **kwargs):

        target_level = kwargs.pop("level", logging.INFO)
        if self.total_level <= target_level and self.default_level <= target_level:
            # システム全体のロギングレベルもクラス単位のロギングレベルもクリアしてる場合のみ出力
            # サブモジュールのハンドラをクリア
            logger = logging.getLogger()
            while logger.hasHandlers():
                logger.removeHandler(logger.handlers[0])
            self.logger.addHandler(self.stream_handler)
            self.logger.addHandler(self.file_handler)

            # モジュール名を出力するよう追加
            extra_args = {}
            extra_args["module_name"] = self.module_name

            # 翻訳する
            if self.mode == LoggingMode.MODE_UPDATE and logging.DEBUG < target_level:
                # 更新ありの場合、既存データのチェックを行って追記する
                messages = []
                with open("i18n/messages.pot", mode="r", encoding="utf-8") as f:
                    messages = f.readlines()

                new_msg = self.re_break.sub("\\\\n", msg)
                added_msg_idxs = [n + 1 for n, inmsg in enumerate(messages) if "msgid" in inmsg and new_msg in inmsg]

                if not added_msg_idxs:
                    messages.append(f'\nmsgid "{new_msg}"\n')
                    messages.append('msgstr ""\n')
                    messages.append("\n")
                    print("add message: %s", new_msg)

                    with open("i18n/messages.pot", mode="w", encoding="utf-8") as f:
                        f.writelines(messages)

            # 翻訳結果を取得する
            trans_msg = self.translater.gettext(msg)

            # ログレコード生成
            if args and isinstance(args[0], Exception) or (args and len(args) > 1 and isinstance(args[0], Exception)):
                trans_msg = f"{trans_msg}\n\n{traceback.format_exc()}"
                args = None
                log_record = self.logger.makeRecord(
                    "name",
                    target_level,
                    "(unknown file)",
                    0,
                    args,
                    None,
                    None,
                    self.module_name,
                )
            else:
                log_record = self.logger.makeRecord(
                    "name",
                    target_level,
                    "(unknown file)",
                    0,
                    trans_msg,
                    args,
                    None,
                    self.module_name,
                )

            target_decoration = kwargs.pop("decoration", None)
            title = kwargs.pop("title", None)

            print_msg = str(trans_msg)
            if kwargs:
                print_msg = print_msg.format(**kwargs)

            if target_decoration:
                if target_decoration == MLogger.DECORATION_BOX:
                    output_msg = self.create_box_message(print_msg, target_level, title)
                elif target_decoration == MLogger.DECORATION_LINE:
                    output_msg = self.create_line_message(print_msg, target_level, title)
                elif target_decoration == MLogger.DECORATION_IN_BOX:
                    output_msg = self.create_in_box_message(print_msg, target_level, title)
                else:
                    output_msg = self.create_simple_message(print_msg, target_level, title)
            else:
                output_msg = self.create_simple_message(print_msg, target_level, title)

            # 出力
            try:
                log_record = self.logger.makeRecord(
                    "name",
                    target_level,
                    "(unknown file)",
                    0,
                    output_msg,
                    None,
                    None,
                    self.module_name,
                )
                self.logger.handle(log_record)
            except:
                # エラーしてたら無視
                pass

    def create_box_message(self, msg, level, title=None):
        msg_block = []
        msg_block.append("■■■■■■■■■■■■■■■■■")

        if level == logging.CRITICAL:
            msg_block.append("■　**CRITICAL**　")

        if level == logging.ERROR:
            msg_block.append("■　**ERROR**　")

        if level == logging.WARNING:
            msg_block.append("■　**WARNING**　")

        if level <= logging.INFO and title:
            msg_block.append("■　**{0}**　".format(title))

        for msg_line in msg.split("\n"):
            msg_block.append("■　{0}".format(msg_line))

        msg_block.append("■■■■■■■■■■■■■■■■■")

        return "\n".join(msg_block)

    def create_line_message(self, msg, level, title=None):
        msg_block = []

        for msg_line in msg.split("\n"):
            msg_block.append("-- {0} --------------------".format(msg_line))

        return "\n".join(msg_block)

    def create_in_box_message(self, msg, level, title=None):
        msg_block = []

        for msg_line in msg.split("\n"):
            msg_block.append("■　{0}".format(msg_line))

        return "\n".join(msg_block)

    def create_simple_message(self, msg, level, title=None):
        msg_block = []

        for msg_line in msg.split("\n"):
            # msg_block.append("[{0}] {1}".format(logging.getLevelName(level)[0], msg_line))
            msg_block.append(msg_line)

        return "\n".join(msg_block)

    @classmethod
    def initialize(cls, lang: str, mode: LoggingMode, level=logging.INFO, out_path=None):
        logging.basicConfig(level=level, format=cls.DEFAULT_FORMAT)
        cls.total_level = level
        cls.mode = mode
        cls.lang = lang

        # 翻訳用クラスの設定
        cls.translater = gettext.translation(
            "messages",  # domain: 辞書ファイルの名前
            localedir="i18n",  # 辞書ファイル配置ディレクトリ
            languages=[lang],  # 翻訳に使用する言語
            fallback=True,  # .moファイルが見つからなかった時は未翻訳の文字列を出力
        )

        outout_datetime = "{0:%Y%m%d_%H%M%S}".format(datetime.now())
        cls.outout_datetime = outout_datetime

        # ファイル出力ありの場合、ログファイル名生成
        if not out_path:
            os.makedirs("../log", exist_ok=True)
            cls.default_out_path = "../log/autotrace3_{0}.log".format(outout_datetime)
        else:
            cls.default_out_path = out_path

        if os.path.exists("../log/quit.log"):
            # 終了ログは初期化時に削除
            os.remove("../log/quit.log")


def parse2str(obj: object) -> str:
    """オブジェクトの変数の名前と値の一覧を文字列で返す

    Parameters
    ----------
    obj : object

    Returns
    -------
    str
        変数リスト文字列
        Sample[x=2, a=sss, child=ChildSample[y=4.5, b=xyz]]
    """
    return f"{obj.__class__.__name__}[{', '.join([f'{k}={v}' for k, v in vars(obj).items()])}]"


def parse_str(v: object, decimals=5) -> str:
    """
    丸め処理付き文字列変換処理

    小数だったら丸めて一定桁数までしか出力しない
    """
    if isinstance(v, float):
        return f"{round(v, decimals)}"
    elif isinstance(v, np.ndarray):
        return f"{np.round(v, decimals)}"
    elif hasattr(v, "data"):
        return f"{np.round(v.__getattribute__('data'), decimals)}"
    else:
        return f"{v}"


# ファイルのエンコードを取得する
def get_file_encoding(file_path):

    try:
        f = open(file_path, "rb")
        fbytes = f.read()
        f.close()
    except:
        raise MLibException("unknown encoding!")

    codelst = ("utf-8", "shift-jis")

    for encoding in codelst:
        try:
            fstr = fbytes.decode(encoding)  # bytes文字列から指定文字コードの文字列に変換
            fstr = fstr.encode("utf-8")  # uft-8文字列に変換
            # 問題なく変換できたらエンコードを返す
            return encoding
        except Exception as e:
            print(e)
            pass

    raise MLibException("unknown encoding!")
