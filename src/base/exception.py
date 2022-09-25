import traceback


class MLibException(Exception):
    """ライブラリ内基本エラー"""

    def __init__(
        self, message: str = "", variants: list = [], exception: Exception = None, *args
    ):
        super().__init__(*args)
        self.message = message
        self.variants = variants
        self.exception = exception

    def __str__(self) -> str:
        return traceback.format_exc()


class MApplicationException(MLibException):
    """ツールがメイン処理出来なかった時のエラー"""

    def __init__(
        self, message: str = "", variants: list = [], exception: Exception = None, *args
    ):
        super().__init__(message, variants, exception, *args)


class MParseException(MLibException):
    """ツールがパース出来なかった時のエラー"""

    def __init__(
        self, message: str = "", variants: list = [], exception: Exception = None, *args
    ):
        super().__init__(message, variants, exception, *args)


class MKilledException(MLibException):
    """ツールの実行が停止された時のエラー"""

    def __init__(
        self, message: str = "", variants: list = [], exception: Exception = None, *args
    ):
        super().__init__(message, variants, exception, *args)
