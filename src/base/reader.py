import functools
import struct
from abc import ABCMeta, abstractmethod
from struct import Struct
from typing import Any, Callable, Generic, TypeVar

import numpy as np

from base.base import BaseModel, Encoding
from base.collection import BaseHashModel
from base.exception import MParseException
from base.math import MQuaternion, MVector2D, MVector3D, MVector4D

TBaseModel = TypeVar("TBaseModel", bound=BaseModel)
TBaseHashModel = TypeVar("TBaseHashModel", bound=BaseHashModel)


class StructUnpackType:
    def __init__(self, reader: Callable, unpack: Callable, size: int) -> None:
        self.reader = reader
        self.unpack = unpack
        self.size = size


class BaseReader(Generic[TBaseHashModel], BaseModel, metaclass=ABCMeta):
    def __init__(self) -> None:
        super().__init__()
        self.offset = 0
        self.buffer: bytes = b""

        # バイナリ解凍処理のマッピング
        self.read_by_format = {
            np.byte: StructUnpackType(self.read_sbyte, Struct("<b").unpack_from, 1),
            np.ubyte: StructUnpackType(self.read_byte, Struct("<B").unpack_from, 1),
            np.short: StructUnpackType(self.read_short, Struct("<h").unpack_from, 2),
            np.ushort: StructUnpackType(self.read_ushort, Struct("<H").unpack_from, 2),
            int: StructUnpackType(self.read_int, Struct("<i").unpack_from, 4),
            np.uint: StructUnpackType(self.read_uint, Struct("<I").unpack_from, 4),
            float: StructUnpackType(self.read_float, Struct("<f").unpack_from, 4),
            np.double: StructUnpackType(self.read_double, Struct("<d").unpack_from, 8),
            MVector2D: StructUnpackType(
                self.read_MVector2D, Struct("<ff").unpack_from, 4 * 2
            ),
            MVector3D: StructUnpackType(
                self.read_MVector3D, Struct("<fff").unpack_from, 4 * 3
            ),
            MVector4D: StructUnpackType(
                self.read_MVector4D, Struct("<ffff").unpack_from, 4 * 4
            ),
            MQuaternion: StructUnpackType(
                self.read_MQuaternion, Struct("<ffff").unpack_from, 4 * 4
            ),
        }

    def read_name_by_filepath(self, path: str) -> str:
        """
        指定されたパスのファイルから該当名称を読み込む

        Parameters
        ----------
        path : str
            ファイルパス

        Returns
        -------
        str
            読み込み結果文字列
        """
        # モデルを新規作成
        model: TBaseHashModel = self.create_model(path)

        # バイナリを解凍してモデルに展開
        try:
            with open(path, "rb") as f:
                self.buffer = f.read()
                self.read_by_buffer_header(model)
        except Exception:
            return ""

        return model.get_name()

    def read_by_filepath(self, path: str) -> TBaseHashModel:
        """
        指定されたパスのファイルからデータを読み込む

        Parameters
        ----------
        path : str
            ファイルパス

        Returns
        -------
        TBaseHashModel
            読み込み結果
        """
        # モデルを新規作成
        model: TBaseHashModel = self.create_model(path)

        # バイナリを解凍してモデルに展開
        try:
            with open(path, "rb") as f:
                self.buffer = f.read()
                self.read_by_buffer_header(model)
                self.read_by_buffer(model)
        except MParseException as pe:
            raise pe
        except Exception as e:
            # TODO
            raise MParseException("予期せぬエラー", exception=e)

        # ハッシュを保持
        model.digest = model.hexdigest()

        return model

    @abstractmethod
    def create_model(self, path: str) -> TBaseHashModel:
        """
        読み取り対象モデルオブジェクトを生成する

        Returns
        -------
        TBaseHashModel
            モデルオブジェクト
        """
        pass

    @abstractmethod
    def read_by_buffer_header(self, model: TBaseHashModel):
        """
        バッファからモデルデータヘッダを読み取る

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        model : TBaseHashModel
            モデルオブジェクト
        """
        pass

    @abstractmethod
    def read_by_buffer(self, model: TBaseHashModel):
        """
        バッファからモデルデータを読み取る

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        model : TBaseHashModel
            モデルオブジェクト
        """
        pass

    def define_encoding(self, encoding: Encoding):
        """
        エンコードを設定し、それに基づくテキストデータ読み取り処理を定義する

        Parameters
        ----------
        encoding : Encoding
            エンコード
        """
        self.encoding = encoding
        self.read_text = self.define_read_text(self.encoding)

    def define_read_text(self, encoding: Encoding):
        """
        テキストの解凍定義

        Parameters
        ----------
        encoding : Encoding
            デコードエンコード
        """

        def read_text() -> str:
            format_size = self.read_int()
            return self.decode_text(encoding, self.unpack_text(format_size))

        return read_text

    @functools.lru_cache()
    def decode_text(self, main_encoding: Encoding, fbytes: bytearray) -> str:
        """
        テキストデコード

        Parameters
        ----------
        main_encoding : Encoding
            基本のエンコーディング
        fbytes : bytearray
            バイト文字列

        Returns
        -------
        Optional[str]
            デコード済み文字列
        """
        # 基本のエンコーディングを第一候補でデコードして、ダメなら順次テスト
        for target_encoding in [
            main_encoding,
            Encoding.SHIFT_JIS,
            Encoding.UTF_8,
            Encoding.UTF_16_LE,
        ]:
            try:
                if target_encoding == Encoding.SHIFT_JIS:
                    # shift-jisは一旦cp932に変換してもう一度戻したので返す
                    return (
                        fbytes.decode(Encoding.SHIFT_JIS.value, errors="replace")
                        .encode(Encoding.CP932.value, errors="replace")
                        .decode(Encoding.CP932.value, errors="replace")
                    )

                # 変換できなかった文字は「?」に変換する
                return fbytes.decode(encoding=target_encoding.value, errors="replace")
            except Exception:
                pass
        return ""

    def read_MVector2D(
        self,
    ) -> MVector2D:
        """
        MVector2Dの解凍

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        offset : int
            オフセット

        Returns
        -------
        tuple[MVector2D, int]
            MVector2Dデータ
            オフセット
        """
        return MVector2D(
            *self.unpack(
                self.read_by_format[MVector2D].unpack,
                self.read_by_format[MVector2D].size,
            )
        )

    def read_MVector3D(
        self,
    ) -> MVector3D:
        """
        MVector3Dの解凍

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        offset : int
            オフセット

        Returns
        -------
        tuple[MVector3D, int]
            MVector3Dデータ
            オフセット
        """
        return MVector3D(
            *self.unpack(
                self.read_by_format[MVector3D].unpack,
                self.read_by_format[MVector3D].size,
            )
        )

    def read_MVector4D(
        self,
    ) -> MVector4D:
        """
        MVector4Dの解凍

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        offset : int
            オフセット

        Returns
        -------
        tuple[MVector4D, int]
            MVector4Dデータ
            オフセット
        """
        return MVector4D(
            *self.unpack(
                self.read_by_format[MVector4D].unpack,
                self.read_by_format[MVector4D].size,
            )
        )

    def read_MQuaternion(
        self,
    ) -> MQuaternion:
        """
        MQuaternionの解凍

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        offset : int
            オフセット

        Returns
        -------
        tuple[MQuaternion, int]
            MQuaternionデータ
            オフセット
        """
        x, y, z, scalar = self.unpack(
            self.read_by_format[MQuaternion].unpack,
            self.read_by_format[MQuaternion].size,
        )
        return MQuaternion(scalar, x, y, z)

    # @profile
    def read_to_model(
        self,
        formats: list[
            tuple[
                str,
                type,
            ]
        ],
        model: TBaseModel,
    ) -> TBaseModel:
        """
        フォーマットに沿って解凍する

        Parameters
        ----------
        buffer : TBuffer
            バッファ
        offset : int
            オフセット
        formats : list[tuple[str, type]]
            フォーマットリスト（属性名、属性クラス）
        model : TBaseModel
            設定対象モデルデータ

        Returns
        -------
        tuple[TBaseModel, int]
            解凍済みモデルデータ
            移動済みオフセット
        """
        v: Any = None
        for attr_name, format_type in formats:
            if isinstance(format_type(), BaseModel):
                submodel: TBaseModel = format_type()
                v = self.read_to_model([(attr_name, submodel.__class__)], submodel)
            else:
                v = self.read_by_format[format_type].reader()

            model.__setattr__(attr_name, v)
        return model

    def read_sbyte(self) -> int:
        """
        byteを読み込む
        np.byte:    sbyte	    : 1  - 符号あり  | char
        """
        return int(
            self.unpack(
                self.read_by_format[np.byte].unpack, self.read_by_format[np.byte].size
            )
        )

    def read_byte(self) -> int:
        """
        byteを読み込む
        np.ubyte:   byte	    : 1  - 符号なし  | unsigned char
        """
        return int(
            self.unpack(
                self.read_by_format[np.ubyte].unpack, self.read_by_format[np.ubyte].size
            )
        )

    def read_short(self) -> int:
        """
        shortを読み込む
        np.short:   short	    : 2  - 符号あり  | short
        """
        return int(
            self.unpack(
                self.read_by_format[np.short].unpack, self.read_by_format[np.short].size
            )
        )

    def read_ushort(self) -> int:
        """
        ushortを読み込む
        np.ushort   ushort	    : 2  - 符号なし  | unsigned short
        """
        return int(
            self.unpack(
                self.read_by_format[np.ushort].unpack,
                self.read_by_format[np.ushort].size,
            )
        )

    def read_int(self) -> int:
        """
        intを読み込む
        int:        int 	    : 4  - 符号あり  | int (32bit固定)
        """
        return int(
            self.unpack(self.read_by_format[int].unpack, self.read_by_format[int].size)
        )

    def read_uint(self) -> int:
        """
        uintを読み込む
        np.uint     uint	    : 4  - 符号なし  | unsigned int
        """
        return int(
            self.unpack(
                self.read_by_format[np.uint].unpack, self.read_by_format[np.uint].size
            )
        )

    def read_float(self) -> float:
        """
        floatを読み込む
        float       float	    : 4  - 単精度実数 | float
        """
        return float(
            self.unpack(
                self.read_by_format[float].unpack, self.read_by_format[float].size
            )
        )

    def read_double(self) -> np.double:
        """
        doubleを読み込む
        np.double   double	    : 8  - 浮動小数点数 | double
        """
        return np.array(
            [
                self.unpack(
                    self.read_by_format[np.double].unpack,
                    self.read_by_format[np.double].size,
                )
            ],
            dtype=np.double,
        )[0]

    def unpack_text(self, format_size: int):
        """
        バイナリを解凍

        Parameters
        ----------
        offset : int
            オフセット

        Returns
        -------
        Any
            読み取り結果
        """
        # バイナリ読み取り
        b: tuple = struct.unpack_from(f"{format_size}s", self.buffer, self.offset)
        # オフセット加算
        self.offset += format_size

        if b:
            return b[0]

        return None

    def unpack(
        self,
        unpack: Callable,
        format_size: int,
    ):
        """
        バイナリを解凍

        Parameters
        ----------
        unpack : StructUnpack
            解凍定義
        format_size : int
            桁数

        Returns
        -------
        Any
            読み取り結果
        """
        # バイナリ読み取り
        b: tuple = unpack(self.buffer, self.offset)
        # オフセット加算
        self.offset += format_size

        if len(b) == 1:
            return b[0]

        return b
