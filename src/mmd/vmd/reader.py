import re
from struct import Struct

from base.base import Encoding
from base.math import MVector3D
from base.reader import BaseReader, StructUnpackType
from mmd.vmd.collection import VmdMotion
from mmd.vmd.part import (
    VmdBoneFrame,
    VmdCameraFrame,
    VmdIkOnoff,
    VmdLightFrame,
    VmdMorphFrame,
    VmdShadowFrame,
    VmdShowIkFrame,
)

RE_TEXT_TRIM = re.compile(rb"\x00+$")


class VmdReader(BaseReader[VmdMotion]):  # type: ignore
    def __init__(self) -> None:
        super().__init__()

    def create_model(self, path: str) -> VmdMotion:
        return VmdMotion(path=path)

    def read_by_buffer_header(self, motion: VmdMotion):
        self.define_encoding(Encoding.SHIFT_JIS)

        self.read_by_format[VmdBoneFrame] = StructUnpackType(
            self.read_bones,
            Struct("<I3f4f64B").unpack_from,
            4 + (4 * 3) + (4 * 4) + 64,
        )

        self.read_by_format[VmdMorphFrame] = StructUnpackType(
            self.read_morphs,
            Struct("<If").unpack_from,
            4 + 4,
        )

        self.read_by_format[VmdCameraFrame] = StructUnpackType(
            self.read_cameras,
            Struct("<I3f3f").unpack_from,
            4 + (4 * 3 * 2),
        )

        self.read_by_format[VmdLightFrame] = StructUnpackType(
            self.read_lights,
            Struct("<If3f3f24BIB").unpack_from,
            4 + 4 + (4 * 3 * 2) + 24 + 4 + 1,
        )

        self.read_by_format[VmdShadowFrame] = StructUnpackType(
            self.read_shadows,
            Struct("<IBf").unpack_from,
            4 + 1 + 4,
        )

        # vmdバージョン
        motion.signature = self.read_text(30)

        # モデル名
        motion.model_name = self.read_text(20)

    def read_by_buffer(self, motion: VmdMotion):
        # ボーンモーション
        self.read_bones(motion)

        # モーフモーション
        self.read_morphs(motion)

        # カメラ
        self.read_cameras(motion)

        # 照明
        self.read_lights(motion)

        # セルフ影
        self.read_shadows(motion)

        # セルフ影
        self.read_show_iks(motion)

    def define_read_text(self, encoding: Encoding):
        """
        テキストの解凍定義

        Parameters
        ----------
        encoding : Encoding
            デコードエンコード
        """

        def read_text(format_size: int) -> str:
            btext = self.unpack_text(format_size)
            # VMDは空白込みで入っているので、空白以降は削除する
            btext = RE_TEXT_TRIM.sub(b"", btext)

            return self.decode_text(encoding, btext)

        return read_text

    def read_bones(self, motion: VmdMotion):
        for _ in range(self.read_uint()):
            bf = VmdBoneFrame(regist=True, read=True)

            bf.name = self.read_text(15)
            (
                bf.index,
                bf.position.x,
                bf.position.y,
                bf.position.z,
                bf.rotation.x,
                bf.rotation.y,
                bf.rotation.z,
                bf.rotation.scalar,
                bf.interpolations.translation_x.start.x,
                __,
                __,
                __,
                bf.interpolations.translation_x.start.y,
                __,
                __,
                __,
                bf.interpolations.translation_x.end.x,
                __,
                __,
                __,
                bf.interpolations.translation_x.end.y,
                __,
                __,
                __,
                bf.interpolations.translation_y.start.x,
                __,
                __,
                __,
                bf.interpolations.translation_y.start.y,
                __,
                __,
                __,
                bf.interpolations.translation_y.end.x,
                __,
                __,
                __,
                bf.interpolations.translation_y.end.y,
                __,
                __,
                __,
                bf.interpolations.translation_z.start.x,
                __,
                __,
                __,
                bf.interpolations.translation_z.start.y,
                __,
                __,
                __,
                bf.interpolations.translation_z.end.x,
                __,
                __,
                __,
                bf.interpolations.translation_z.end.y,
                __,
                bf.interpolations.residue0,
                bf.interpolations.residue1,
                bf.interpolations.rotation.start.x,
                __,
                __,
                __,
                bf.interpolations.rotation.start.y,
                __,
                __,
                __,
                bf.interpolations.rotation.end.x,
                __,
                __,
                __,
                bf.interpolations.rotation.end.y,
                bf.interpolations.residue2,
                bf.interpolations.residue3,
                bf.interpolations.residue4,
            ) = self.unpack(
                self.read_by_format[VmdBoneFrame].unpack,
                self.read_by_format[VmdBoneFrame].size,
            )

            motion.bones.append(bf)

    def read_morphs(self, motion: VmdMotion):
        for _ in range(self.read_uint()):
            mf = VmdMorphFrame(regist=True, read=True)

            mf.name = self.read_text(15)
            (mf.index, mf.ratio,) = self.unpack(
                self.read_by_format[VmdMorphFrame].unpack,
                self.read_by_format[VmdMorphFrame].size,
            )

            motion.morphs.append(mf)

    def read_cameras(self, motion: VmdMotion):
        for _ in range(self.read_uint()):
            cf = VmdCameraFrame(regist=True, read=True)
            degrees = MVector3D()

            (
                cf.index,
                cf.distance,
                cf.position.x,
                cf.position.y,
                cf.position.z,
                degrees.x,
                degrees.y,
                degrees.z,
                cf.interpolations.translation_x.start.x,
                cf.interpolations.translation_y.start.x,
                cf.interpolations.translation_z.start.x,
                cf.interpolations.rotation.start.x,
                cf.interpolations.distance.start.x,
                cf.interpolations.viewing_angle.start.x,
                cf.interpolations.translation_x.start.y,
                cf.interpolations.translation_y.start.y,
                cf.interpolations.translation_z.start.y,
                cf.interpolations.rotation.start.y,
                cf.interpolations.distance.start.y,
                cf.interpolations.viewing_angle.start.y,
                cf.interpolations.translation_x.end.x,
                cf.interpolations.translation_y.end.x,
                cf.interpolations.translation_z.end.x,
                cf.interpolations.rotation.end.x,
                cf.interpolations.distance.end.x,
                cf.interpolations.viewing_angle.end.x,
                cf.interpolations.translation_x.end.y,
                cf.interpolations.translation_y.end.y,
                cf.interpolations.translation_z.end.y,
                cf.interpolations.rotation.end.y,
                cf.interpolations.distance.end.y,
                cf.interpolations.viewing_angle.end.y,
            ) = self.unpack(
                self.read_by_format[VmdCameraFrame].unpack,
                self.read_by_format[VmdCameraFrame].size,
            )

            cf.rotation.degrees = degrees
            motion.cameras.append(cf)

    def read_lights(self, motion: VmdMotion):
        for _ in range(self.read_uint()):
            lf = VmdLightFrame(regist=True, read=True)

            (
                lf.index,
                lf.color.x,
                lf.color.y,
                lf.color.z,
                lf.position.x,
                lf.position.y,
                lf.position.z,
            ) = self.unpack(
                self.read_by_format[VmdLightFrame].unpack,
                self.read_by_format[VmdLightFrame].size,
            )

            motion.lights.append(lf)

    def read_shadows(self, motion: VmdMotion):
        for _ in range(self.read_uint()):
            sf = VmdShadowFrame(regist=True, read=True)

            (sf.index, sf.type, sf.distance,) = self.unpack(
                self.read_by_format[VmdShadowFrame].unpack,
                self.read_by_format[VmdShadowFrame].size,
            )

            motion.shadows.append(sf)

    def read_show_iks(self, motion: VmdMotion):
        for _ in range(self.read_uint()):
            kf = VmdShowIkFrame(regist=True, read=True)
            kf.index = self.read_uint()
            kf.show = bool(self.read_byte())

            for _i in range(self.read_uint()):
                kf.iks.append(VmdIkOnoff(self.read_text(20), bool(self.read_byte())))

            motion.showiks.append(kf)
