import struct
from struct import Struct

from base.base import BaseModel
from base.logger import MLogger
from mmd.vmd.collection import VmdMotion

logger = MLogger(__name__)


class VmdWriter(BaseModel):
    @classmethod
    def write(cls, model_name: str, motion: VmdMotion, output_vmd_path: str):
        fout = open(output_vmd_path, "wb")

        # header
        fout.write(b"Vocaloid Motion Data 0002\x00\x00\x00\x00\x00")

        try:
            # モデル名を20byteで切る
            model_bname = (
                model_name.encode("cp932").decode("shift_jis").encode("shift_jis")[:20]
            )
        except Exception:
            logger.warning(
                "モデル名に日本語・英語で判読できない文字が含まれているため、仮モデル名を設定します。 %s",
                model_name,
                decoration=MLogger.DECORATION_BOX,
            )
            model_bname = "Vmd Sized Model".encode("shift_jis")[:20]

        # 20文字に満たなかった場合、埋める
        model_bname = model_bname.ljust(20, b"\x00")

        fout.write(model_bname)

        # bone frames
        fout.write(struct.pack("<L", motion.get_bone_count()))  # ボーンフレーム数
        for bone_name in motion.bones.names():
            for fno in reversed(motion.bones[bone_name].indices()):
                bf = motion.bones[bone_name][fno]
                # INDEXを逆順に出力する
                bname = (
                    bf.name.encode("cp932")
                    .decode("shift_jis")
                    .encode("shift_jis")[:15]
                    .ljust(15, b"\x00")
                )  # 15文字制限
                fout.write(bname)
                fout.write(struct.pack("<L", int(bf.index)))
                fout.write(struct.pack("<f", float(bf.position.x)))
                fout.write(struct.pack("<f", float(bf.position.y)))
                fout.write(struct.pack("<f", float(bf.position.z)))
                v = bf.rotation.normalized().to_vector4()
                fout.write(struct.pack("<f", float(v.x)))
                fout.write(struct.pack("<f", float(v.y)))
                fout.write(struct.pack("<f", float(v.z)))
                fout.write(struct.pack("<f", float(v.w)))
                fout.write(
                    bytearray(
                        [int(min(127, max(0, x))) for x in bf.interpolations.merge()]
                    )
                )

        fout.write(struct.pack("<L", len(motion.morphs)))  # 表情フレーム数
        fout.write(struct.pack("<L", len(motion.cameras)))  # カメラキーフレーム数
        fout.write(struct.pack("<L", len(motion.lights)))  # 照明キーフレーム数
        fout.write(struct.pack("<L", len(motion.shadows)))  # セルフ影キーフレーム数

        fout.write(struct.pack("<L", len(motion.showiks)))  # モデル表示・IK on/offキーフレーム数
        for sk in motion.showiks:
            fout.write(struct.pack("<L", sk.index))
            fout.write(struct.pack("b", sk.show))
            fout.write(struct.pack("<L", len(sk.iks)))
            for ik in sk.iks:
                bname = (
                    ik.name.encode("cp932")
                    .decode("shift_jis")
                    .encode("shift_jis")[:20]
                    .ljust(20, b"\x00")
                )  # 20文字制限
                fout.write(bname)
                fout.write(struct.pack("b", ik.onoff))
