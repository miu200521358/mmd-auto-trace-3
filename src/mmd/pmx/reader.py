from struct import Struct

from base.base import Encoding
from base.exception import MParseException
from base.math import MVector3D
from base.part import Switch
from mmd.pmx.collection import PmxModel
from mmd.pmx.part import (
    Bdef1,
    Bdef2,
    Bdef4,
    Bone,
    BoneFlg,
    BoneMorphOffset,
    DeformType,
    DisplaySlot,
    DisplaySlotReference,
    DisplayType,
    DrawFlg,
    Face,
    GroupMorphOffset,
    Ik,
    IkLink,
    Joint,
    Material,
    MaterialMorphCalcMode,
    MaterialMorphOffset,
    Morph,
    MorphPanel,
    MorphType,
    RigidBody,
    RigidBodyCollisionGroup,
    Sdef,
    Texture,
    ToonSharing,
    UvMorphOffset,
    Vertex,
    VertexMorphOffset,
)
from base.reader import BaseReader, StructUnpackType


class PmxReader(BaseReader[PmxModel]):
    def __init__(self) -> None:
        super().__init__()

    def create_model(self, path: str) -> PmxModel:
        return PmxModel(path=path)

    def read_by_buffer_header(self, model: PmxModel):
        # pmx宣言
        model.signature = self.unpack_text(4)

        # pmxバージョン
        model.version = self.read_float()

        if model.signature[:3] != b"PMX" or f"{model.version:.1f}" not in [
            "2.0",
            "2.1",
        ]:
            # 整合性チェック
            raise MParseException(
                "PMX2.0/2.1形式外のデータです。signature: %s, version: %s ",
                [model.signature, model.version],
            )

        # 後続するデータ列のバイトサイズ  PMX2.0は 8 で固定
        _ = self.read_byte()

        # [0] - エンコード方式  | 0:UTF16 1:UTF8
        encode_type = self.read_byte()
        self.define_encoding(Encoding.UTF_8 if encode_type else Encoding.UTF_16_LE)

        # [1] - 追加UV数 	| 0～4 詳細は頂点参照
        # [2] - 頂点Indexサイズ | 1,2,4 のいずれか
        # [3] - テクスチャIndexサイズ | 1,2,4 のいずれか
        # [4] - 材質Indexサイズ | 1,2,4 のいずれか
        # [5] - ボーンIndexサイズ | 1,2,4 のいずれか
        # [6] - モーフIndexサイズ | 1,2,4 のいずれか
        # [7] - 剛体Indexサイズ | 1,2,4 のいずれか

        (
            model.extended_uv_count,
            model.vertex_count,
            model.texture_count,
            model.material_count,
            model.bone_count,
            model.morph_count,
            model.rigidbody_count,
        ) = self.unpack(Struct("<BBBBBBB").unpack_from, 7)

        # モデル名（日本語）
        model.name = self.read_text()

    def read_by_buffer(self, model: PmxModel):

        # モデルの各要素サイズから読み取り処理を設定
        self.read_vertex_index, self.vertex_index_format = self.define_read_index(
            model.vertex_count, is_vertex=True
        )
        self.read_texture_index, self.texture_index_format = self.define_read_index(
            model.texture_count
        )
        self.read_material_index, self.material_index_format = self.define_read_index(
            model.material_count
        )
        self.read_bone_index, self.bone_index_format = self.define_read_index(
            model.bone_count
        )
        self.read_morph_index, self.morph_index_format = self.define_read_index(
            model.morph_count
        )
        self.read_rigidbody_index, self.rigidbody_index_format = self.define_read_index(
            model.rigidbody_count
        )

        self.read_by_format[Vertex] = StructUnpackType(
            self.read_vertices, Struct(f"<{'fff' * 2}{'ff'}").unpack_from, 4 * 8
        )
        self.read_by_format[Bdef2] = StructUnpackType(
            self.read_vertices,
            Struct(f"<{self.bone_index_format * 2}f").unpack_from,
            model.bone_count * 2 + 4,
        )
        self.read_by_format[Bdef4] = StructUnpackType(
            self.read_vertices,
            Struct(f"<{self.bone_index_format * 4}{'f' * 4}").unpack_from,
            model.bone_count * 4 + 4 * 4,
        )
        self.read_by_format[Sdef] = StructUnpackType(
            self.read_vertices,
            Struct(f"<{self.bone_index_format * 2}f{'fff' * 3}").unpack_from,
            model.bone_count * 2 + 4 + (4 * 3) * 3,
        )
        self.read_by_format[Material] = StructUnpackType(
            self.read_materials,
            Struct(f"<fffffffffffbfffff{self.texture_index_format * 2}bb").unpack_from,
            (model.texture_count * 2) + (1 * 3) + (4 * 16),
        )
        self.read_by_format[RigidBody] = StructUnpackType(
            self.read_rigidbodies,
            Struct(f"<{self.bone_index_format}bHb{'fff' * 3}fffffb").unpack_from,
            (model.bone_count) + (1 * 3) + (2) + (4 * 3 * 3) + (4 * 5),
        )
        self.read_by_format[Joint] = StructUnpackType(
            self.read_joints,
            Struct(f"<b{self.rigidbody_index_format * 2}{'fff' * 8}").unpack_from,
            1 + (model.rigidbody_count * 2) + (4 * 3 * 8),
        )

        # モデル名（英語）
        model.english_name = self.read_text()

        # コメント
        model.comment = self.read_text()

        # コメント英
        model.english_comment = self.read_text()

        # 頂点
        self.read_vertices(model)

        # 面
        self.read_faces(model)

        # テクスチャ
        self.read_textures(model)

        # 材質
        self.read_materials(model)

        # ボーン
        self.read_bones(model)

        # モーフ
        self.read_morphs(model)

        # 表示枠
        self.read_display_slots(model)

        # 剛体
        self.read_rigidbodies(model)

        # ジョイント
        self.read_joints(model)

    def read_vertices(self, model: PmxModel):
        """頂点データ読み込み"""
        for _ in range(self.read_int()):
            vertex = Vertex()
            (
                vertex.position.x,
                vertex.position.y,
                vertex.position.z,
                vertex.normal.x,
                vertex.normal.y,
                vertex.normal.z,
                vertex.uv.x,
                vertex.uv.y,
            ) = self.unpack(
                self.read_by_format[Vertex].unpack, self.read_by_format[Vertex].size
            )

            if model.extended_uv_count > 0:
                vertex.extended_uvs.append(self.read_MVector4D())

            vertex.deform_type = DeformType(self.read_byte())
            if vertex.deform_type == DeformType.BDEF1:
                vertex.deform = Bdef1(self.read_bone_index())
            elif vertex.deform_type == DeformType.BDEF2:
                vertex.deform = Bdef2(
                    *self.unpack(
                        self.read_by_format[Bdef2].unpack,
                        self.read_by_format[Bdef2].size,
                    )
                )
            elif vertex.deform_type == DeformType.BDEF4:
                vertex.deform = Bdef4(
                    *self.unpack(
                        self.read_by_format[Bdef4].unpack,
                        self.read_by_format[Bdef4].size,
                    )
                )
            else:
                vertex.deform = Sdef(
                    *self.unpack(
                        self.read_by_format[Sdef].unpack, self.read_by_format[Sdef].size
                    )
                )
            vertex.edge_factor = self.read_float()
            model.vertices.append(vertex)

    def read_faces(self, model: PmxModel):
        """面データ読み込み"""
        faces_vertex_count = self.read_int()
        faces_vertices = self.unpack(
            Struct(f"<{self.vertex_index_format * faces_vertex_count}").unpack_from,
            model.vertex_count * faces_vertex_count,
        )

        for v0, v1, v2 in zip(
            faces_vertices[:-2:3], faces_vertices[1:-1:3], faces_vertices[2::3]
        ):
            model.faces.append(Face(v0, v1, v2))

    def read_textures(self, model: PmxModel):
        """テクスチャデータ読み込み"""
        for _ in range(self.read_int()):
            model.textures.append(Texture(self.read_text()))

    def read_materials(self, model: PmxModel):
        """材質データ読み込み"""
        for _ in range(self.read_int()):
            material = Material()
            material.name = self.read_text()
            material.english_name = self.read_text()

            (
                material.diffuse_color.x,
                material.diffuse_color.y,
                material.diffuse_color.z,
                material.diffuse_color.w,
                material.specular_color.x,
                material.specular_color.y,
                material.specular_color.z,
                material.specular_factor,
                material.ambient_color.x,
                material.ambient_color.y,
                material.ambient_color.z,
                draw_flg,
                material.edge_color.x,
                material.edge_color.y,
                material.edge_color.z,
                material.edge_color.w,
                material.edge_size,
                material.texture_index,
                material.sphere_texture_index,
                material.sphere_mode,
                material.toon_sharing_flg,
            ) = self.unpack(
                self.read_by_format[Material].unpack, self.read_by_format[Material].size
            )

            material.draw_flg = DrawFlg(draw_flg)

            if material.toon_sharing_flg == ToonSharing.INDIVIDUAL:
                # 個別の場合、テクスチャINDEX
                material.toon_texture_index = self.read_texture_index()
            else:
                # 共有の場合、0-9の共有テクスチャINDEX
                material.toon_texture_index = self.read_byte()
            material.comment = self.read_text()
            material.vertices_count = self.read_int()
            model.materials.append(material)

    def read_bones(self, model: PmxModel):
        """ボーンデータ読み込み"""
        for _ in range(self.read_int()):
            bone = Bone()
            bone.name = self.read_text()
            bone.english_name = self.read_text()
            bone.position = self.read_MVector3D()
            bone.parent_index = self.read_bone_index()
            bone.layer = self.read_int()
            bone.bone_flg = BoneFlg(self.read_short())

            if BoneFlg.TAIL_IS_BONE in bone.bone_flg:
                bone.tail_index = self.read_bone_index()
            else:
                bone.tail_position = self.read_MVector3D()

            if (
                BoneFlg.IS_EXTERNAL_TRANSLATION in bone.bone_flg
                or BoneFlg.IS_EXTERNAL_ROTATION in bone.bone_flg
            ):
                bone.effect_index = self.read_bone_index()
                bone.effect_factor = self.read_float()

            if BoneFlg.HAS_FIXED_AXIS in bone.bone_flg:
                bone.fixed_axis = self.read_MVector3D()

            if BoneFlg.HAS_LOCAL_COORDINATE in bone.bone_flg:
                bone.local_x_vector = self.read_MVector3D()
                bone.local_z_vector = self.read_MVector3D()

            if BoneFlg.IS_EXTERNAL_PARENT_DEFORM in bone.bone_flg:
                bone.external_key = self.read_int()

            if BoneFlg.IS_IK in bone.bone_flg:
                ik = Ik()
                ik.bone_index = self.read_bone_index()
                ik.loop_count = self.read_int()
                ik.unit_rotation.radians = MVector3D(self.read_float(), 0, 0)
                for _i in range(self.read_int()):
                    ik_link = IkLink()
                    ik_link.bone_index = self.read_bone_index()
                    ik_link.angle_limit = bool(self.read_byte())
                    if ik_link.angle_limit:
                        ik_link.min_angle_limit.radians = self.read_MVector3D()
                        ik_link.max_angle_limit.radians = self.read_MVector3D()
                    ik.links.append(ik_link)
                bone.ik = ik

            model.bones.append(bone)

    def read_morphs(self, model: PmxModel):
        """モーフデータ読み込み"""
        for _ in range(self.read_int()):
            morph = Morph()
            morph.name = self.read_text()
            morph.english_name = self.read_text()
            morph.panel = MorphPanel(self.read_byte())
            morph.morph_type = MorphType(self.read_byte())

            for _ in range(self.read_int()):
                if morph.morph_type == MorphType.GROUP:
                    morph.offsets.append(
                        GroupMorphOffset(self.read_morph_index(), self.read_float())
                    )
                elif morph.morph_type == MorphType.VERTEX:
                    morph.offsets.append(
                        VertexMorphOffset(
                            self.read_vertex_index(), self.read_MVector3D()
                        )
                    )
                elif morph.morph_type == MorphType.BONE:
                    morph.offsets.append(
                        BoneMorphOffset(
                            self.read_bone_index(),
                            self.read_MVector3D(),
                            self.read_MQuaternion(),
                        )
                    )
                elif morph.morph_type in [
                    MorphType.UV,
                    MorphType.EXTENDED_UV1,
                    MorphType.EXTENDED_UV2,
                    MorphType.EXTENDED_UV3,
                    MorphType.EXTENDED_UV4,
                ]:
                    morph.offsets.append(
                        UvMorphOffset(self.read_vertex_index(), self.read_MVector4D())
                    )
                elif morph.morph_type == MorphType.MATERIAL:
                    morph.offsets.append(
                        MaterialMorphOffset(
                            self.read_material_index(),
                            MaterialMorphCalcMode(self.read_byte()),
                            self.read_MVector4D(),
                            self.read_MVector3D(),
                            self.read_float(),
                            self.read_MVector3D(),
                            self.read_MVector4D(),
                            self.read_float(),
                            self.read_MVector4D(),
                            self.read_MVector4D(),
                            self.read_MVector4D(),
                        )
                    )

            model.morphs.append(morph)

    def read_display_slots(self, model: PmxModel):
        """表示枠データ読み込み"""
        for _ in range(self.read_int()):
            display_slot = DisplaySlot()
            display_slot.name = self.read_text()
            display_slot.english_name = self.read_text()
            display_slot.special_flg = Switch(self.read_byte())
            for _i in range(self.read_int()):
                reference = DisplaySlotReference()
                reference.display_type = DisplayType(self.read_byte())
                if reference.display_type == DisplayType.BONE:
                    reference.display_index = self.read_bone_index()
                else:
                    reference.display_index = self.read_morph_index()
                display_slot.references.append(reference)

            model.display_slots.append(display_slot)

    def read_rigidbodies(self, model: PmxModel):
        """剛体データ読み込み"""
        for _ in range(self.read_int()):
            rigidbody = RigidBody()
            rigidbody.name = self.read_text()
            rigidbody.english_name = self.read_text()

            shape_rotation_radians = MVector3D()

            (
                rigidbody.bone_index,
                rigidbody.collision_group,
                no_collision_group,
                rigidbody.shape_type,
                rigidbody.shape_size.x,
                rigidbody.shape_size.y,
                rigidbody.shape_size.z,
                rigidbody.shape_position.x,
                rigidbody.shape_position.y,
                rigidbody.shape_position.z,
                shape_rotation_radians.x,
                shape_rotation_radians.y,
                shape_rotation_radians.z,
                rigidbody.param.mass,
                rigidbody.param.linear_damping,
                rigidbody.param.angular_damping,
                rigidbody.param.restitution,
                rigidbody.param.friction,
                rigidbody.mode,
            ) = self.unpack(
                self.read_by_format[RigidBody].unpack,
                self.read_by_format[RigidBody].size,
            )

            rigidbody.no_collision_group = RigidBodyCollisionGroup(no_collision_group)
            rigidbody.shape_rotation.radians = shape_rotation_radians

            model.rigidbodies.append(rigidbody)

    def read_joints(self, model: PmxModel):
        """モデルデータ読み込み"""
        for _ in range(self.read_int()):
            joint = Joint()

            rotation_radians = MVector3D()
            rotation_limit_min_radians = MVector3D()
            rotation_limit_max_radians = MVector3D()

            joint.name = self.read_text()
            joint.english_name = self.read_text()
            (
                joint.joint_type,
                joint.rigidbody_index_a,
                joint.rigidbody_index_b,
                joint.position.x,
                joint.position.y,
                joint.position.z,
                rotation_radians.x,
                rotation_radians.y,
                rotation_radians.z,
                joint.param.translation_limit_min.x,
                joint.param.translation_limit_min.y,
                joint.param.translation_limit_min.z,
                joint.param.translation_limit_max.x,
                joint.param.translation_limit_max.y,
                joint.param.translation_limit_max.z,
                rotation_limit_min_radians.x,
                rotation_limit_min_radians.y,
                rotation_limit_min_radians.z,
                rotation_limit_max_radians.x,
                rotation_limit_max_radians.y,
                rotation_limit_max_radians.z,
                joint.param.spring_constant_translation.x,
                joint.param.spring_constant_translation.y,
                joint.param.spring_constant_translation.z,
                joint.param.spring_constant_rotation.x,
                joint.param.spring_constant_rotation.y,
                joint.param.spring_constant_rotation.z,
            ) = self.unpack(
                self.read_by_format[Joint].unpack,
                self.read_by_format[Joint].size,
            )

            joint.rotation.radians = rotation_radians
            joint.param.rotation_limit_min.radians = rotation_limit_min_radians
            joint.param.rotation_limit_max.radians = rotation_limit_max_radians

            model.joints.append(joint)

    def define_read_index(self, count: int, is_vertex=False):
        """
        INDEX読み取り定義

        Parameters
        ----------
        count : int
            Indexサイズ数
        is_vertex : bool
            頂点データの場合、1,2 は unsigned なので切り分け

        Returns
        -------
        function
            読み取り定義関数
        """
        if count == 1 and is_vertex:

            def read_index():
                return self.read_byte()

            return read_index, "B"
        elif count == 2 and is_vertex:

            def read_index():
                return self.read_ushort()

            return read_index, "H"
        elif count == 1 and not is_vertex:

            def read_index():
                return self.read_sbyte()

            return read_index, "b"
        elif count == 2 and not is_vertex:

            def read_index():
                return self.read_short()

            return read_index, "h"
        else:

            def read_index():
                return self.read_int()

            return read_index, "i"
