import struct
from math import isinf, isnan

from base.base import BaseModel
from base.logger import MLogger
from mmd.pmx.collection import PmxModel
from mmd.pmx.part import (Bdef1, Bdef2, Bdef4, Bone, BoneFlg, BoneMorphOffset,
                          DeformType, DisplaySlot, DisplaySlotReference,
                          DisplayType, DrawFlg, Face, GroupMorphOffset, Ik,
                          IkLink, Joint, Material, MaterialMorphCalcMode,
                          MaterialMorphOffset, Morph, MorphPanel, MorphType,
                          RigidBody, RigidBodyCollisionGroup, Sdef, Texture,
                          ToonSharing, UvMorphOffset, Vertex,
                          VertexMorphOffset)
from tqdm import tqdm

logger = MLogger(__name__)

TYPE_FLOAT = "f"
TYPE_BOOL = "c"
TYPE_BYTE = "<b"
TYPE_UNSIGNED_BYTE = "<B"
TYPE_SHORT = "<h"
TYPE_UNSIGNED_SHORT = "<H"
TYPE_INT = "<i"
TYPE_UNSIGNED_INT = "<I"
TYPE_LONG = "<l"
TYPE_UNSIGNED_LONG = "<L"


class PmxWriter(BaseModel):
    @classmethod
    def write(cls, model: PmxModel, output_path: str):
        with open(output_path, "wb") as fout:
            # シグニチャ
            fout.write(b"PMX ")
            fout.write(struct.pack(TYPE_FLOAT, float(2)))
            # 後続するデータ列のバイトサイズ  PMX2.0は 8 で固定
            fout.write(struct.pack(TYPE_BYTE, int(8)))
            # エンコード方式  | 0:UTF16
            fout.write(struct.pack(TYPE_BYTE, 0))
            # 追加UV数
            fout.write(struct.pack(TYPE_BYTE, model.extended_uv_count))
            # 頂点Indexサイズ | 1,2,4 のいずれか
            vertex_idx_size, vertex_idx_type = define_write_index(len(model.vertices))
            fout.write(struct.pack(TYPE_BYTE, vertex_idx_size))
            # テクスチャIndexサイズ | 1,2,4 のいずれか
            texture_idx_size, texture_idx_type = define_write_index(len(model.textures))
            fout.write(struct.pack(TYPE_BYTE, texture_idx_size))
            # 材質Indexサイズ | 1,2,4 のいずれか
            material_idx_size, material_idx_type = define_write_index(len(model.materials))
            fout.write(struct.pack(TYPE_BYTE, material_idx_size))
            # ボーンIndexサイズ | 1,2,4 のいずれか
            bone_idx_size, bone_idx_type = define_write_index(len(model.bones))
            fout.write(struct.pack(TYPE_BYTE, bone_idx_size))
            # モーフIndexサイズ | 1,2,4 のいずれか
            morph_idx_size, morph_idx_type = define_write_index(len(model.morphs))
            fout.write(struct.pack(TYPE_BYTE, morph_idx_size))
            # 剛体Indexサイズ | 1,2,4 のいずれか
            rigidbody_idx_size, rigidbody_idx_type = define_write_index(len(model.rigidbodies))
            fout.write(struct.pack(TYPE_BYTE, rigidbody_idx_size))

            # モデル名(日本語)
            write_text(fout, model.name, "Vrm Model")
            # モデル名(英語)
            write_text(fout, model.english_name, "Vrm Model")
            # コメント(日本語)
            write_text(fout, model.comment, "")
            # コメント(英語)
            write_text(fout, model.english_comment, "")

            fout.write(struct.pack(TYPE_INT, len(model.vertices)))

            # 頂点データ
            for vertex in tqdm(model.vertices):
                # position
                write_number(fout, TYPE_FLOAT, float(vertex.position.x))
                write_number(fout, TYPE_FLOAT, float(vertex.position.y))
                write_number(fout, TYPE_FLOAT, float(vertex.position.z))
                # normal
                write_number(fout, TYPE_FLOAT, float(vertex.normal.x))
                write_number(fout, TYPE_FLOAT, float(vertex.normal.y))
                write_number(fout, TYPE_FLOAT, float(vertex.normal.z))
                # uv
                write_number(fout, TYPE_FLOAT, float(vertex.uv.x))
                write_number(fout, TYPE_FLOAT, float(vertex.uv.y))
                # 追加uv
                for uv in vertex.extended_uvs:
                    write_number(fout, TYPE_FLOAT, float(uv.x))
                    write_number(fout, TYPE_FLOAT, float(uv.y))
                    write_number(fout, TYPE_FLOAT, float(uv.z))
                    write_number(fout, TYPE_FLOAT, float(uv.w))

                # deform
                if type(vertex.deform) is Bdef1:
                    fout.write(struct.pack(TYPE_BYTE, 0))
                    write_number(fout, bone_idx_type, vertex.deform.indecies[0])
                elif type(vertex.deform) is Bdef2:
                    fout.write(struct.pack(TYPE_BYTE, 1))
                    write_number(fout, bone_idx_type, vertex.deform.indecies[0])
                    write_number(fout, bone_idx_type, vertex.deform.indecies[1])

                    write_number(fout, TYPE_FLOAT, vertex.deform.weights[0], True)
                elif type(vertex.deform) is Bdef4:
                    fout.write(struct.pack(TYPE_BYTE, 2))
                    write_number(fout, bone_idx_type, vertex.deform.indecies[0])
                    write_number(fout, bone_idx_type, vertex.deform.indecies[1])
                    write_number(fout, bone_idx_type, vertex.deform.indecies[2])
                    write_number(fout, bone_idx_type, vertex.deform.indecies[3])

                    write_number(fout, TYPE_FLOAT, vertex.deform.weights[0], True)
                    write_number(fout, TYPE_FLOAT, vertex.deform.weights[1], True)
                    write_number(fout, TYPE_FLOAT, vertex.deform.weights[2], True)
                    write_number(fout, TYPE_FLOAT, vertex.deform.weights[3], True)
                elif type(vertex.deform) is Sdef:
                    fout.write(struct.pack(TYPE_BYTE, 3))
                    write_number(fout, bone_idx_type, vertex.deform.indecies[0])
                    write_number(fout, bone_idx_type, vertex.deform.indecies[1])
                    write_number(fout, TYPE_FLOAT, vertex.deform.weights[0], True)
                    write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_c.x))
                    write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_c.y))
                    write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_c.z))
                    write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_r0.x))
                    write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_r0.y))
                    write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_r0.z))
                    write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_r1.x))
                    write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_r1.y))
                    write_number(fout, TYPE_FLOAT, float(vertex.deform.sdef_r1.z))
                else:
                    logger.error("頂点deformなし: {vertex}", vertex=str(vertex))

                write_number(fout, TYPE_FLOAT, float(vertex.edge_factor), True)

            logger.debug("-- 頂点データ出力終了({count})", count=len(model.vertices))

            # 面の数
            fout.write(struct.pack(TYPE_INT, len(model.faces) * 3))

            # 面データ
            for face in tqdm(model.faces):
                for vidx in face.vertices:
                    fout.write(struct.pack(vertex_idx_type, vidx))

            logger.debug("-- 面データ出力終了({count})", count=len(model.faces))

            # テクスチャの数
            fout.write(struct.pack(TYPE_INT, len(model.textures)))

            # テクスチャデータ
            for tex_path in model.textures:
                write_text(fout, tex_path, "")

            logger.debug("-- テクスチャデータ出力終了({count})", count=len(model.textures))

            # 材質の数
            fout.write(struct.pack(TYPE_INT, len(model.materials)))

            # 材質データ
            for midx, material in enumerate(tqdm(model.materials)):
                # 材質名
                write_text(fout, material.name, f"Material {midx}")
                write_text(fout, material.english_name, f"Material {midx}")
                # Diffuse
                write_number(fout, TYPE_FLOAT, float(material.diffuse_color.x), True)
                write_number(fout, TYPE_FLOAT, float(material.diffuse_color.y), True)
                write_number(fout, TYPE_FLOAT, float(material.diffuse_color.z), True)
                write_number(fout, TYPE_FLOAT, float(material.diffuse_color.w), True)
                # Specular
                write_number(fout, TYPE_FLOAT, float(material.specular_color.x), True)
                write_number(fout, TYPE_FLOAT, float(material.specular_color.y), True)
                write_number(fout, TYPE_FLOAT, float(material.specular_color.z), True)
                # Specular係数
                write_number(fout, TYPE_FLOAT, float(material.specular_factor), True)
                # Ambient
                write_number(fout, TYPE_FLOAT, float(material.ambient_color.x), True)
                write_number(fout, TYPE_FLOAT, float(material.ambient_color.y), True)
                write_number(fout, TYPE_FLOAT, float(material.ambient_color.z), True)
                # 描画フラグ(8bit)
                fout.write(struct.pack(TYPE_BYTE, material.draw_flg.value))
                # エッジ色 (R,G,B,A)
                write_number(fout, TYPE_FLOAT, float(material.edge_color.x), True)
                write_number(fout, TYPE_FLOAT, float(material.edge_color.y), True)
                write_number(fout, TYPE_FLOAT, float(material.edge_color.z), True)
                write_number(fout, TYPE_FLOAT, float(material.edge_color.w), True)
                # エッジサイズ
                write_number(fout, TYPE_FLOAT, float(material.edge_size), True)
                # 通常テクスチャ
                fout.write(struct.pack(texture_idx_type, material.texture_index))
                # スフィアテクスチャ
                fout.write(struct.pack(texture_idx_type, material.sphere_texture_index))
                # スフィアモード
                fout.write(struct.pack(TYPE_BYTE, material.sphere_mode))
                # 共有Toonフラグ
                fout.write(struct.pack(TYPE_BYTE, material.toon_sharing_flg))
                if material.toon_sharing_flg == ToonSharing.INDIVIDUAL:
                    # 個別Toonテクスチャ
                    fout.write(struct.pack(texture_idx_type, material.toon_texture_index))
                else:
                    # 共有Toonテクスチャ[0～9]
                    fout.write(struct.pack(TYPE_BYTE, material.toon_texture_index))
                # コメント
                write_text(fout, material.comment, "")
                # 材質に対応する面(頂点)数
                write_number(fout, TYPE_INT, material.vertices_count)

            logger.debug("-- 材質データ出力終了({count})", count=len(model.materials))

            # ボーンの数
            fout.write(struct.pack(TYPE_INT, len(model.bones)))

            for bidx, bone in enumerate(tqdm(model.bones)):
                # ボーン名
                write_text(fout, bone.name, f"Bone {bidx}")
                write_text(fout, bone.english_name, f"Bone {bidx}")
                # position
                write_number(fout, TYPE_FLOAT, float(bone.position.x))
                write_number(fout, TYPE_FLOAT, float(bone.position.y))
                write_number(fout, TYPE_FLOAT, float(bone.position.z))
                # 親ボーンのボーンIndex
                fout.write(struct.pack(bone_idx_type, bone.parent_index))
                # 変形階層
                write_number(fout, TYPE_INT, bone.layer, True)
                # ボーンフラグ
                fout.write(struct.pack(TYPE_SHORT, bone.bone_flg.value))

                if BoneFlg.TAIL_IS_BONE in bone.bone_flg:
                    # 接続先ボーンのボーンIndex
                    fout.write(struct.pack(bone_idx_type, bone.tail_index))
                else:
                    # 接続先位置
                    write_number(fout, TYPE_FLOAT, float(bone.tail_position.x))
                    write_number(fout, TYPE_FLOAT, float(bone.tail_position.y))
                    write_number(fout, TYPE_FLOAT, float(bone.tail_position.z))

                if BoneFlg.IS_EXTERNAL_TRANSLATION in bone.bone_flg or BoneFlg.IS_EXTERNAL_ROTATION in bone.bone_flg:
                    # 付与親指定ありの場合
                    fout.write(struct.pack(bone_idx_type, bone.effect_index))
                    write_number(fout, TYPE_FLOAT, bone.effect_factor)

                if BoneFlg.HAS_FIXED_AXIS in bone.bone_flg:
                    # 軸制限先
                    write_number(fout, TYPE_FLOAT, float(bone.fixed_axis.x))
                    write_number(fout, TYPE_FLOAT, float(bone.fixed_axis.y))
                    write_number(fout, TYPE_FLOAT, float(bone.fixed_axis.z))

                if BoneFlg.HAS_LOCAL_COORDINATE in bone.bone_flg:
                    # ローカルX
                    write_number(fout, TYPE_FLOAT, float(bone.local_x_vector.x))
                    write_number(fout, TYPE_FLOAT, float(bone.local_x_vector.y))
                    write_number(fout, TYPE_FLOAT, float(bone.local_x_vector.z))
                    # ローカルZ
                    write_number(fout, TYPE_FLOAT, float(bone.local_z_vector.x))
                    write_number(fout, TYPE_FLOAT, float(bone.local_z_vector.y))
                    write_number(fout, TYPE_FLOAT, float(bone.local_z_vector.z))

                if BoneFlg.IS_EXTERNAL_PARENT_DEFORM in bone.bone_flg:
                    write_number(fout, TYPE_INT, bone.external_key)

                if BoneFlg.IS_IK in bone.bone_flg:
                    # IKボーン
                    # n  : ボーンIndexサイズ  | IKターゲットボーンのボーンIndex
                    fout.write(struct.pack(bone_idx_type, bone.ik.bone_index))
                    # 4  : int  	| IKループ回数
                    write_number(fout, TYPE_INT, bone.ik.loop_count)
                    # 4  : float	| IKループ計算時の1回あたりの制限角度 -> ラジアン角
                    write_number(fout, TYPE_FLOAT, bone.ik.unit_rotation.radians.x)
                    # 4  : int  	| IKリンク数 : 後続の要素数
                    write_number(fout, TYPE_INT, len(bone.ik.links))

                    for link in bone.ik.links:
                        # n  : ボーンIndexサイズ  | リンクボーンのボーンIndex
                        fout.write(struct.pack(bone_idx_type, link.bone_index))
                        # 1  : byte	| 角度制限 0:OFF 1:ON
                        fout.write(struct.pack(TYPE_BYTE, int(link.angle_limit)))

                        if link.angle_limit == 1:
                            write_number(fout, TYPE_FLOAT, float(link.min_angle_limit.radians.x))
                            write_number(fout, TYPE_FLOAT, float(link.min_angle_limit.radians.y))
                            write_number(fout, TYPE_FLOAT, float(link.min_angle_limit.radians.z))

                            write_number(fout, TYPE_FLOAT, float(link.max_angle_limit.radians.x))
                            write_number(fout, TYPE_FLOAT, float(link.max_angle_limit.radians.y))
                            write_number(fout, TYPE_FLOAT, float(link.max_angle_limit.radians.z))

            logger.debug("-- ボーンデータ出力終了({count})", count=len(model.bones))

            # モーフの数
            write_number(fout, TYPE_INT, len(model.morphs))

            for midx, morph in enumerate(tqdm(model.morphs)):
                # モーフ名
                write_text(fout, morph.name, f"Morph {midx}")
                write_text(fout, morph.english_name, f"Morph {midx}")
                # 操作パネル (PMD:カテゴリ) 1:眉(左下) 2:目(左上) 3:口(右上) 4:その他(右下)  | 0:システム予約
                fout.write(struct.pack(TYPE_BYTE, morph.panel))
                # モーフ種類 - 0:グループ, 1:頂点, 2:ボーン, 3:UV, 4:追加UV1, 5:追加UV2, 6:追加UV3, 7:追加UV4, 8:材質
                fout.write(struct.pack(TYPE_BYTE, morph.morph_type))
                # モーフのオフセット数 : 後続の要素数
                write_number(fout, TYPE_INT, len(morph.offsets))

                for offset in morph.offsets:
                    if type(offset) is VertexMorphOffset:
                        # 頂点モーフ
                        fout.write(struct.pack(vertex_idx_type, offset.vertex_index))
                        write_number(fout, TYPE_FLOAT, float(offset.position_offset.x))
                        write_number(fout, TYPE_FLOAT, float(offset.position_offset.y))
                        write_number(fout, TYPE_FLOAT, float(offset.position_offset.z))
                    elif type(offset) is UvMorphOffset:
                        # UVモーフ
                        fout.write(struct.pack(vertex_idx_type, offset.vertex_index))
                        write_number(fout, TYPE_FLOAT, float(offset.uv.x))
                        write_number(fout, TYPE_FLOAT, float(offset.uv.y))
                        write_number(fout, TYPE_FLOAT, float(offset.uv.z))
                        write_number(fout, TYPE_FLOAT, float(offset.uv.w))
                    elif type(offset) is BoneMorphOffset:
                        # ボーンモーフ
                        fout.write(struct.pack(bone_idx_type, offset.bone_index))
                        write_number(fout, TYPE_FLOAT, float(offset.position.x))
                        write_number(fout, TYPE_FLOAT, float(offset.position.y))
                        write_number(fout, TYPE_FLOAT, float(offset.position.z))
                        write_number(fout, TYPE_FLOAT, float(offset.rotation.x))
                        write_number(fout, TYPE_FLOAT, float(offset.rotation.y))
                        write_number(fout, TYPE_FLOAT, float(offset.rotation.z))
                        write_number(fout, TYPE_FLOAT, float(offset.rotation.scalar))
                    elif type(offset) is MaterialMorphOffset:
                        # 材質モーフ
                        fout.write(struct.pack(material_idx_type, offset.material_index))
                        fout.write(struct.pack(TYPE_BYTE, int(offset.calc_mode)))
                        write_number(fout, TYPE_FLOAT, float(offset.diffuse.x))
                        write_number(fout, TYPE_FLOAT, float(offset.diffuse.y))
                        write_number(fout, TYPE_FLOAT, float(offset.diffuse.z))
                        write_number(fout, TYPE_FLOAT, float(offset.diffuse.w))
                        write_number(fout, TYPE_FLOAT, float(offset.specular.x))
                        write_number(fout, TYPE_FLOAT, float(offset.specular.y))
                        write_number(fout, TYPE_FLOAT, float(offset.specular.z))
                        write_number(fout, TYPE_FLOAT, float(offset.specular_factor))
                        write_number(fout, TYPE_FLOAT, float(offset.ambient.x))
                        write_number(fout, TYPE_FLOAT, float(offset.ambient.y))
                        write_number(fout, TYPE_FLOAT, float(offset.ambient.z))
                        write_number(fout, TYPE_FLOAT, float(offset.edge_color.x))
                        write_number(fout, TYPE_FLOAT, float(offset.edge_color.y))
                        write_number(fout, TYPE_FLOAT, float(offset.edge_color.z))
                        write_number(fout, TYPE_FLOAT, float(offset.edge_color.w))
                        write_number(fout, TYPE_FLOAT, float(offset.edge_size))
                        write_number(fout, TYPE_FLOAT, float(offset.texture_factor.x))
                        write_number(fout, TYPE_FLOAT, float(offset.texture_factor.y))
                        write_number(fout, TYPE_FLOAT, float(offset.texture_factor.z))
                        write_number(fout, TYPE_FLOAT, float(offset.texture_factor.w))
                        write_number(fout, TYPE_FLOAT, float(offset.sphere_texture_factor.x))
                        write_number(fout, TYPE_FLOAT, float(offset.sphere_texture_factor.y))
                        write_number(fout, TYPE_FLOAT, float(offset.sphere_texture_factor.z))
                        write_number(fout, TYPE_FLOAT, float(offset.sphere_texture_factor.w))
                        write_number(fout, TYPE_FLOAT, float(offset.toon_texture_factor.x))
                        write_number(fout, TYPE_FLOAT, float(offset.toon_texture_factor.y))
                        write_number(fout, TYPE_FLOAT, float(offset.toon_texture_factor.z))
                        write_number(fout, TYPE_FLOAT, float(offset.toon_texture_factor.w))
                    elif type(offset) is GroupMorphOffset:
                        # グループモーフ
                        fout.write(struct.pack(morph_idx_type, offset.morph_index))
                        write_number(fout, TYPE_FLOAT, float(offset.morph_factor))

            logger.debug("-- モーフデータ出力終了({count})", count=len(model.morphs))

            # 表示枠の数
            write_number(fout, TYPE_INT, len(model.display_slots))

            for didx, display_slot in enumerate(tqdm(model.display_slots)):
                # 表示枠名
                write_text(fout, display_slot.name, f"Display {didx}")
                write_text(fout, display_slot.english_name, f"Display {didx}")
                # 特殊枠フラグ - 0:通常枠 1:特殊枠
                fout.write(struct.pack(TYPE_BYTE, display_slot.special_flg.value))
                # 枠内要素数
                write_number(fout, TYPE_INT, len(display_slot.references))
                # ボーンの場合
                for reference in display_slot.references:
                    # 要素対象 0:ボーン 1:モーフ
                    fout.write(struct.pack(TYPE_BYTE, reference.display_type))
                    if reference.display_type == 0:
                        # ボーンIndex
                        fout.write(struct.pack(bone_idx_type, reference.display_index))
                    else:
                        # モーフIndex
                        fout.write(struct.pack(morph_idx_type, reference.display_index))

            logger.debug("-- 表示枠データ出力終了({count})", count=len(model.display_slots))

            # 剛体の数
            write_number(fout, TYPE_INT, len(list(model.rigidbodies)))

            for ridx, rigidbody in enumerate(tqdm(model.rigidbodies)):
                # 剛体名
                write_text(fout, rigidbody.name, f"Rigidbody {ridx}")
                write_text(fout, rigidbody.english_name, f"Rigidbody {ridx}")
                # ボーンIndex
                fout.write(struct.pack(bone_idx_type, rigidbody.bone_index))
                # 1  : byte	| グループ
                fout.write(struct.pack(TYPE_BYTE, rigidbody.collision_group))
                # 2  : ushort	| 非衝突グループフラグ
                fout.write(struct.pack(TYPE_UNSIGNED_SHORT, rigidbody.no_collision_group))
                # 1  : byte	| 形状 - 0:球 1:箱 2:カプセル
                fout.write(struct.pack(TYPE_BYTE, rigidbody.shape_type))
                # 12 : float3	| サイズ(x,y,z)
                write_number(fout, TYPE_FLOAT, float(rigidbody.shape_size.x), True)
                write_number(fout, TYPE_FLOAT, float(rigidbody.shape_size.y), True)
                write_number(fout, TYPE_FLOAT, float(rigidbody.shape_size.z), True)
                # 12 : float3	| 位置(x,y,z)
                write_number(fout, TYPE_FLOAT, float(rigidbody.shape_position.x))
                write_number(fout, TYPE_FLOAT, float(rigidbody.shape_position.y))
                write_number(fout, TYPE_FLOAT, float(rigidbody.shape_position.z))
                # 12 : float3	| 回転(x,y,z)
                write_number(fout, TYPE_FLOAT, float(rigidbody.shape_rotation.radians.x))
                write_number(fout, TYPE_FLOAT, float(rigidbody.shape_rotation.radians.y))
                write_number(fout, TYPE_FLOAT, float(rigidbody.shape_rotation.radians.z))
                # 4  : float	| 質量
                write_number(fout, TYPE_FLOAT, float(rigidbody.param.mass), True)
                # 4  : float	| 移動減衰
                write_number(fout, TYPE_FLOAT, float(rigidbody.param.linear_damping), True)
                # 4  : float	| 回転減衰
                write_number(fout, TYPE_FLOAT, float(rigidbody.param.angular_damping), True)
                # 4  : float	| 反発力
                write_number(fout, TYPE_FLOAT, float(rigidbody.param.restitution), True)
                # 4  : float	| 摩擦力
                write_number(fout, TYPE_FLOAT, float(rigidbody.param.friction), True)
                # 1  : byte	| 剛体の物理演算 - 0:ボーン追従(static) 1:物理演算(dynamic) 2:物理演算 + Bone位置合わせ
                fout.write(struct.pack(TYPE_BYTE, rigidbody.mode))

            logger.debug("-- 剛体データ出力終了({count})", count=len(model.rigidbodies))

            # ジョイントの数
            write_number(fout, TYPE_INT, len(list(model.joints)))

            for jidx, joint in enumerate(tqdm(model.joints)):
                # ジョイント名
                write_text(fout, joint.name, f"Joint {jidx}")
                write_text(fout, joint.english_name, f"Joint {jidx}")
                # 1  : byte	| Joint種類 - 0:スプリング6DOF   | PMX2.0では 0 のみ(拡張用)
                fout.write(struct.pack(TYPE_BYTE, joint.joint_type))
                # n  : 剛体Indexサイズ  | 関連剛体AのIndex - 関連なしの場合は-1
                fout.write(struct.pack(rigidbody_idx_type, joint.rigidbody_index_a))
                # n  : 剛体Indexサイズ  | 関連剛体BのIndex - 関連なしの場合は-1
                fout.write(struct.pack(rigidbody_idx_type, joint.rigidbody_index_b))
                # 12 : float3	| 位置(x,y,z)
                write_number(fout, TYPE_FLOAT, float(joint.position.x))
                write_number(fout, TYPE_FLOAT, float(joint.position.y))
                write_number(fout, TYPE_FLOAT, float(joint.position.z))
                # 12 : float3	| 回転(x,y,z) -> ラジアン角
                write_number(fout, TYPE_FLOAT, float(joint.rotation.x))
                write_number(fout, TYPE_FLOAT, float(joint.rotation.y))
                write_number(fout, TYPE_FLOAT, float(joint.rotation.z))
                # 12 : float3	| 移動制限-下限(x,y,z)
                write_number(fout, TYPE_FLOAT, float(joint.param.translation_limit_min.x))
                write_number(fout, TYPE_FLOAT, float(joint.param.translation_limit_min.y))
                write_number(fout, TYPE_FLOAT, float(joint.param.translation_limit_min.z))
                # 12 : float3	| 移動制限-上限(x,y,z)
                write_number(fout, TYPE_FLOAT, float(joint.param.translation_limit_max.x))
                write_number(fout, TYPE_FLOAT, float(joint.param.translation_limit_max.y))
                write_number(fout, TYPE_FLOAT, float(joint.param.translation_limit_max.z))
                # 12 : float3	| 回転制限-下限(x,y,z) -> ラジアン角
                write_number(fout, TYPE_FLOAT, float(joint.param.rotation_limit_min.radians.x))
                write_number(fout, TYPE_FLOAT, float(joint.param.rotation_limit_min.radians.y))
                write_number(fout, TYPE_FLOAT, float(joint.param.rotation_limit_min.radians.z))
                # 12 : float3	| 回転制限-上限(x,y,z) -> ラジアン角
                write_number(fout, TYPE_FLOAT, float(joint.param.rotation_limit_max.radians.x))
                write_number(fout, TYPE_FLOAT, float(joint.param.rotation_limit_max.radians.y))
                write_number(fout, TYPE_FLOAT, float(joint.param.rotation_limit_max.radians.z))
                # 12 : float3	| バネ定数-移動(x,y,z)
                write_number(fout, TYPE_FLOAT, float(joint.param.spring_constant_translation.x))
                write_number(fout, TYPE_FLOAT, float(joint.param.spring_constant_translation.y))
                write_number(fout, TYPE_FLOAT, float(joint.param.spring_constant_translation.z))
                # 12 : float3	| バネ定数-回転(x,y,z)
                write_number(fout, TYPE_FLOAT, float(joint.param.spring_constant_rotation.x))
                write_number(fout, TYPE_FLOAT, float(joint.param.spring_constant_rotation.y))
                write_number(fout, TYPE_FLOAT, float(joint.param.spring_constant_rotation.z))

            logger.debug("-- ジョイントデータ出力終了({count})", count=len(model.joints))


def define_write_index(size: int, is_vertex=False) -> tuple[int, str]:
    if 256 > size and is_vertex:
        return 4, TYPE_UNSIGNED_BYTE
    elif 256 <= size <= 65535 and is_vertex:
        return 2, TYPE_UNSIGNED_SHORT
    elif 128 > size and not is_vertex:
        return 1, TYPE_BYTE
    elif 128 <= size <= 32767 and not is_vertex:
        return 2, TYPE_SHORT
    else:
        return 1, TYPE_INT


def write_text(fout, text: str, default_text: str, type=TYPE_INT):
    try:
        btxt = text.encode("utf-16-le")
    except Exception:
        btxt = default_text.encode("utf-16-le")
    fout.write(struct.pack(type, len(btxt)))
    fout.write(btxt)


def write_number(fout, val_type: str, val: float, is_positive_only=False):
    if isnan(val) or isinf(val):
        # 正常な値を強制設定
        val = 0
    val = max(0, val) if is_positive_only else val

    try:
        # INT型の場合、INT変換
        if val_type in [TYPE_FLOAT]:
            fout.write(struct.pack(val_type, float(val)))
        else:
            fout.write(struct.pack(val_type, int(val)))
    except Exception as e:
        logger.error("val_type in [float]: %s", val_type in [TYPE_FLOAT])
        logger.error("write_number失敗: type: %s, val: %s, int(val): %s", val_type, val, int(val))
        raise e
