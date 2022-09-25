import os
from abc import ABC, abstractmethod
from enum import Flag, IntEnum, unique
from typing import List, Optional

import numpy as np
from base.base import BaseModel
from base.math import MQuaternion, MVector2D, MVector3D, MVector4D
from base.part import BaseIndexModel, BaseIndexNameModel, BaseRotationModel, Switch
from PIL import Image, ImageOps


@unique
class DeformType(IntEnum):
    """ウェイト変形方式"""

    BDEF1 = 0
    """0:BDEF1"""
    BDEF2 = 1
    """1:BDEF2"""
    BDEF4 = 2
    """2:BDEF4"""
    SDEF = 3
    """3:SDEF"""


class Deform(BaseModel, ABC):
    """
    デフォーム基底クラス

    Parameters
    ----------
    indecies : List[int]
        ボーンINDEXリスト
    weights : List[float]
        ウェイトリスト
    count : int
        デフォームボーン個数
    """

    def __init__(self, indecies: List[int], weights: List[float], count: int):
        super().__init__()
        self.indecies = np.array(indecies, dtype=np.int32)
        self.weights = np.array(weights, dtype=np.float64)
        self.count: int = count

    def get_indecies(self, weight_threshold: float = 0) -> np.ndarray:
        """
        デフォームボーンINDEXリスト取得

        Parameters
        ----------
        weight_threshold : float, optional
            ウェイト閾値, by default 0
            指定された場合、このweight以上のウェイトを持っているINDEXのみを取得する

        Returns
        -------
        np.ndarray
            デフォームボーンINDEXリスト
        """
        return self.indecies[self.weights >= weight_threshold]

    def get_weights(self, weight_threshold: float = 0) -> np.ndarray:
        """
        デフォームウェイトリスト取得

        Parameters
        ----------
        weight_threshold : float, optional
            ウェイト閾値, by default 0
            指定された場合、このweight以上のウェイトを持っているウェイトのみを取得する

        Returns
        -------
        np.ndarray
            デフォームウェイトリスト
        """
        return self.weights[self.weights >= weight_threshold]

    def normalize(self, align=False):
        """
        ウェイト正規化

        Parameters
        ----------
        align : bool, optional
            countのボーン数に揃えるか, by default False
        """
        if align:
            # 揃える必要がある場合
            # 数が足りるよう、かさ増しする
            ilist = np.array(self.indices.tolist() + [0, 0, 0, 0])
            wlist = np.array(self.weights.tolist() + [0, 0, 0, 0])
            # 正規化
            wlist /= wlist.sum(axis=0, keepdims=1)

            # ウェイトの大きい順に指定個数までを対象とする
            self.indecies = ilist[np.argsort(-wlist)][: self.count]
            self.weights = wlist[np.argsort(-wlist)][: self.count]

        # ウェイト正規化
        self.weights /= self.weights.sum(axis=0, keepdims=1)

    @abstractmethod
    def type(self) -> int:
        """
        デフォームタイプ
        """
        return -1


class Bdef1(Deform):
    def __init__(self, index0: int):
        super().__init__([index0], [1.0], 1)

    def type(self) -> int:
        return 0


class Bdef2(Deform):
    def __init__(self, index0: int, index1: int, weight0: float):
        super().__init__([index0, index1], [weight0, 1 - weight0], 2)

    def type(self) -> int:
        return 1


class Bdef4(Deform):
    def __init__(
        self,
        index0: int,
        index1: int,
        index2: int,
        index3: int,
        weight0: float,
        weight1: float,
        weight2: float,
        weight3: float,
    ):
        super().__init__(
            [index0, index1, index2, index3], [weight0, weight1, weight2, weight3], 4
        )

    def type(self) -> int:
        return 2


class Sdef(Deform):
    def __init__(
        self,
        index0: int,
        index1: int,
        weight0: float,
        sdef_c_x: float,
        sdef_c_y: float,
        sdef_c_z: float,
        sdef_r0_x: float,
        sdef_r0_y: float,
        sdef_r0_z: float,
        sdef_r1_x: float,
        sdef_r1_y: float,
        sdef_r1_z: float,
    ):
        super().__init__([index0, index1], [weight0, 1 - weight0], 2)
        self.sdef_c = MVector3D(sdef_c_x, sdef_c_y, sdef_c_z)
        self.sdef_r0 = MVector3D(sdef_r0_x, sdef_r0_y, sdef_r0_z)
        self.sdef_r1 = MVector3D(sdef_r1_x, sdef_r1_y, sdef_r1_z)

    def type(self) -> int:
        return 3


class Vertex(BaseIndexModel):
    """
    頂点

    Parameters
    ----------
    position : MVector3D, optional
        頂点位置, by default MVector3D()
    normal : MVector3D, optional
        頂点法線, by default MVector3D()
    uv : MVector2D, optional
        UV, by default MVector2D()
    extended_uvs : List[MVector4D], optional
        追加UV, by default []
    deform_type: DeformType, optional
        ウェイト変形方式 0:BDEF1 1:BDEF2 2:BDEF4 3:SDEF, by default DeformType.BDEF1
    deform : Deform, optional
        デフォーム, by default Deform([], [], 0)
    edge_factor : float, optional
        エッジ倍率, by default 0
    """

    def __init__(
        self,
        position: MVector3D = None,
        normal: MVector3D = None,
        uv: MVector2D = None,
        extended_uvs: List[MVector4D] = None,
        deform_type: DeformType = None,
        deform: Deform = None,
        edge_factor: float = None,
    ):
        super().__init__()
        self.position: MVector3D = position or MVector3D()
        self.normal: MVector3D = normal or MVector3D()
        self.uv: MVector2D = uv or MVector2D()
        self.extended_uvs: List[MVector4D] = extended_uvs or []
        self.deform_type: DeformType = deform_type or DeformType.BDEF1
        self.deform: Deform = deform or Bdef1(-1)
        self.edge_factor: float = edge_factor or 0


class Face(BaseIndexModel):
    """
    面データ

    Parameters
    ----------
    vertex_index0 : int
        頂点0
    vertex_index1 : int
        頂点1
    vertex_index2 : int
        頂点2
    """

    def __init__(self, vertex_index0: int, vertex_index1: int, vertex_index2: int):
        super().__init__()
        self.vertices = [vertex_index0, vertex_index1, vertex_index2]


@unique
class TextureType(IntEnum):
    TEXTURE = 0
    TOON = 1
    SPHERE = 2


class Texture(BaseIndexModel):
    """
    テクスチャ

    Parameters
    ----------
    texture_path : str
        テクスチャパス
    """

    def __init__(self, texture_path: str):
        super().__init__()
        self.texture_path = texture_path
        self.for_draw = False


@unique
class SphereMode(IntEnum):
    """スフィアモード"""

    INVALID = 0
    """0:無効"""
    MULTIPLICATION = 1
    """1:乗算(sph)"""
    ADDITION = 2
    """2:加算(spa)"""
    SUBTEXTURE = 3
    """3:サブテクスチャ(追加UV1のx,yをUV参照して通常テクスチャ描画を行う)"""


@unique
class DrawFlg(Flag):
    """描画フラグ"""

    NONE = 0x0000
    """"初期値"""
    DOUBLE_SIDED_DRAWING = 0x0001
    """0x01:両面描画"""
    GROUND_SHADOW = 0x0002
    """0x02:地面影"""
    DRAWING_ON_SELF_SHADOW_MAPS = 0x0004
    """0x04:セルフシャドウマップへの描画"""
    DRAWING_SELF_SHADOWS = 0x0008
    """0x08:セルフシャドウの描画"""
    DRAWING_EDGE = 0x0010
    """0x10:エッジ描画"""


@unique
class ToonSharing(IntEnum):
    """スフィアモード"""

    INDIVIDUAL = 0
    """0:継続値は個別Toon"""
    SHARING = 1
    """1:継続値は共有Toon"""


class Material(BaseIndexNameModel):
    """
    材質

    Parameters
    ----------
    name : str, optional
        材質名, by default ""
    english_name : str, optional
        材質名英, by default ""
    diffuse_color : MVector4D, optional
        Diffuse (R,G,B,A), by default MVector4D()
    specular_color : MVector3D, optional
        Specular (R,G,B), by default MVector3D()
    specular_factor : float, optional
        Specular係数, by default 0
    ambient_color : MVector3D, optional
        Ambient (R,G,B), by default MVector3D()
    draw_flg : DrawFlg, optional
        描画フラグ(8bit) - 各bit 0:OFF 1:ON
        0x01:両面描画, 0x02:地面影, 0x04:セルフシャドウマップへの描画, 0x08:セルフシャドウの描画, 0x10:エッジ描画, by default DrawFlg.NONE
    edge_color : MVector4D, optional
        エッジ色 (R,G,B,A), by default MVector4D()
    edge_size : float, optional
        エッジサイズ, by default 0
    texture_index : int, optional
        通常テクスチャINDEX, by default -1
    sphere_texture_index : int, optional
        スフィアテクスチャINDEX, by default -1
    sphere_mode : SphereMode, optional
        スフィアモード 0:無効 1:乗算(sph) 2:加算(spa) 3:サブテクスチャ(追加UV1のx,yをUV参照して通常テクスチャ描画を行う), by default INVALID
    toon_sharing_flg : Switch, optional
        共有Toonフラグ 0:継続値は個別Toon 1:継続値は共有Toon, by default OFF
    toon_texture_index : int, optional
        ToonテクスチャINDEX, by default -1
    comment : str, optional
        メモ, by default ""
    vertices_count : int, optional
        材質に対応する面(頂点)数 (必ず3の倍数になる), by default 0
    """

    def __init__(
        self,
        name: str = None,
        english_name: str = None,
        diffuse_color: MVector4D = None,
        specular_color: MVector3D = None,
        specular_factor: float = None,
        ambient_color: MVector3D = None,
        draw_flg: DrawFlg = None,
        edge_color: MVector4D = None,
        edge_size: float = None,
        texture_index: int = None,
        sphere_texture_index: int = None,
        sphere_mode: SphereMode = None,
        toon_sharing_flg: ToonSharing = None,
        toon_texture_index: int = None,
        comment: str = None,
        vertices_count: int = None,
    ):
        super().__init__(name=name or "", english_name=english_name or "")
        self.diffuse_color: MVector4D = diffuse_color or MVector4D()
        self.specular_color: MVector3D = specular_color or MVector3D()
        self.specular_factor: float = specular_factor or 0
        self.ambient_color: MVector3D = ambient_color or MVector3D()
        self.draw_flg: DrawFlg = draw_flg or DrawFlg.NONE
        self.edge_color: MVector4D = edge_color or MVector4D()
        self.edge_size: float = edge_size or 0
        self.texture_index: int = texture_index or -1
        self.sphere_texture_index: int = sphere_texture_index or -1
        self.sphere_mode: SphereMode = sphere_mode or SphereMode.INVALID
        self.toon_sharing_flg: ToonSharing = toon_sharing_flg or ToonSharing.SHARING
        self.toon_texture_index: int = toon_texture_index or -1
        self.comment: str = comment or ""
        self.vertices_count: int = vertices_count or 0


class IkLink(BaseModel):
    """
    IKリンク

    Parameters
    ----------
    bone_index : int, optional
        リンクボーンのボーンIndex, by default -1
    angle_limit : bool, optional
        角度制限 0:OFF 1:ON, by default False
    min_angle_limit_radians : MVector3D, optional
        下限 (x,y,z) -> ラジアン角, by default MVector3D()
    max_angle_limit_radians : MVector3D, optional
        上限 (x,y,z) -> ラジアン角, by default MVector3D()
    """

    def __init__(
        self,
        bone_index: int = None,
        angle_limit: bool = None,
        min_angle_limit_radians: MVector3D = None,
        max_angle_limit_radians: MVector3D = None,
    ):
        super().__init__()
        self.bone_index: int = bone_index or -1
        self.angle_limit: bool = angle_limit or False
        self.min_angle_limit: BaseRotationModel = BaseRotationModel(
            min_angle_limit_radians or MVector3D()
        )
        self.max_angle_limit: BaseRotationModel = BaseRotationModel(
            max_angle_limit_radians or MVector3D()
        )


class Ik(BaseModel):
    """
    IK

    Parameters
    ----------
    bone_index : int, optional
        IKターゲットボーンのボーンIndex, by default -1
    loop_count : int, optional
        IKループ回数 (最大255), by default 0
    unit_radians : float, optional
        IKループ計算時の1回あたりの制限角度 -> ラジアン角, by default 0
        unit_rotation の x に値が入っている
    links : List[IkLink], optional
        IKリンクリスト, by default []
    """

    def __init__(
        self,
        bone_index: int = None,
        loop_count: int = None,
        unit_radians: float = None,
        links: List[IkLink] = None,
    ):
        super().__init__()
        self.bone_index = bone_index or -1
        self.loop_count = loop_count or 0
        self.unit_rotation: BaseRotationModel = BaseRotationModel(
            MVector3D(unit_radians or 0, 0, 0)
        )
        self.links: List[IkLink] = links or []


@unique
class BoneFlg(Flag):
    """ボーンフラグ"""

    NONE = 0x0000
    """"初期値"""
    TAIL_IS_BONE = 0x0001
    """接続先(PMD子ボーン指定)表示方法 -> 0:座標オフセットで指定 1:ボーンで指定"""
    CAN_ROTATE = 0x0002
    """回転可能"""
    CAN_TRANSLATE = 0x0004
    """移動可能"""
    IS_VISIBLE = 0x0008
    """表示"""
    CAN_MANIPULATE = 0x0010
    """操作可"""
    IS_IK = 0x0020
    """IK"""
    IS_EXTERNAL_LOCAL = 0x0080
    """ローカル付与 | 付与対象 0:ユーザー変形値／IKリンク／多重付与 1:親のローカル変形量"""
    IS_EXTERNAL_ROTATION = 0x0100
    """回転付与"""
    IS_EXTERNAL_TRANSLATION = 0x0200
    """移動付与"""
    HAS_FIXED_AXIS = 0x0400
    """軸固定"""
    HAS_LOCAL_COORDINATE = 0x0800
    """ローカル軸"""
    IS_AFTER_PHYSICS_DEFORM = 0x1000
    """物理後変形"""
    IS_EXTERNAL_PARENT_DEFORM = 0x2000
    """外部親変形"""


class Bone(BaseIndexNameModel):
    """
    ボーン

    Parameters
    ----------
    name : str, optional
        ボーン名, by default ""
    english_name : str, optional
        ボーン名英, by default ""
    position : MVector3D, optional
        位置, by default MVector3D()
    parent_index : int, optional
        親ボーンのボーンIndex, by default -1
    layer : int, optional
        変形階層, by default 0
    bone_flg : BoneFlg, optional
        ボーンフラグ(16bit) 各bit 0:OFF 1:ON, by default BoneFlg.NONE
    tail_position : MVector3D, optional
        接続先:0 の場合 座標オフセット, ボーン位置からの相対分, by default MVector3D()
    tail_index : int, optional
        接続先:1 の場合 接続先ボーンのボーンIndex, by default -1
    effect_index : int, optional
        回転付与:1 または 移動付与:1 の場合 付与親ボーンのボーンIndex, by default -1
    effect_factor : float, optional
        付与率, by default 0
    fixed_axis : MVector3D, optional
        軸固定:1 の場合 軸の方向ベクトル, by default MVector3D()
    local_x_vector : MVector3D, optional
        ローカル軸:1 の場合 X軸の方向ベクトル, by default MVector3D()
    local_z_vector : MVector3D, optional
        ローカル軸:1 の場合 Z軸の方向ベクトル, by default MVector3D()
    external_key : int, optional
        外部親変形:1 の場合 Key値, by default -1
    ik : Optional[Ik], optional
        IK:1 の場合 IKデータを格納, by default None
    is_system : bool, optional
        システム計算用追加ボーン, by default False
    """

    def __init__(
        self,
        name: str = None,
        english_name: str = None,
        position: MVector3D = None,
        parent_index: int = None,
        layer: int = None,
        bone_flg: BoneFlg = None,
        tail_position: MVector3D = None,
        tail_index: int = None,
        effect_index: int = None,
        effect_factor: float = None,
        fixed_axis: MVector3D = None,
        local_x_vector: MVector3D = None,
        local_z_vector: MVector3D = None,
        external_key: int = None,
        ik: Optional[Ik] = None,
        display: bool = None,
        is_system: bool = None,
    ):
        super().__init__(name=name or "", english_name=english_name or "")
        self.position: MVector3D = position or MVector3D()
        self.parent_index: int = parent_index or -1
        self.layer: int = layer or 0
        self.bone_flg: BoneFlg = bone_flg or BoneFlg.NONE
        self.tail_position: MVector3D = tail_position or MVector3D()
        self.tail_index: int = tail_index or -1
        self.effect_index: int = effect_index or -1
        self.effect_factor: float = effect_factor or 0
        self.fixed_axis: MVector3D = fixed_axis or MVector3D()
        self.local_x_vector: MVector3D = local_x_vector or MVector3D()
        self.local_z_vector: MVector3D = local_z_vector or MVector3D()
        self.external_key: int = external_key or -1
        self.ik: Optional[Ik] = ik or None
        self.display: bool = display or False
        self.is_system: bool = is_system or False


class BoneTree(BaseModel):
    """ボーンリンク"""

    __slots__ = ["childs"]

    def __init__(self, bone: Bone) -> None:
        super().__init__()
        self.bone = bone
        self.children: list[BoneTree] = []

    def make_tree(
        self, bones: list[Bone], bone_link_indecies: list[tuple[int, int]], index: int
    ):
        if index >= len(bone_link_indecies):
            return
        child_index = bone_link_indecies[index][1]
        is_exist_child = False
        for ci, child in enumerate(self.children):
            if child.bone.index == child_index:
                self.children[ci].make_tree(bones, bone_link_indecies, index + 1)
                is_exist_child = True
                break

        if not is_exist_child:
            self.children.append(BoneTree(bones[child_index]))
            self.children[-1].make_tree(bones, bone_link_indecies, index + 1)


class MorphOffset(BaseModel):
    """モーフオフセット基底クラス"""

    def __init__(self):
        super().__init__()


class VertexMorphOffset(MorphOffset):
    """
    頂点モーフ

    Parameters
    ----------
    vertex_index : int
        頂点INDEX
    position_offset : MVector3D
        座標オフセット量(x,y,z)
    """

    def __init__(self, vertex_index: int, position_offset: MVector3D):
        super().__init__()
        self.vertex_index = vertex_index
        self.position_offset = position_offset


class UvMorphOffset(MorphOffset):
    """
    UVモーフ

    Parameters
    ----------
    vertex_index : int
        頂点INDEX
    uv : MVector4D
        UVオフセット量(x,y,z,w) ※通常UVはz,wが不要項目になるがモーフとしてのデータ値は記録しておく
    """

    def __init__(self, vertex_index: int, uv: MVector4D):
        super().__init__()
        self.vertex_index = vertex_index
        self.uv = uv


class BoneMorphOffset(MorphOffset):
    """
    ボーンモーフ

    Parameters
    ----------
    bone_index : int
        ボーンIndex
    position : MVector3D
        移動量(x,y,z)
    rotation : MQuaternion
        回転量-クォータニオン(x,y,z,w)
    """

    def __init__(self, bone_index: int, position: MVector3D, rotation: MQuaternion):
        super().__init__()
        self.bone_index = bone_index
        self.position = position
        self.rotation = rotation


class GroupMorphOffset(MorphOffset):
    """
    グループモーフ

    Parameters
    ----------
    morph_index : int
        モーフINDEX
    morph_factor : float
        モーフ変動量
    """

    def __init__(self, morph_index: int, morph_factor: float):
        super().__init__()
        self.morph_index = morph_index
        self.morph_factor = morph_factor


@unique
class MaterialMorphCalcMode(IntEnum):
    """材質モーフ：計算モード"""

    MULTIPLICATION = 0
    """0:乗算"""
    ADDITION = 1
    """1:加算"""


class MaterialMorphOffset(MorphOffset):
    """
    材質モーフ

    Parameters
    ----------
    material_index : int
        材質Index -> -1:全材質対象
    calc_mode : CalcMode
        0:乗算, 1:加算
    diffuse : MVector4D
        Diffuse (R,G,B,A)
    specular : MVector3D
        Specular (R,G,B)
    specular_factor : float
        Specular係数
    ambient : MVector3D
        Ambient (R,G,B)
    edge_color : MVector4D
        エッジ色 (R,G,B,A)
    edge_size : float
        エッジサイズ
    texture_factor : MVector4D
        テクスチャ係数 (R,G,B,A)
    sphere_texture_factor : MVector4D
        スフィアテクスチャ係数 (R,G,B,A)
    toon_texture_factor : MVector4D
        Toonテクスチャ係数 (R,G,B,A)
    """

    def __init__(
        self,
        material_index: int,
        calc_mode: MaterialMorphCalcMode,
        diffuse: MVector4D,
        specular: MVector3D,
        specular_factor: float,
        ambient: MVector3D,
        edge_color: MVector4D,
        edge_size: float,
        texture_factor: MVector4D,
        sphere_texture_factor: MVector4D,
        toon_texture_factor: MVector4D,
    ):
        super().__init__()
        self.material_index = material_index
        self.calc_mode = calc_mode
        self.diffuse = diffuse
        self.specular = specular
        self.specular_factor = specular_factor
        self.ambient = ambient
        self.edge_color = edge_color
        self.edge_size = edge_size
        self.texture_factor = texture_factor
        self.sphere_texture_factor = sphere_texture_factor
        self.toon_texture_factor = toon_texture_factor


@unique
class MorphPanel(IntEnum):
    """操作パネル"""

    SYSTEM = 0
    """0:システム予約"""
    EYEBROW_LOWER_LEFT = 1
    """1:眉(左下)"""
    EYE_UPPER_LEFT = 2
    """2:目(左上)"""
    LIP_UPPER_RIGHT = 3
    """3:口(右上)"""
    OTHER_LOWER_RIGHT = 4
    """4:その他(右下)"""

    @property
    def panel_name(self):
        if self.value == 1:
            return "眉"
        elif self.value == 2:
            return "目"
        elif self.value == 3:
            return "口"
        elif self.value == 4:
            return "他"
        else:
            return "システム"


@unique
class MorphType(IntEnum):
    """モーフ種類"""

    GROUP = 0
    """0:グループ"""
    VERTEX = 1
    """1:頂点"""
    BONE = 2
    """2:ボーン"""
    UV = 3
    """3:UV"""
    EXTENDED_UV1 = 4
    """4:追加UV1"""
    EXTENDED_UV2 = 5
    """5:追加UV2"""
    EXTENDED_UV3 = 6
    """6:追加UV3"""
    EXTENDED_UV4 = 7
    """7:追加UV4"""
    MATERIAL = 8
    """"8:材質"""


class Morph(BaseIndexNameModel):
    """
    _summary_

    Parameters
    ----------
    name : str, optional
        モーフ名, by default ""
    english_name : str, optional
        モーフ名英, by default ""
    panel : MorphPanel, optional
        モーフパネル, by default MorphPanel.UPPER_LEFT_EYE
    morph_type : MorphType, optional
        モーフ種類, by default MorphType.GROUP
    offsets : List[TMorphOffset], optional
        モーフオフセット
    """

    def __init__(
        self,
        name: str = None,
        english_name: str = None,
        panel: MorphPanel = None,
        morph_type: MorphType = None,
        offsets: List[MorphOffset] = None,
    ):
        super().__init__(name=name or "", english_name=english_name or "")
        self.panel: MorphPanel = panel or MorphPanel.EYE_UPPER_LEFT
        self.morph_type: MorphType = morph_type or MorphType.GROUP
        self.offsets: List[MorphOffset] = offsets or []


@unique
class DisplayType(IntEnum):
    """表示枠要素タイプ"""

    BONE = 0
    """0:ボーン"""
    MORPH = 1
    """1:モーフ"""


class DisplaySlotReference(BaseModel):
    """
    表示枠要素

    Parameters
    ----------
    display_type : DisplayType, optional
        要素対象 0:ボーン 1:モーフ, by default DisplayType.BONE
    display_index : int, optional
        ボーンIndex or モーフIndex, by default -1
    """

    def __init__(self, display_type: DisplayType = None, display_index: int = None):
        super().__init__()
        self.display_type = display_type or DisplayType.BONE
        self.display_index = display_index or -1


class DisplaySlot(BaseIndexNameModel):
    """
    表示枠

    Parameters
    ----------
    name : str, optional
        枠名, by default ""
    english_name : str, optional
        枠名英, by default ""
    special_flg : Switch, optional
        特殊枠フラグ - 0:通常枠 1:特殊枠, by default Switch.OFF
    references : List[DisplaySlotReference], optional
        表示枠要素, by default []
    """

    def __init__(
        self, name: str = None, english_name: str = None, special_flg: Switch = None
    ):
        super().__init__(name=name or "", english_name=english_name or "")
        self.special_flg = special_flg or Switch.OFF
        self.references: List[DisplaySlotReference] = []


class RigidBodyParam(BaseModel):
    """
    剛体パラ

    Parameters
    ----------
    mass : float, optional
        質量, by default 0
    linear_damping : float, optional
        移動減衰, by default 0
    angular_damping : float, optional
        回転減衰, by default 0
    restitution : float, optional
        反発力, by default 0
    friction : float, optional
        摩擦力, by default 0
    """

    def __init__(
        self,
        mass: float = None,
        linear_damping: float = None,
        angular_damping: float = None,
        restitution: float = None,
        friction: float = None,
    ) -> None:
        super().__init__()
        self.mass = mass or 0
        self.linear_damping = linear_damping or 0
        self.angular_damping = angular_damping or 0
        self.restitution = restitution or 0
        self.friction = friction or 0


@unique
class RigidBodyShape(IntEnum):
    """剛体の形状"""

    SPHERE = 0
    """0:球"""
    BOX = 1
    """1:箱"""
    CAPSULE = 2
    """2:カプセル"""


@unique
class RigidBodyMode(IntEnum):
    """剛体物理の計算モード"""

    STATIC = 0
    """0:ボーン追従(static)"""
    DYNAMIC = 1
    """1:物理演算(dynamic)"""
    DYNAMIC_BONE = 2
    """2:物理演算 + Bone位置合わせ"""


@unique
class RigidBodyCollisionGroup(Flag):
    """剛体の衝突グループ"""

    NONE = 0x0000
    """0:グループなし"""
    GROUP01 = 0x0001
    GROUP02 = 0x0002
    GROUP03 = 0x0004
    GROUP04 = 0x0008
    GROUP05 = 0x0010
    GROUP06 = 0x0020
    GROUP07 = 0x0040
    GROUP08 = 0x0080
    GROUP09 = 0x0100
    GROUP10 = 0x0200
    GROUP11 = 0x0400
    GROUP12 = 0x0800
    GROUP13 = 0x1000
    GROUP14 = 0x2000
    GROUP15 = 0x4000
    GROUP16 = 0x8000


class RigidBody(BaseIndexNameModel):
    """
    剛体

    Parameters
    ----------
    name : str, optional
        剛体名, by default ""
    english_name : str, optional
        剛体名英, by default ""
    bone_index : int, optional
        関連ボーンIndex, by default -1
    collision_group : int, optional
        グループ, by default 0
    no_collision_group : RigidBodyCollisionGroup, optional
        非衝突グループフラグ, by default 0
    shape_type : RigidBodyShape, optional
        形状, by default RigidBodyShape.SPHERE
    shape_size : MVector3D, optional
        サイズ(x,y,z), by default MVector3D()
    shape_position : MVector3D, optional
        位置(x,y,z), by default MVector3D()
    shape_rotation_radians : MVector3D, optional
        回転(x,y,z) -> ラジアン角, by default MVector3D()
    mass : float, optional
        質量, by default 0
    linear_damping : float, optional
        移動減衰, by default 0
    angular_damping : float, optional
        回転減衰, by default 0
    restitution : float, optional
        反発力, by default 0
    friction : float, optional
        摩擦力, by default 0
    mode : RigidBodyMode, optional
        剛体の物理演算, by default RigidBodyMode.STATIC
    """

    def __init__(
        self,
        name: str = None,
        english_name: str = None,
        bone_index: int = None,
        collision_group: int = None,
        no_collision_group: RigidBodyCollisionGroup = None,
        shape_type: RigidBodyShape = None,
        shape_size: MVector3D = None,
        shape_position: MVector3D = None,
        shape_rotation: BaseRotationModel = None,
        param: RigidBodyParam = None,
        mode: RigidBodyMode = None,
    ) -> None:
        super().__init__(name=name or "", english_name=english_name or "")
        self.bone_index: int = bone_index or -1
        self.collision_group: int = collision_group or 0
        self.no_collision_group: RigidBodyCollisionGroup = (
            no_collision_group or RigidBodyCollisionGroup.NONE
        )
        self.shape_type: RigidBodyShape = shape_type or RigidBodyShape.SPHERE
        self.shape_size: MVector3D = shape_size or MVector3D()
        self.shape_position: MVector3D = shape_position or MVector3D()
        self.shape_rotation: BaseRotationModel = shape_rotation or BaseRotationModel()
        self.param: RigidBodyParam = param or RigidBodyParam()
        self.mode: RigidBodyMode = mode or RigidBodyMode.STATIC
        # 軸方向
        self.x_direction = MVector3D()
        self.y_direction = MVector3D()
        self.z_direction = MVector3D()


class JointLimitParam(BaseModel):
    """
    ジョイント制限パラメーター

    Parameters
    ----------
    limit_min : MVector3D, optional
        制限最小角度, by default MVector3D()
    limit_max : MVector3D, optional
        制限最大角度, by default MVector3D()
    """

    def __init__(
        self,
        limit_min: MVector3D = None,
        limit_max: MVector3D = None,
    ) -> None:
        super().__init__()
        self.limit_min = limit_min or MVector3D()
        self.limit_max = limit_max or MVector3D()


class JointParam(BaseModel):
    """
    ジョイントパラメーター

    Parameters
    ----------
    translation_limit_min : MVector3D, optional
        移動制限-下限(x,y,z), by default MVector3D()
    translation_limit_max : MVector3D, optional
        移動制限-上限(x,y,z), by default MVector3D()
    rotation_limit_min : BaseRotationModel, optional
        回転制限-下限(x,y,z) -> ラジアン角, by default BaseRotationModel()
    rotation_limit_max : BaseRotationModel, optional
        回転制限-上限(x,y,z) -> ラジアン角, by default BaseRotationModel()
    spring_constant_translation : MVector3D, optional
        バネ定数-移動(x,y,z), by default MVector3D()
    spring_constant_rotation : MVector3D, optional
        バネ定数-回転(x,y,z), by default MVector3D()
    """

    def __init__(
        self,
        translation_limit_min: MVector3D = None,
        translation_limit_max: MVector3D = None,
        rotation_limit_min_radians: MVector3D = None,
        rotation_limit_max_radians: MVector3D = None,
        spring_constant_translation: MVector3D = None,
        spring_constant_rotation: MVector3D = None,
    ) -> None:
        super().__init__()
        self.translation_limit_min = translation_limit_min or MVector3D()
        self.translation_limit_max = translation_limit_max or MVector3D()
        self.rotation_limit_min: BaseRotationModel = BaseRotationModel(
            rotation_limit_min_radians or MVector3D()
        )
        self.rotation_limit_max: BaseRotationModel = BaseRotationModel(
            rotation_limit_max_radians or MVector3D()
        )
        self.spring_constant_translation = spring_constant_translation or MVector3D()
        self.spring_constant_rotation = spring_constant_rotation or MVector3D()


class Joint(BaseIndexNameModel):
    """
    ジョイント

    Parameters
    ----------
    name : str, optional
        Joint名, by default ""
    english_name : str, optional
        Joint名英, by default ""
    joint_type : int, optional
        Joint種類, by default 0
    rigidbody_index_a : int, optional
        関連剛体AのIndex, by default -1
    rigidbody_index_b : int, optional
        関連剛体BのIndex, by default -1
    position : MVector3D, optional
        位置(x,y,z), by default MVector3D()
    rotation : BaseRotationModel, optional
        回転(x,y,z) -> ラジアン角, by default BaseRotationModel()
    translation_limit_min : MVector3D, optional
        移動制限-下限(x,y,z), by default MVector3D()
    translation_limit_max : MVector3D, optional
        移動制限-上限(x,y,z), by default MVector3D()
    rotation_limit_min : BaseRotationModel, optional
        回転制限-下限(x,y,z) -> ラジアン角, by default BaseRotationModel()
    rotation_limit_max : BaseRotationModel, optional
        回転制限-上限(x,y,z) -> ラジアン角, by default BaseRotationModel()
    spring_constant_translation : MVector3D, optional
        バネ定数-移動(x,y,z), by default MVector3D()
    spring_constant_rotation : MVector3D, optional
        バネ定数-回転(x,y,z), by default MVector3D()
    """

    def __init__(
        self,
        name: str = None,
        english_name: str = None,
        joint_type: int = None,
        rigidbody_index_a: int = None,
        rigidbody_index_b: int = None,
        position: MVector3D = None,
        rotation: BaseRotationModel = None,
        param=None,
    ) -> None:
        super().__init__(name=name or "", english_name=english_name or "")
        self.joint_type: int = joint_type or 0
        self.rigidbody_index_a: int = rigidbody_index_a or -1
        self.rigidbody_index_b: int = rigidbody_index_b or -1
        self.position: MVector3D = position or MVector3D()
        self.rotation: BaseRotationModel = rotation or BaseRotationModel()
        self.param = param or JointParam()
