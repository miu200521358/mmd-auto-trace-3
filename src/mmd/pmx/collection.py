import numpy as np
from base.collection import BaseHashModel, BaseIndexDictModel, BaseIndexListModel, BaseIndexNameListModel
from base.math import MVector3D
from mmd.pmx.part import (
    Bone,
    BoneFlg,
    BoneTree,
    DisplaySlot,
    DrawFlg,
    Face,
    Joint,
    Material,
    Morph,
    RigidBody,
    Texture,
    TextureType,
    ToonSharing,
    Vertex,
)


class Vertices(BaseIndexListModel[Vertex]):
    """
    頂点リスト
    """

    def __init__(self):
        super().__init__()


class Faces(BaseIndexListModel[Face]):
    """
    面リスト
    """

    def __init__(self):
        super().__init__()


class Textures(BaseIndexListModel[Texture]):
    """
    テクスチャリスト
    """

    def __init__(self):
        super().__init__()


class ToonTextures(BaseIndexDictModel[Texture]):
    """
    共有テクスチャ辞書
    """

    def __init__(self):
        super().__init__()


class Materials(BaseIndexNameListModel[Material]):
    """
    材質リスト
    """

    def __init__(self):
        super().__init__()


class Bones(BaseIndexNameListModel[Bone]):
    """
    ボーンリスト
    """

    def __init__(self):
        super().__init__()

    def get_max_layer(self) -> int:
        """
        最大変形階層を取得

        Returns
        -------
        int
            最大変形階層
        """
        return max([b.layer for b in self.data])

    def get_bone_name_by_layer(self) -> list[str]:
        """
        レイヤー順ボーン名リスト

        Returns
        -------
        list[str]
            レイヤー順ボーン名リスト
        """
        return [b.name for layer in range(self.get_max_layer() + 1) for b in self.data if b.layer == layer]

    def create_bone_links(self) -> dict[int, BoneTree]:
        # 根元ボーンリスト（親ボーンがないボーンリスト）
        bone_trees: dict[int, BoneTree] = dict(
            [(bidx, BoneTree(self[bidx])) for bidx in list(set([b.index for b in self.data if 0 > b.parent_index]))]
        )

        # 親ボーンとして登録されているボーンリスト
        parent_indices = list(set([b.parent_index for b in self.data]))
        # 末端ボーンリスト（親ボーンとして登録が1件もないボーンのリスト）
        for end_bone_index in [b.index for b in self.data if b.index not in parent_indices and b.index not in list(bone_trees.keys())]:
            # レイヤー込みのINDEXリスト取得
            bone_link_indecies = sorted(self.create_bone_link_indecies(end_bone_index))
            bone_trees[bone_link_indecies[0][1]].make_tree(self.data, bone_link_indecies, index=1)

        return bone_trees

    def create_bone_link_indecies(self, child_idx: int, bone_link_indecies=None) -> list[tuple[int, int]]:
        # 階層＞リスト順（＞FK＞IK＞付与）
        if not bone_link_indecies:
            bone_link_indecies = []

        for b in reversed(self.data):
            if b.index == self[child_idx].parent_index:
                bone_link_indecies.append((b.layer, b.index))
                return self.create_bone_link_indecies(b.index, bone_link_indecies)

        return bone_link_indecies

    # ローカルX軸の取得
    def get_local_x_axis(self, bone_name: str):
        if bone_name not in self:
            return MVector3D()

        bone = self[bone_name]
        to_pos = MVector3D()

        if bone.fixed_axis != MVector3D():
            # 軸制限がある場合、親からの向きを保持
            fixed_x_axis = bone.fixed_axis.normalized()
        else:
            fixed_x_axis = MVector3D()

        from_pos = self[bone.name].position
        if BoneFlg.TAIL_IS_BONE not in bone.bone_flg and bone.tail_position != MVector3D():
            # 表示先が相対パスの場合、保持
            to_pos = from_pos + bone.tail_position
        elif BoneFlg.TAIL_IS_BONE in bone.bone_flg and bone.tail_index >= 0 and bone.tail_index in self:
            # 表示先が指定されているの場合、保持
            to_pos = self[bone.tail_index].position
        else:
            # 表示先がない場合、とりあえず親ボーンからの向きにする
            from_pos = self[bone.parent_index].position
            to_pos = self[bone.name].position

        # 軸制限の指定が無い場合、子の方向
        x_axis = (to_pos - from_pos).normalized()

        if fixed_x_axis != MVector3D() and np.sign(fixed_x_axis.x) != np.sign(x_axis.x):
            # 軸制限の軸方向と計算上の軸方向が違う場合、逆ベクトル
            x_axis = -fixed_x_axis

        return x_axis


class Morphs(BaseIndexNameListModel[Morph]):
    """
    モーフリスト
    """

    def __init__(self):
        super().__init__()


class DisplaySlots(BaseIndexNameListModel[DisplaySlot]):
    """
    表示枠リスト
    """

    def __init__(
        self,
    ):
        super().__init__()


class RigidBodies(BaseIndexNameListModel[RigidBody]):
    """
    剛体リスト
    """

    def __init__(self):
        super().__init__()


class Joints(BaseIndexNameListModel[Joint]):
    """
    ジョイントリスト
    """

    def __init__(self):
        super().__init__()


class PmxModel(BaseHashModel):
    """
    Pmxモデルデータ

    Parameters
    ----------
    path : str, optional
        パス, by default ""
    signature : str, optional
        signature, by default ""
    version : float, optional
        バージョン, by default 0.0
    extended_uv_count : int, optional
        追加UV数, by default 0
    vertex_count : int, optional
        頂点数, by default 0
    texture_count : int, optional
        テクスチャ数, by default 0
    material_count : int, optional
        材質数, by default 0
    bone_count : int, optional
        ボーン数, by default 0
    morph_count : int, optional
        モーフ数, by default 0
    rigidbody_count : int, optional
        剛体数, by default 0
    name : str, optional
        モデル名, by default ""
    english_name : str, optional
        モデル名英, by default ""
    comment : str, optional
        コメント, by default ""
    english_comment : str, optional
        コメント英, by default ""
    json_data : dict, optional
        JSONデータ（vroidデータ用）, by default {}
    """

    def __init__(
        self,
        path: str = None,
    ):
        super().__init__(path=path or "")
        self.signature: str = ""
        self.version: float = 0.0
        self.extended_uv_count: int = 0
        self.vertex_count: int = 0
        self.texture_count: int = 0
        self.material_count: int = 0
        self.bone_count: int = 0
        self.morph_count: int = 0
        self.rigidbody_count: int = 0
        self.name: str = ""
        self.english_name: str = ""
        self.comment: str = ""
        self.english_comment: str = ""
        self.json_data: dict = {}
        self.vertices = Vertices()
        self.faces = Faces()
        self.textures = Textures()
        self.toon_textures = ToonTextures()
        self.materials = Materials()
        self.bones = Bones()
        self.morphs = Morphs()
        self.display_slots = DisplaySlots()
        self.rigidbodies = RigidBodies()
        self.joints = Joints()
        self.for_draw = False
        self.meshs = None

    def get_name(self) -> str:
        return self.name
