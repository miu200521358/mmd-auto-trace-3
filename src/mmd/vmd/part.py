from base.base import BaseModel
from base.bezier import Interpolation
from base.math import MMatrix4x4, MQuaternion, MVector3D
from base.part import BaseIndexModel, BaseIndexNameModel, BaseRotationModel


class BaseVmdFrame(BaseIndexModel):
    """
    VMD用基底クラス

    Parameters
    ----------
    index : int, optional
        キーフレ, by default None
    regist : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    def __init__(
        self,
        index: int = None,
        regist: bool = None,
        read: bool = None,
    ):
        super().__init__(index or 0)
        self.regist = regist or False
        self.read = read or False


class BaseVmdNameFrame(BaseIndexNameModel):
    """
    VMD用基底クラス(名前あり)

    Parameters
    ----------
    index : int, optional
        キーフレ, by default None
    name : str, optional
        名前, by default None
    regist : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    def __init__(
        self,
        index: int = None,
        name: str = None,
        regist: bool = None,
        read: bool = None,
    ):
        super().__init__(index or 0, name or "")
        self.regist = regist or False
        self.read = read or False


# https://hariganep.seesaa.net/article/201103article_1.html
class BoneInterpolations(BaseModel):
    """
    ボーンキーフレ用補間曲線

    Parameters
    ----------
    translation_x : Interpolation, optional
        移動X, by default None
    translation_y : Interpolation, optional
        移動Y, by default None
    translation_z : Interpolation, optional
        移動Z, by default None
    rotation : Interpolation, optional
        回転, by default None
    """

    def __init__(
        self,
        translation_x: Interpolation = None,
        translation_y: Interpolation = None,
        translation_z: Interpolation = None,
        rotation: Interpolation = None,
        vals: list[int] = None,
    ):
        self.translation_x: Interpolation = translation_x or Interpolation()
        self.translation_y: Interpolation = translation_y or Interpolation()
        self.translation_z: Interpolation = translation_z or Interpolation()
        self.rotation: Interpolation = rotation or Interpolation()
        self.vals = vals or [
            20,
            20,
            0,
            0,
            20,
            20,
            20,
            20,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            20,
            20,
            20,
            20,
            20,
            20,
            20,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            0,
            20,
            20,
            20,
            20,
            20,
            20,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            0,
            0,
            20,
            20,
            20,
            20,
            20,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            107,
            0,
            0,
            0,
        ]

    def merge(self) -> list[int]:
        return [
            self.translation_x.start.x,
            self.translation_y.start.x,
            self.translation_z.start.x,
            self.rotation.start.x,
            self.translation_x.start.y,
            self.translation_y.start.y,
            self.translation_z.start.y,
            self.rotation.start.y,
            self.translation_x.end.x,
            self.translation_y.end.x,
            self.translation_z.end.x,
            self.rotation.end.x,
            self.translation_x.end.y,
            self.translation_y.end.y,
            self.translation_z.end.y,
            self.rotation.end.y,
            self.translation_y.start.x,
            self.translation_z.start.x,
            self.rotation.start.x,
            self.translation_x.start.y,
            self.translation_y.start.y,
            self.translation_z.start.y,
            self.rotation.start.y,
            self.translation_x.end.x,
            self.translation_y.end.x,
            self.translation_z.end.x,
            self.rotation.end.x,
            self.translation_x.end.y,
            self.translation_y.end.y,
            self.translation_z.end.y,
            self.rotation.end.y,
            self.vals[31],
            self.translation_z.start.x,
            self.rotation.start.x,
            self.translation_x.start.y,
            self.translation_y.start.y,
            self.translation_z.start.y,
            self.rotation.start.y,
            self.translation_x.end.x,
            self.translation_y.end.x,
            self.translation_z.end.x,
            self.rotation.end.x,
            self.translation_x.end.y,
            self.translation_y.end.y,
            self.translation_z.end.y,
            self.rotation.end.y,
            self.vals[46],
            self.vals[47],
            self.rotation.start.x,
            self.translation_x.start.y,
            self.translation_y.start.y,
            self.translation_z.start.y,
            self.rotation.start.y,
            self.translation_x.end.x,
            self.translation_y.end.x,
            self.translation_z.end.x,
            self.rotation.end.x,
            self.translation_x.end.y,
            self.translation_y.end.y,
            self.translation_z.end.y,
            self.rotation.end.y,
            self.vals[61],
            self.vals[62],
            self.vals[63],
        ]


class VmdBoneFrame(BaseVmdNameFrame):
    """
    VMDのボーン1フレーム

    Parameters
    ----------
    name : str, optional
        ボーン名, by default None
    index : int, optional
        キーフレ, by default None
    position : MVector3D, optional
        位置, by default None
    rotation : MQuaternion, optional
        回転, by default None
    interpolations : Interpolations, optional
        補間曲線, by default None
    regist : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    def __init__(
        self,
        name: str = None,
        index: int = None,
        position: MVector3D = None,
        rotation: MQuaternion = None,
        matrix: MMatrix4x4 = None,
        interpolations: BoneInterpolations = None,
        regist: bool = None,
        read: bool = None,
    ):
        super().__init__(index, name, regist, read)
        self.position: MVector3D = position or MVector3D()
        self.rotation: MQuaternion = rotation or MQuaternion()
        self.interpolations: BoneInterpolations = interpolations or BoneInterpolations()
        self.matrix: MMatrix4x4 = matrix or MMatrix4x4(identity=True)

    def init_matrix(self):
        self.matrix = MMatrix4x4(identity=True)
        self.matrix.translate(self.position)
        self.matrix.rotate(self.rotation)


class VmdMorphFrame(BaseVmdNameFrame):
    """
    VMDのモーフ1フレーム

    Parameters
    ----------
    name : str, optional
        モーフ名, by default None
    index : int, optional
        キーフレ, by default None
    ratio : float, optional
        変化量, by default None
    regist : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    def __init__(
        self,
        index: int = None,
        name: str = None,
        ratio: float = None,
        regist: bool = None,
        read: bool = None,
    ):
        super().__init__(index, name, regist, read)
        self.name: str = name or ""
        self.ratio: float = ratio or 0.0


class CameraInterpolations(BaseModel):
    """
    カメラ補間曲線

    Parameters
    ----------
    translation_x : Interpolation, optional
        移動X, by default None
    translation_y : Interpolation, optional
        移動Y, by default None
    translation_z : Interpolation, optional
        移動Z, by default None
    rotation : Interpolation, optional
        回転, by default None
    distance : Interpolation, optional
        距離, by default None
    viewing_angle : Interpolation, optional
        視野角, by default None
    """

    def __init__(
        self,
        translation_x: Interpolation = None,
        translation_y: Interpolation = None,
        translation_z: Interpolation = None,
        rotation: Interpolation = None,
        distance: Interpolation = None,
        viewing_angle: Interpolation = None,
    ):
        self.translation_x: Interpolation = translation_x or Interpolation()
        self.translation_y: Interpolation = translation_y or Interpolation()
        self.translation_z: Interpolation = translation_z or Interpolation()
        self.rotation: Interpolation = rotation or Interpolation()
        self.distance: Interpolation = distance or Interpolation()
        self.viewing_angle: Interpolation = viewing_angle or Interpolation()


class VmdCameraFrame(BaseVmdFrame):
    """
    カメラキーフレ

    Parameters
    ----------
    index : int, optional
        キーフレ, by default None
    position : MVector3D, optional
        位置, by default None
    rotation : BaseRotationModel, optional
        回転, by default None
    distance : float, optional
        距離, by default None
    viewing_angle : int, optional
        視野角, by default None
    perspective : bool, optional
        パース, by default None
    interpolations : CameraInterpolations, optional
        補間曲線, by default None
    regist : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    def __init__(
        self,
        index: int = None,
        position: MVector3D = None,
        rotation: BaseRotationModel = None,
        distance: float = None,
        viewing_angle: int = None,
        perspective: bool = None,
        interpolations: CameraInterpolations = None,
        regist: bool = None,
        read: bool = None,
    ):
        super().__init__(index, regist, read)
        self.position: MVector3D = position or MVector3D()
        self.rotation: BaseRotationModel = rotation or BaseRotationModel()
        self.distance: float = distance or 0.0
        self.viewing_angle: int = viewing_angle or 0
        self.perspective: bool = perspective or False
        self.interpolations: CameraInterpolations = (
            interpolations or CameraInterpolations()
        )


class VmdLightFrame(BaseVmdFrame):
    """
    照明キーフレ

    Parameters
    ----------
    index : int, optional
        キーフレ, by default None
    color : MVector3D, optional
        色, by default None
    position : MVector3D, optional
        位置, by default None
    regist : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    def __init__(
        self,
        index: int = None,
        color: MVector3D = None,
        position: MVector3D = None,
        regist: bool = None,
        read: bool = None,
    ):
        super().__init__(index, regist, read)
        self.color: MVector3D = color or MVector3D()
        self.position: MVector3D = position or MVector3D()


class VmdShadowFrame(BaseVmdFrame):
    """
    セルフ影キーフレ

    Parameters
    ----------
    index : int, optional
        キーフレ, by default None
    mode : int, optional
        セルフ影モード, by default None
    distance : float, optional
        影範囲距離, by default None
    regist : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    def __init__(
        self,
        index: int = None,
        mode: int = None,
        distance: float = None,
        regist: bool = None,
        read: bool = None,
    ):
        super().__init__(index, regist, read)
        self.type: int = mode or 0
        self.distance: float = distance or 0.0


class VmdIkOnoff(BaseModel):
    """
    IKのONOFF

    Parameters
    ----------
    name : str, optional
        IK名, by default None
    onoff : bool, optional
        ONOFF, by default None
    """

    def __init__(
        self,
        name: str = None,
        onoff: bool = None,
    ):
        super().__init__()
        self.name: str = name or ""
        self.onoff: bool = onoff or True


class VmdShowIkFrame(BaseVmdFrame):
    """
    IKキーフレ

    Parameters
    ----------
    index : int, optional
        キーフレ, by default None
    show : bool, optional
        表示有無, by default None
    iks : list[VmdIk], optional
        IKリスト, by default None
    regist : bool, optional
        登録対象か否か, by default None
    read : bool, optional
        VMDデータから読み込んだデータか, by default None
    """

    def __init__(
        self,
        index: int = None,
        show: bool = None,
        iks: list[VmdIkOnoff] = None,
        regist: bool = None,
        read: bool = None,
    ):
        super().__init__(index, regist, read)
        self.show: bool = show or True
        self.iks: list[VmdIkOnoff] = iks or []
