from enum import Enum, unique

import numpy as np

from base.base import BaseModel
from base.math import MQuaternion, MVector3D


@unique
class Switch(Enum):
    """ONOFFスイッチ"""

    OFF = 0
    ON = 1


class BaseRotationModel(BaseModel):
    def __init__(self, v_radians: MVector3D = MVector3D()) -> None:
        super().__init__()
        self.__radians = MVector3D()
        self.__degrees = MVector3D()
        self.__qq = MQuaternion()
        self.radians = v_radians

    @property
    def qq(self) -> MQuaternion:
        """
        回転情報をクォータニオンとして受け取る
        """
        return self.__qq

    @property
    def radians(self) -> MVector3D:
        """
        回転情報をラジアンとして受け取る
        """
        return self.__radians

    @radians.setter
    def radians(self, v: MVector3D):
        """
        ラジアンを回転情報として設定する

        Parameters
        ----------
        v : MVector3D
            ラジアン
        """
        self.__radians = v
        self.__degrees = MVector3D(*np.degrees(v.vector))
        self.__qq = MQuaternion.from_euler_degrees(self.degrees)

    @property
    def degrees(self) -> MVector3D:
        """
        回転情報を度として受け取る
        """
        return self.__degrees

    @degrees.setter
    def degrees(self, v: MVector3D):
        """
        度を回転情報として設定する

        Parameters
        ----------
        v : MVector3D
            度
        """
        self.__degrees = v
        self.__radians = MVector3D(*np.radians(v.vector))
        self.__qq = MQuaternion.from_euler_degrees(v)


class BaseIndexModel(BaseModel):
    """
    INDEXを持つ基底クラス
    """

    def __init__(self, index: int = -1) -> None:
        """
        初期化

        Parameters
        ----------
        index : int, optional
            INDEX, by default -1
        """
        super().__init__()
        self.index = index


class BaseIndexNameModel(BaseIndexModel):
    """
    INDEXと名前を持つ基底クラス
    """

    def __init__(self, index: int = -1, name: str = "", english_name: str = "") -> None:
        """
        初期化

        Parameters
        ----------
        index : int, optional
            INDEX, by default -1
        name : str, optional
            名前, by default ""
        english_name : str, optional
            英語名, by default ""
        """
        super().__init__()
        self.index = index
        self.name = name
        self.english_name = english_name
