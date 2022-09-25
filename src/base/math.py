import operator
from math import acos, cos, degrees, radians, sin, sqrt  # type: ignore
from typing import Any, Union

import numpy as np
from quaternion import from_rotation_matrix, quaternion

from base.base import BaseModel


class MRect(BaseModel):
    """
    矩形クラス

    Parameters
    ----------
    x : int
        x座標
    y : int
        y座標
    width : int
        横幅
    height : int
        縦幅
    """

    def __init__(self, x=0, y=0, width=0, height=0):
        self.x = int(x)
        self.y = int(y)
        self.width = int(width)
        self.height = int(height)

    @property
    def x(self) -> int:
        return int(self.x)

    @x.setter
    def x(self, v: int):
        self.x = int(v)

    @property
    def y(self) -> int:
        return int(self.y)

    @y.setter
    def y(self, v: int):
        self.y = int(v)

    @property
    def width(self) -> int:
        return int(self.width)

    @width.setter
    def width(self, v: int):
        self.width = int(v)

    @property
    def height(self) -> int:
        return int(self.height)

    @height.setter
    def height(self, v: int):
        self.height = int(v)


class MVector(BaseModel):
    """ベクトル基底クラス"""

    __slots__ = ["vector"]

    def __init__(self, x):
        """初期化

        Parameters
        ----------
        x : float, MVector, np.ndarray
        """
        if isinstance(x, int) or isinstance(x, float):
            # 実数の場合
            self.vector = np.array([x], dtype=np.float64)
        elif isinstance(x, np.ndarray):
            # numpy-arrayの場合
            self.vector = np.copy(x)
        else:
            self.vector = x.copy()

    def copy(self):
        return self.__class__(np.copy(self.vector))

    def length(self) -> float:
        """
        ベクトルの長さ
        """
        return float(np.linalg.norm(self.vector, ord=2))

    def length_squared(self) -> float:
        """
        ベクトルの長さの二乗
        """
        return float(np.linalg.norm(self.vector, ord=2) ** 2)

    def effective(self):
        self.vector[np.isinf(self.vector)] = 0
        self.vector[np.isnan(self.vector)] = 0

    def round(self, decimals: int):
        """
        丸め処理

        Parameters
        ----------
        decimals : int
            丸め桁数

        Returns
        -------
        MVector
        """
        return self.__class__(np.round(self.vector, decimals=decimals))

    def normalized(self):
        """
        正規化した値を返す
        """
        self.effective()
        l2 = np.linalg.norm(self.vector, ord=2, axis=-1, keepdims=True)
        l2[l2 == 0] = 1
        normv = self.vector / l2
        return self.__class__(normv)

    def normalize(self):
        """
        自分自身の正規化
        """
        self.effective()
        l2 = np.linalg.norm(self.vector, ord=2, axis=-1, keepdims=True)
        l2[l2 == 0] = 1
        self.vector /= l2

    def distance(self, other) -> float:
        """
        他のベクトルとの距離

        Parameters
        ----------
        other : MVector
            他のベクトル

        Returns
        -------
        float
        """
        if not isinstance(other, self.__class__):
            raise ValueError("同じ型同士で計算してください")
        return self.__class__(self.vector - other.vector).length()

    def abs(self):
        """
        絶対値変換
        """
        return self.__class__(np.abs(self.vector))

    def cross(self, other):
        """
        外積
        """
        return self.__class__(np.cross(self.vector, other.vector))

    def inner(self, other) -> float:
        """
        内積（一次元配列）
        """
        return float(np.inner(self.vector, other.vector))

    def dot(self, other) -> float:
        """
        内積（二次元の場合、二次元のまま返す）
        """
        return float(np.dot(self.vector, other.vector))

    def __lt__(self, other) -> bool:
        if isinstance(other, MVector):
            return bool(np.all(np.less(self.vector, other.vector)))
        else:
            return bool(np.all(np.less(self.vector, other)))

    def __le__(self, other) -> bool:
        if isinstance(other, MVector):
            return bool(np.all(np.less_equal(self.vector, other.vector)))
        else:
            return bool(np.all(np.less_equal(self.vector, other)))

    def __eq__(self, other) -> bool:
        if isinstance(other, MVector):
            return bool(np.all(np.equal(self.vector, other.vector)))
        else:
            return bool(np.all(np.equal(self.vector, other)))

    def __ne__(self, other) -> bool:
        if isinstance(other, MVector):
            return bool(np.all(np.not_equal(self.vector, other.vector)))
        else:
            return bool(np.all(np.not_equal(self.vector, other)))

    def __gt__(self, other) -> bool:
        if isinstance(other, MVector):
            return bool(np.all(np.greater(self.vector, other.vector)))
        else:
            return bool(np.all(np.greater(self.vector, other)))

    def __ge__(self, other) -> bool:
        if isinstance(other, MVector):
            return bool(np.all(np.greater_equal(self.vector, other.vector)))
        else:
            return bool(np.all(np.greater_equal(self.vector, other)))

    def __bool__(self) -> bool:
        return not bool(np.all(self.vector == 0))

    def __add__(self, other):
        return operate_vector(self, other, operator.add)

    def __sub__(self, other):
        return operate_vector(self, other, operator.sub)

    def __mul__(self, other):
        return operate_vector(self, other, operator.mul)

    def __truediv__(self, other):
        return operate_vector(self, other, operator.truediv)

    def __floordiv__(self, other):
        return operate_vector(self, other, operator.floordiv)

    def __mod__(self, other):
        return operate_vector(self, other, operator.mod)

    def __iadd__(self, other):
        self.vector = operate_vector(self, other, operator.add).vector
        return self

    def __isub__(self, other):
        self.vector = operate_vector(self, other, operator.sub).vector
        return self

    def __imul__(self, other):
        self.vector = operate_vector(self, other, operator.mul).vector
        return self

    def __itruediv__(self, other):
        self.vector = operate_vector(self, other, operator.truediv).vector
        return self

    def __ifloordiv__(self, other):
        self.vector = operate_vector(self, other, operator.floordiv).vector
        return self

    def __imod__(self, other):
        self.vector = operate_vector(self, other, operator.mod).vector
        return self

    def __lshift__(self, other):
        return operate_vector(self, other, operator.lshift)

    def __rshift__(self, other):
        return operate_vector(self, other, operator.rshift)

    def __and__(self, other):
        return operate_vector(self, other, operator.and_)

    def __or__(self, other):
        return operate_vector(self, other, operator.or_)

    def __neg__(self):
        return self.__class__(operator.neg(self.vector))

    def __pos__(self):
        return self.__class__(operator.pos(self.vector))

    def __invert__(self):
        return self.__class__(operator.invert(self.vector))

    @property
    def x(self):
        return self.vector[0]

    @x.setter
    def x(self, v):
        self.vector[0] = v


class MVector2D(MVector):
    """
    2次元ベクトルクラス
    """

    def __init__(self, x=0.0, y=0.0):
        """
        初期化

        Parameters
        ----------
        x : float, optional
            by default 0.0
        y : float, optional
            by default 0.0
        """
        if isinstance(x, int) or isinstance(x, float):
            # 実数の場合
            self.vector = np.array([x, y], dtype=np.float64)
        elif isinstance(x, np.ndarray):
            # numpy-arrayの場合
            self.vector = np.copy(x)
        else:
            self.vector = x.copy()

    def to_log(self) -> str:
        return f"[x={round(self.vector[0], 3)}, y={round(self.vector[1], 3)}]"

    @property
    def y(self):
        return self.vector[1]

    @y.setter
    def y(self, v):
        self.vector[1] = v


class MVector3D(MVector):
    """
    3次元ベクトルクラス
    """

    def __init__(self, x=0.0, y=0.0, z=0.0):
        """
        初期化

        Parameters
        ----------
        x : float, optional
            by default 0.0
        y : float, optional
            by default 0.0
        z : float, optional
            by default 0.0
        """
        if isinstance(x, int) or isinstance(x, float):
            # 実数の場合
            self.vector = np.array([x, y, z], dtype=np.float64)
        elif isinstance(x, np.ndarray):
            # numpy-arrayの場合
            self.vector = np.copy(x)
        else:
            self.vector = x.copy()

    def to_log(self) -> str:
        """
        ログ用文字列に変換
        """
        return f"[x={round(self.vector[0], 3)}, y={round(self.vector[1], 3)}, z={round(self.vector[2], 3)}]"

    def to_key(self, threshold=0.1) -> tuple:
        """
        キー用値に変換

        Parameters
        ----------
        threshold : float, optional
            閾値, by default 0.1

        Returns
        -------
        tuple
            (x, y, z)
        """
        return (
            round(self.vector[0] / threshold),
            round(self.vector[1] / threshold),
            round(self.vector[2] / threshold),
        )

    @property
    def y(self):
        return self.vector[1]

    @y.setter
    def y(self, v):
        self.vector[1] = v

    @property
    def z(self):
        return self.vector[2]

    @z.setter
    def z(self, v):
        self.vector[2] = v


class MVector4D(MVector):
    """
    4次元ベクトルクラス
    """

    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
        """
        初期化

        Parameters
        ----------
        x : float, optional
            by default 0.0
        y : float, optional
            by default 0.0
        z : float, optional
            by default 0.0
        """
        if isinstance(x, int) or isinstance(x, float):
            # 実数の場合
            self.vector = np.array([x, y, z, w], dtype=np.float64)
        elif isinstance(x, np.ndarray):
            # numpy-arrayの場合
            self.vector = np.copy(x)
        else:
            self.vector = x.copy()

    def to_log(self) -> str:
        return (
            f"[x={round(self.vector[0], 3)}, y={round(self.vector[1], 3)}, "
            + f"z={round(self.vector[2], 3)}], w={round(self.vector[2], 3)}]"
        )

    @property
    def y(self):
        return self.vector[1]

    @y.setter
    def y(self, v):
        self.vector[1] = v

    @property
    def z(self):
        return self.vector[2]

    @z.setter
    def z(self, v):
        self.vector[2] = v

    @property
    def w(self):
        return self.vector[3]

    @w.setter
    def w(self, v):
        self.vector[3] = v


class MVectorDict:
    """ベクトル辞書基底クラス"""

    __slots__ = ["vectors"]

    def __init__(self):
        """初期化"""
        self.vectors = {}

    def __iter__(self):
        return self.vectors.items()

    def keys(self) -> list:
        return list(self.vectors.keys())

    def values(self) -> list:
        return list(self.vectors.values())

    def append(self, vkey: Any, v: MVector) -> None:
        self.vectors[vkey] = v.vector

    def distances(self, v: MVector):
        return np.linalg.norm(
            (np.array(list(self.vectors.values())) - v.vector), ord=2, axis=1
        )

    def nearest_distance(self, v: MVector) -> float:
        """
        指定ベクトル直近値

        Parameters
        ----------
        v : MVector
            比較対象ベクトル

        Returns
        -------
        float
            直近距離
        """
        return float(np.min(self.distances(v)))

    def nearest_value(self, v: MVector) -> MVector:
        """
        指定ベクトル直近値

        Parameters
        ----------
        v : MVector
            比較対象ベクトル

        Returns
        -------
        float
            直近値
        """
        return v.__class__(np.array(self.values())[np.argmin(self.distances(v))])

    def nearest_key(self, v: MVector) -> Any:
        """
        指定ベクトル直近キー

        Parameters
        ----------
        v : MVector
            比較対象ベクトル

        Returns
        -------
        Any
            直近キー
        """
        return np.array(self.keys())[np.argmin(self.distances(v))]


class MQuaternion(MVector):
    """
    クォータニオンクラス
    """

    def __init__(self, scalar=0, x=0, y=0, z=0):
        if isinstance(scalar, int) or isinstance(scalar, float):
            # 実数の場合
            if np.isclose([scalar, x, y, z], 0).all():
                self.vector = quaternion(1, 0, 0, 0)
            else:
                self.vector = quaternion(scalar, x, y, z)
        elif isinstance(scalar, quaternion):
            # quaternionの場合
            self.vector = quaternion(*scalar.components)
        else:
            # numpy-array, listの場合
            self.vector = quaternion(*scalar)

    @property
    def scalar(self) -> float:
        return self.vector.components[0]

    @scalar.setter
    def scalar(self, v):
        self.vector.components[0] = v

    @property
    def x(self) -> float:
        return self.vector.components[1]

    @x.setter
    def x(self, v):
        self.vector.components[1] = v

    @property
    def y(self) -> float:
        return self.vector.components[2]

    @y.setter
    def y(self, v):
        self.vector.components[2] = v

    @property
    def z(self) -> float:
        return self.vector.components[3]

    @z.setter
    def z(self, v):
        self.vector.components[3] = v

    @property
    def xyz(self) -> MVector3D:
        return MVector3D(self.vector.components[1:])

    def effective(self):
        self.vector.components[np.isnan(self.vector.components)] = 0
        self.vector.components[np.isinf(self.vector.components)] = 0

    def length(self) -> float:
        """
        ベクトルの長さ
        """
        return float(self.vector.abs())

    def length_squared(self) -> float:
        """
        ベクトルの長さの二乗
        """
        return float(self.vector.abs() ** 2)

    def inverse(self):
        """
        逆回転
        """
        return MQuaternion(self.vector.inverse())

    def normalized(self):
        """
        正規化した値を返す
        """
        self.effective()
        l2 = np.linalg.norm(self.vector.components, ord=2, axis=-1, keepdims=True)
        l2[l2 == 0] = 1
        normv = self.vector.components / l2
        return self.__class__(normv)

    def normalize(self):
        """
        自分自身の正規化
        """
        self.effective()
        l2 = np.linalg.norm(self.vector.components, ord=2, axis=-1, keepdims=True)
        l2[l2 == 0] = 1
        self.vector.components /= l2

    def to_vector4(self) -> MVector4D:
        return MVector4D(self.x, self.y, self.z, self.scalar)

    def copy(self):
        return MQuaternion(np.copy(self.vector.components))

    def dot(self, v):
        return np.sum(self.vector.components * v.vector.components)

    def to_euler_degrees(self) -> MVector3D:
        """
        クォータニオンをオイラー角に変換する
        """
        xx = self.x * self.x
        xy = self.x * self.y
        xz = self.x * self.z
        xw = self.x * self.scalar
        yy = self.y * self.y
        yz = self.y * self.z
        yw = self.y * self.scalar
        zz = self.z * self.z
        zw = self.z * self.scalar
        lengthSquared = xx + yy + zz + self.scalar**2

        if not np.isclose([lengthSquared, lengthSquared - 1.0], 0).any():
            xx, xy, xz, xw, yy, yz, yw, zz, zw = (
                np.array([xx, xy, xz, xw, yy, yz, yw, zz, zw]) / lengthSquared
            )

        pitch = np.arcsin(max(-1, min(1, -2.0 * (yz - xw))))
        yaw = 0
        roll = 0

        if pitch < (np.pi / 2):
            if pitch > -(np.pi / 2):
                yaw = np.arctan2(2.0 * (xz + yw), 1.0 - 2.0 * (xx + yy))
                roll = np.arctan2(2.0 * (xy + zw), 1.0 - 2.0 * (xx + zz))
            else:
                # not a unique solution
                roll = 0
                yaw = -np.arctan2(-2.0 * (xy - zw), 1.0 - 2.0 * (yy + zz))
        else:
            # not a unique solution
            roll = 0
            yaw = np.arctan2(-2.0 * (xy - zw), 1.0 - 2.0 * (yy + zz))

        return MVector3D(np.degrees([pitch, yaw, roll]))

    def to_euler_degrees_mmd(self) -> MVector3D:
        """
        MMDの表記に合わせたオイラー角
        """
        euler = self.to_euler_degrees()
        return MVector3D(euler.x, -euler.y, -euler.z)

    def to_degrees(self) -> float:
        """
        角度に変換
        """
        return degrees(2 * acos(min(1, max(-1, self.scalar))))

    def to_signed_degrees(self, local_axis: MVector3D) -> float:
        """
        軸による符号付き角度に変換
        """
        deg = degrees(2 * acos(min(1, max(-1, self.scalar))))
        sign = np.sign(self.xyz.dot(local_axis)) * np.sign(self.scalar)

        if sign != 0:
            deg *= sign

        if abs(deg) > 180:
            # 180度を超してる場合、フリップなので、除去
            return (abs(deg) - 180) * np.sign(deg)

        return deg

    def to_theta(self, v):
        """
        自分ともうひとつの値vとのtheta（変位量）を返す
        """
        return acos(min(1, max(-1, self.normalized().dot(v.normalized()))))

    def to_matrix4x4(self):
        # q(w,x,y,z)から(x,y,z,w)に並べ替え.
        q2 = np.array([self.x, self.y, self.z, self.scalar], dtype=np.float64)

        mat = MMatrix4x4(identity=True)
        mat[0, 0] = q2[3] * q2[3] + q2[0] * q2[0] - q2[1] * q2[1] - q2[2] * q2[2]
        mat[0, 1] = 2.0 * q2[0] * q2[1] - 2.0 * q2[3] * q2[2]
        mat[0, 2] = 2.0 * q2[0] * q2[2] + 2.0 * q2[3] * q2[1]
        mat[0, 3] = 0.0

        mat[1, 0] = 2.0 * q2[0] * q2[1] + 2.0 * q2[3] * q2[2]
        mat[1, 1] = q2[3] * q2[3] - q2[0] * q2[0] + q2[1] * q2[1] - q2[2] * q2[2]
        mat[1, 2] = 2.0 * q2[1] * q2[2] - 2.0 * q2[3] * q2[0]
        mat[1, 3] = 0.0

        mat[2, 0] = 2.0 * q2[0] * q2[2] - 2.0 * q2[3] * q2[1]
        mat[2, 1] = 2.0 * q2[1] * q2[2] + 2.0 * q2[3] * q2[0]
        mat[2, 2] = q2[3] * q2[3] - q2[0] * q2[0] - q2[1] * q2[1] + q2[2] * q2[2]
        mat[2, 3] = 0.0

        mat[3, 0] = 0.0
        mat[3, 1] = 0.0
        mat[3, 2] = 0.0
        mat[3, 3] = q2[3] * q2[3] + q2[0] * q2[0] + q2[1] * q2[1] + q2[2] * q2[2]

        mat /= mat[3, 3]
        mat[3, 3] = 1.0

        return mat

    def __mul__(self, other):
        if isinstance(other, MVector3D):
            # quaternion と vec3 のかけ算は vec3 を返す
            return self.to_matrix4x4() * other
        return super().__mul__(other)

    @classmethod
    def from_euler_degrees(cls, a: Union[int, float, MVector3D], b=0, c=0):
        """
        オイラー角をクォータニオンに変換する
        """
        euler = np.zeros(3)
        if isinstance(a, int) or isinstance(a, float):
            euler = np.radians([a, b, c], dtype=np.double)
        else:
            euler = np.radians([a.x, a.y, a.z], dtype=np.double)

        euler *= 0.5

        c1, c2, c3 = np.cos([euler[1], euler[2], euler[0]])
        s1, s2, s3 = np.sin([euler[1], euler[2], euler[0]])
        w = c1 * c2 * c3 + s1 * s2 * s3
        x = c1 * c2 * s3 + s1 * s2 * c3
        y = s1 * c2 * c3 - c1 * s2 * s3
        z = c1 * s2 * c3 - s1 * c2 * s3

        return MQuaternion(w, x, y, z)

    @classmethod
    def from_axis_angles(cls, v: MVector3D, a: float):
        """
        軸と角度からクォータニオンに変換する
        """
        vv = v.normalized()
        length = sqrt(vv.x**2 + vv.y**2 + vv.z**2)

        xyz = vv.vector
        if not np.isclose([length - 1.0, length], 0).any():
            xyz /= length

        a = radians(a / 2.0)
        return MQuaternion(cos(a), *(xyz * sin(a))).normalized()

    @classmethod
    def from_direction(cls, direction: MVector3D, up: MVector3D):
        """
        軸と角度からクォータニオンに変換する
        """
        if np.isclose(direction.vector, 0).all():
            return MQuaternion()

        z_axis = direction.normalized()
        x_axis = up.cross(z_axis).normalized()

        if np.isclose(x_axis.length_squared(), 0).all():
            # collinear or invalid up vector derive shortest arc to new direction
            return MQuaternion.rotate(MVector3D(0.0, 0.0, 1.0), z_axis)

        y_axis = z_axis.cross(x_axis)

        return MQuaternion.from_axes(x_axis, y_axis, z_axis)

    @classmethod
    def rotate(cls, from_v: MVector3D, to_v: MVector3D):
        """
        fromベクトルからtoベクトルまでの回転量
        """
        v0 = from_v.normalized()
        v1 = to_v.normalized()
        d = v0.dot(v1) + 1.0

        # if dest vector is close to the inverse of source vector, ANY axis of rotation is valid
        if np.isclose(d, 0).all():
            axis = MVector3D(1.0, 0.0, 0.0).cross(v0)
            if np.isclose(axis.lengthSquared(), 0).all():
                axis = MVector3D(0.0, 1.0, 0.0).cross(v0)
            axis.normalize()
            # same as MQuaternion.fromAxisAndAngle(axis, 180.0)
            return MQuaternion(0.0, axis.x, axis.y, axis.z).normalized()

        d = sqrt(2.0 * d)
        axis = v0.cross(v1) / d
        return MQuaternion(d * 0.5, axis.x, axis.y, axis.z).normalized()

    @classmethod
    def from_axes(cls, x_axis: MVector3D, y_axis: MVector3D, z_axis: MVector3D):
        return MQuaternion(
            from_rotation_matrix(
                np.array(
                    [
                        [x_axis.x, y_axis.x, z_axis.x],
                        [x_axis.y, y_axis.y, z_axis.y],
                        [x_axis.z, y_axis.z, z_axis.z],
                    ],
                    dtype=np.float64,
                )
            )
        )

    @classmethod
    def nlerp(cls, q1, q2, t):
        """
        線形補間
        """
        # Handle the easy cases first.
        if t <= 0.0:
            return q1
        elif t >= 1.0:
            return q2

        q2b = q2.copy()
        d = q1.dot(q2)

        if d < 0.0:
            q2b = -q2b

        return (q1 * (1.0 - t) + q2b * t).normalized()

    @classmethod
    def slerp(cls, q1, q2, t):
        """
        球形補間
        """
        # Handle the easy cases first.
        if t <= 0.0:
            return q1
        elif t >= 1.0:
            return q2

        # Determine the angle between the two quaternions.
        q2b = q2.copy()
        d = q1.dot(q2)

        if d < 0.0:
            q2b = -q2b
            d = -d

        # Get the scale factors.  If they are too small,
        # then revert to simple linear interpolation.
        factor1 = 1.0 - t
        factor2 = t

        if not np.isclose(1.0 - d, 0):
            angle = acos(max(0, min(1, d)))
            sinOfAngle = sin(angle)
            if not np.isclose(sinOfAngle, 0):
                factor1 = sin((1.0 - t) * angle) / sinOfAngle
                factor2 = sin(t * angle) / sinOfAngle

        # Construct the result quaternion.
        return q1 * factor1 + q2b * factor2


class MMatrix4x4(MVector):
    """
    4x4行列クラス
    """

    def __init__(
        self,
        m11=1.0,
        m12=0.0,
        m13=0.0,
        m14=0.0,
        m21=0.0,
        m22=1.0,
        m23=0.0,
        m24=0.0,
        m31=0.0,
        m32=0.0,
        m33=1.0,
        m34=0.0,
        m41=0.0,
        m42=0.0,
        m43=0.0,
        m44=1.0,
        identity=False,
    ):
        if isinstance(m11, int) or isinstance(m11, float):
            # 実数の場合
            self.vector = np.array(
                [
                    [m11, m12, m13, m14],
                    [m21, m22, m23, m24],
                    [m31, m32, m33, m34],
                    [m41, m42, m43, m44],
                ],
                dtype=np.float64,
            )
        elif isinstance(m11, MMatrix4x4):
            # quaternionの場合
            self.vector = np.array(
                [
                    [
                        m11.vector[0, 0],
                        m11.vector[0, 1],
                        m11.vector[0, 2],
                        m11.vector[0, 3],
                    ],
                    [
                        m11.vector[1, 0],
                        m11.vector[1, 1],
                        m11.vector[1, 2],
                        m11.vector[1, 3],
                    ],
                    [
                        m11.vector[2, 0],
                        m11.vector[2, 1],
                        m11.vector[2, 2],
                        m11.vector[2, 3],
                    ],
                    [
                        m11.vector[3, 0],
                        m11.vector[3, 1],
                        m11.vector[3, 2],
                        m11.vector[3, 3],
                    ],
                ],
                dtype=np.float64,
            )
        elif isinstance(m11, np.ndarray):
            # 行列そのものの場合
            self.vector = np.array(
                [
                    [m11[0, 0], m11[0, 1], m11[0, 2], m11[0, 3]],
                    [m11[1, 0], m11[1, 1], m11[1, 2], m11[1, 3]],
                    [m11[2, 0], m11[2, 1], m11[2, 2], m11[2, 3]],
                    [m11[3, 0], m11[3, 1], m11[3, 2], m11[3, 3]],
                ],
                dtype=np.float64,
            )
        else:
            # listの場合
            self.vector = MMatrix4x4(*m11)
        if identity:
            # フラグが立ってたら初期化を実行する
            self.identity()

    def inverse(self):
        """
        逆行列
        """
        return MMatrix4x4(np.linalg.inv(self.vector))

    def rotate(self, q: MQuaternion):
        """
        回転行列
        """
        self.vector = self.vector.dot(q.to_matrix4x4().vector)

    def translate(self, v: MVector3D):
        """
        平行移動行列
        """
        vmat = self.vector[:, :3] * v.vector
        self.vector[:, 3] += np.sum(vmat, axis=1)

    def scale(self, v: MVector3D):
        """
        縮尺行列
        """
        self.vector[:, :3] *= v.vector

    def identity(self):
        """
        初期化
        """
        self.vector = np.eye(4, dtype=np.float64)

    def look_at(self, eye: MVector3D, center: MVector3D, up: MVector3D):
        forward = center - eye
        forward.normalize()
        if np.isclose(forward, 0).all():
            return

        side = forward.cross(up).normalized()
        upv = side.cross(forward).normalized()

        m = MMatrix4x4()
        m.vector[0, :-1] = side.vector
        m.vector[1, :-1] = upv.vector
        m.vector[2, :-1] = -forward.vector
        m.vector[-1, -1] = 1.0

        self *= m
        self.translate(-eye)

    def perspective(
        self,
        vertical_angle: float,
        aspect_ratio: float,
        near_plane: float,
        far_plane: float,
    ):
        """
        パースペクティブ行列
        """
        if near_plane == far_plane or aspect_ratio == 0:
            return

        rad = radians(vertical_angle / 2)
        sine = sin(rad)

        if sine == 0:
            return

        cotan = cos(rad) / sine
        clip = far_plane - near_plane

        m = MMatrix4x4()
        m.vector[0, 0] = cotan / aspect_ratio
        m.vector[1, 1] = cotan
        m.vector[2, 2] = -(near_plane + far_plane) / clip
        m.vector[2, 3] = -(2 * near_plane * far_plane) / clip
        m.vector[3, 2] = -1

        self *= m

    def map_vector(self, v: MVector3D) -> MVector3D:
        return MVector3D(*np.sum(v.vector * self.vector[:3, :3], axis=1))

    def to_quternion(self):
        q = MQuaternion()
        v = self.vector

        # I removed + 1
        trace = v[0, 0] + v[1, 1] + v[2, 2]
        # I changed M_EPSILON to 0
        if trace > 0:
            s = 0.5 / sqrt(trace + 1)
            q.scalar = 0.25 / s
            q.x = (v[2, 1] - v[1, 2]) * s
            q.y = (v[0, 2] - v[2, 0]) * s
            q.z = (v[1, 0] - v[0, 1]) * s
        else:
            if v[0, 0] > v[1, 1] and v[0, 0] > v[2, 2]:
                s = 2 * sqrt(1 + v[0, 0] - v[1, 1] - v[2, 2])
                q.scalar = (v[2, 1] - v[1, 2]) / s
                q.x = 0.25 * s
                q.y = (v[0, 1] + v[1, 0]) / s
                q.z = (v[0, 2] + v[2, 0]) / s
            elif v[1, 1] > v[2, 2]:
                s = 2 * sqrt(1 + v[1, 1] - v[0, 0] - v[2, 2])
                q.scalar = (v[0, 2] - v[2, 0]) / s
                q.x = (v[0, 1] + v[1, 0]) / s
                q.y = 0.25 * s
                q.z = (v[1, 2] + v[2, 1]) / s
            else:
                s = 2 * sqrt(1 + v[2, 2] - v[0, 0] - v[1, 1])
                q.scalar = (v[1, 0] - v[0, 1]) / s
                q.x = (v[0, 2] + v[2, 0]) / s
                q.y = (v[1, 2] + v[2, 1]) / s
                q.z = 0.25 * s

        return q

    def to_position(self) -> MVector3D:
        return MVector3D(self.vector[3, 0:3])

    def __mul__(self, other):
        if isinstance(other, MMatrix4x4):
            # 行列同士のかけ算
            return MMatrix4x4(np.matmul(self.vector, other.vector))
        elif isinstance(other, MVector3D):
            # vec3 とのかけ算は vec3 を返す
            s = np.sum(self.vector[:, :3] * other.vector, axis=1) + self.vector[:, 3]
            if s[3] == 1.0:
                return MVector3D(s[:3])
            elif s[3] == 0.0:
                return MVector3D()
            else:
                return MVector3D(s[:3] / s[3])
        elif isinstance(other, MVector4D):
            # vec4 とのかけ算は vec4 を返す
            return MVector4D(np.sum(self.vector * other.vector, axis=1))
        return super().__mul__(other)

    def __imul__(self, other):
        if isinstance(other, MMatrix4x4):
            self.vector = np.matmul(self.vector, other.vector)
        else:
            raise ValueError("MMatrix4x4同士で計算してください")

    def __getitem__(self, index) -> float:
        y, x = index
        return self.vector[y, x]

    def __setitem__(self, index, v: float):
        y, x = index
        self.vector[y, x] = v


class MMatrix4x4List(MVector):
    """
    4x4行列クラスリスト
    """

    def __init__(self, keys: dict[str, list[str]]):
        self.vector = {}
        self.vector = dict(
            [
                (k, [MMatrix4x4(identity=True) for _ in range(len(vs))])
                for k, vs in keys.items()
            ]
        )

    def __setitem__(self, key: Any, value: MMatrix4x4):
        row, col = key
        self.vector[row][col] = value

    def multiply(self) -> dict[str, MMatrix4x4]:
        """
        行列リストを一括で積算する

        Returns
        -------
        list[MMatrix4x4]
            結果行列リスト
        """
        results = dict([(k, MMatrix4x4(identity=True)) for k in self.vector.keys()])
        for key, rmats in self.vector.items():
            for mat in rmats:
                results[key] *= mat
        return results


def operate_vector(v: MVector, other: Union[MVector, float, int], op) -> MVector:
    """
    演算処理

    Parameters
    ----------
    v : MVector
        計算主対象
    other : Union[MVector, float, int]
        演算対象
    op : 演算処理

    Returns
    -------
    MVector
        演算結果
    """
    if isinstance(other, MVector):
        v1 = op(v.vector, other.vector)
    else:
        v1 = op(v.vector, other)
    v2 = v.__class__(v1)
    v2.effective()
    return v2
