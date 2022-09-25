import numpy as np
from scipy.signal import savgol_filter

import bezier
from base.base import BaseModel
from base.math import MVector2D

# MMDでの補間曲線の最大値
IP_MAX = 127


class Interpolation(BaseModel):
    def __init__(
        self,
        begin: MVector2D = None,
        start: MVector2D = None,
        end: MVector2D = None,
        finish: MVector2D = None,
    ):
        """
        補間曲線

        Parameters
        ----------
        start : MVector2D, optional
            補間曲線開始, by default None
        end : MVector2D, optional
            補間曲線終了, by default None
        """
        self.begin: MVector2D = begin or MVector2D(0, 0)
        self.start: MVector2D = start or MVector2D(20, 20)
        self.end: MVector2D = end or MVector2D(107, 107)
        self.finish: MVector2D = finish or MVector2D(IP_MAX, IP_MAX)

    def normalize(self):
        diff = self.finish - self.begin
        b = self.begin.copy()
        self.begin = Interpolation.round_mmd((self.begin - b) / diff, MVector2D())
        self.start = Interpolation.round_mmd((self.start - b) / diff, MVector2D())
        self.end = Interpolation.round_mmd(
            (self.end - b) / diff, MVector2D(IP_MAX, IP_MAX)
        )
        self.finish = Interpolation.round_mmd(
            (self.finish - b) / diff, MVector2D(IP_MAX, IP_MAX)
        )

    @classmethod
    def round_mmd(cls, t: MVector2D, s: MVector2D) -> MVector2D:
        t.x = Interpolation.round(t.x * IP_MAX)
        t.y = Interpolation.round(t.y * IP_MAX)

        if not (0 <= t.x and 0 <= t.y):
            # 範囲に収まってない場合、縮める
            v = (t - (t * ((s - t) / np.max((s - t).vector)))) * 0.95
            t.x = Interpolation.round(v.x)
            t.y = Interpolation.round(v.y)

        elif not (t.x <= IP_MAX and t.y <= IP_MAX):
            # 範囲に収まってない場合、縮める
            v = (t * IP_MAX / np.max(t.vector)) * 0.95
            t.x = Interpolation.round(v.x)
            t.y = Interpolation.round(v.y)

        return t

    @classmethod
    def round(cls, t: float) -> int:
        t2 = t * 1000000
        # pythonは偶数丸めなので、整数部で丸めた後、元に戻す
        return int(round(round(t2, -6) / 1000000))


def get_infections(values: list[float], threshold, decimals) -> list[int]:
    extract_idxs = np.where(np.abs(np.round(np.diff(values), decimals)) > threshold)[0]
    if len(extract_idxs) <= 1:
        return []

    extracts = np.array(values)[extract_idxs]
    f_prime = np.gradient(extracts)
    infections = extract_idxs[np.where(np.diff(np.sign(f_prime)))[0]]

    return infections


def create_interpolation(values: list[float]):
    if len(values) <= 2 or abs(np.max(values) - np.min(values)) < 0.0001:
        return Interpolation()

    # Xは次数（フレーム数）分移動
    xs = np.arange(0, len(values))

    # YはXの移動分を許容範囲とする
    ys = np.array(values)

    # https://github.com/dhermes/bezier/issues/242
    s_vals = np.linspace(0, 1, len(values))
    representative = bezier.Curve.from_nodes(np.eye(4))
    transform = representative.evaluate_multi(s_vals).T
    nodes = np.vstack([xs, ys])
    reduced_t, residuals, rank, _ = np.linalg.lstsq(transform, nodes.T, rcond=None)
    reduced = reduced_t.T
    joined_curve = bezier.Curve.from_nodes(reduced)

    nodes = joined_curve.nodes

    # 次数を減らしたベジェ曲線をMMD用補間曲線に変換
    org_ip = Interpolation(
        begin=MVector2D(nodes[0, 0], nodes[1, 0]),
        start=MVector2D(nodes[0, 1], nodes[1, 1]),
        end=MVector2D(nodes[0, 2], nodes[1, 2]),
        finish=MVector2D(nodes[0, 3], nodes[1, 3]),
    )
    org_ip.normalize()

    return org_ip


# http://d.hatena.ne.jp/edvakf/20111016/1318716097
# https://pomax.github.io/bezierinfo
# https://shspage.hatenadiary.org/entry/20140625/1403702735
# https://bezier.readthedocs.io/en/stable/python/reference/bezier.curve.html#bezier.curve.Curve.evaluate
def evaluate(
    interpolation: Interpolation, start: int, now: int, end: int
) -> tuple[float, float, float]:
    """
    補間曲線を求める

    Parameters
    ----------
    interpolation : Interpolation
        補間曲線
    start : int
        開始キーフレ
    now : int
        計算キーフレ
    end : int
        終端キーフレ

    Returns
    -------
    tuple[float, float, float]
        x（計算キーフレ時点のX値）, y（計算キーフレ時点のY値）, t（計算キーフレまでの変化量）
    """
    if (now - start) == 0 or (end - start) == 0:
        return 0, 0, 0

    x = (now - start) / (end - start)
    x1 = interpolation.start.x / IP_MAX
    y1 = interpolation.start.y / IP_MAX
    x2 = interpolation.end.x / IP_MAX
    y2 = interpolation.end.y / IP_MAX

    t = 0.5
    s = 0.5

    # 二分法
    for i in range(15):
        ft = (3 * (s * s) * t * x1) + (3 * s * (t * t) * x2) + (t * t * t) - x

        if ft > 0:
            t -= 1 / (4 << i)
        else:
            t += 1 / (4 << i)

        s = 1 - t

    y = (3 * (s * s) * t * y1) + (3 * s * (t * t) * y2) + (t * t * t)

    return x, y, t
