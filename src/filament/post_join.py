"""近接成分の併合つき個体化（探索j）。`postprocess.instances` / `post_pos.instances_banded` の `join_gap` 追加版（新規ファイル）。

併合の定義: 確率 > thr の8連結成分のうち、互いの最短距離が join_gap 画素程度以下のものを1個体にまとめる。
実装は距離変換で、前景からの距離が join_gap/2 以下の画素（各成分を半径 join_gap/2 で膨張した領域）が8連結になる成分同士を同じ個体とする
（join_gap>0 のとき半径は最低1）。2成分の最短距離（ユークリッド）がおおむね join_gap 以下で併合され、8連結の判定のため境界は±1画素ほど揺れる。
併合は推移的（A-B、B-C が近ければ A,B,C が1個体）。個体の画素は元の成分の画素の和で、橋渡しの画素は足さない（画素は他の個体と重ならない）。
併合後の個体の面積・最大確率・重心は併合後の領域で再計算し、その値で min_area / min_peak / 帯の判定をする。join_gap=0 は既存の関数と完全に一致する。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .post_pos import Bands, Geometry, position_values
from .postprocess import EIGHT, binarize


def join_labels(lab: np.ndarray, join_gap: float, dist: np.ndarray | None = None) -> np.ndarray:
    """ラベル画像 lab（0=背景）の成分を、距離 join_gap 以下で併合したラベル画像（int32）。dist は lab==0 の距離変換（再利用用）。"""
    if join_gap < 0:
        raise ValueError("join_gap must be >= 0")
    if join_gap == 0 or lab.max() == 0:
        return lab
    if dist is None:
        dist = ndi.distance_transform_edt(lab == 0)
    l2, _ = ndi.label(dist <= max(join_gap / 2.0, 1.0), structure=EIGHT)
    return np.where(lab > 0, l2, 0).astype(np.int32)


def _check(prob: np.ndarray) -> None:
    if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1 + 1e-4:
        raise ValueError("probability map has NaN/inf or is outside [0,1]")


def labeled_joined(prob: np.ndarray, thr: float, join_gap: float = 0.0, low: float | None = None, closing: int = 0) -> np.ndarray:
    """しきい値処理 → 8連結成分 → 併合したラベル画像。"""
    lab, _ = ndi.label(binarize(prob, thr, low, closing), structure=EIGHT)
    return join_labels(lab, join_gap)


def instances_joined(prob: np.ndarray, thr: float = 0.5, join_gap: float = 0.0, min_area: int = 30, min_peak: float = 0.0,
                     low: float | None = None, closing: int = 0) -> np.ndarray:
    """postprocess.instances と同じ引数・戻り値（bool[N,H,W]）に join_gap を加えたもの。併合は面積・peak の絞り込みの前。"""
    _check(prob)
    lab = labeled_joined(prob, thr, join_gap, low, closing)
    n = int(lab.max())
    if n == 0:
        return np.zeros((0, *prob.shape), bool)
    idx = np.arange(1, n + 1)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    peak = np.asarray(ndi.maximum(prob, lab, idx))
    keep = [i for i, a, p in zip(idx, area, peak) if a >= min_area and p >= min_peak]
    return np.stack([lab == i for i in keep]) if keep else np.zeros((0, *prob.shape), bool)


def instances_joined_banded(prob: np.ndarray, bands: Bands, params: list[dict], join_gap: float = 0.0,
                            geom: Geometry = Geometry()) -> np.ndarray:
    """post_pos.instances_banded と同じ形式（uint8[N,H,W]）に join_gap を加えたもの。params は帯ごとの {'thr','min_area','min_peak'}（thr は全帯共通）。
    重心の帯の判定は併合後の個体で行う。"""
    if len(params) != bands.n_bands:
        raise ValueError(f"params must have {bands.n_bands} entries")
    thrs = {p["thr"] for p in params}
    if len(thrs) != 1:
        raise ValueError("instances_joined_banded requires a common thr")
    _check(prob)
    lab = labeled_joined(prob, thrs.pop(), join_gap)
    n = int(lab.max())
    if n == 0:
        return np.zeros((0, *prob.shape), np.uint8)
    idx = np.arange(1, n + 1)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    peak = np.asarray(ndi.maximum(prob, lab, idx))
    cy, cx = (np.asarray(c) for c in zip(*ndi.center_of_mass(np.ones_like(lab), lab, idx)))
    band = bands.index(position_values(cy, cx, geom, bands.quantity))
    ma = np.array([p["min_area"] for p in params])[band]
    mp = np.array([p["min_peak"] for p in params])[band]
    keep = idx[(area >= ma) & (peak >= mp)]
    if len(keep) == 0:
        return np.zeros((0, *prob.shape), np.uint8)
    return np.stack([(lab == i).astype(np.uint8) for i in keep])
