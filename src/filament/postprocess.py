"""確率 -> 非重複な個体マスク。8連結成分が既定（画素重複が起きない）。

後処理の候補（exp023）: closing（細い途切れの橋渡し）、hysteresis（低いしきい値まで領域を広げるが、高いしきい値の種を含む成分のみ残す）、
min_peak（成分内の最大確率が低い成分を捨てる）、min_area（小領域除去）。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), int)


def _disk(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return (yy**2 + xx**2) <= r * r


def binarize(prob: np.ndarray, thr: float = 0.5, low: float | None = None, closing: int = 0) -> np.ndarray:
    """しきい値処理。low を与えると hysteresis（prob>low の連結成分のうち prob>thr を含むものだけ残す）。"""
    m = prob > thr
    if low is not None and low < thr:
        lab, n = ndi.label(prob > low, structure=EIGHT)
        if n:
            seeded = np.zeros(n + 1, bool)
            seeded[np.unique(lab[m])] = True
            seeded[0] = False
            m = seeded[lab]
    if closing > 0:
        m = ndi.binary_closing(np.pad(m, closing), structure=_disk(closing))[closing:-closing, closing:-closing]
    return m


def instances(prob: np.ndarray, thr: float = 0.5, min_area: int = 30, min_peak: float = 0.0,
              low: float | None = None, closing: int = 0) -> np.ndarray:
    """prob: float[H,W] (原寸)。戻り値 bool[N,H,W]。成分の最大確率が min_peak 未満なら捨てる。"""
    if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1 + 1e-4:
        raise ValueError("probability map has NaN/inf or is outside [0,1]")
    lab, n = ndi.label(binarize(prob, thr, low, closing), structure=EIGHT)
    if n == 0:
        return np.zeros((0, *prob.shape), bool)
    idx = np.arange(1, n + 1)
    area = ndi.sum(np.ones_like(lab), lab, idx)
    peak = ndi.maximum(prob, lab, idx)
    keep = [i for i, a, p in zip(idx, area, peak) if a >= min_area and p >= min_peak]
    return np.stack([lab == i for i in keep]) if keep else np.zeros((0, *prob.shape), bool)


def check_disjoint(masks: np.ndarray) -> None:
    if len(masks) and (masks.sum(0) > 1).any():
        raise ValueError("overlapping predicted masks")
