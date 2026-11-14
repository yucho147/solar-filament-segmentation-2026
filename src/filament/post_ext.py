"""後処理の拡張（exp026）: ヒステリシス（低しきい値）・閉じ演算・穴埋めを持つ確率 -> 個体マスク。

処理順（`components_ext`）:
  1. 高しきい値 thr で二値化。t_lo（< thr）を与えた場合は「prob > t_lo の 8連結領域のうち、prob > thr の画素を含むものだけ残す」。
  2. close_r > 0 なら、二値マスクを半径 close_r の円板で閉じる（細い途切れの橋渡し）。成分化の「前」に行う
     （成分化の後では、途切れて別成分になった断片をつなげないため）。
     close_guard=True（既定）では、閉じた結果の 1 成分が「閉じる前の成分のうち面積 guard_area 画素以上のもの」を
     2 個以上含む場合、その成分では閉じる前のマスクに戻す。大きさのある別個体どうしが閉じ演算の橋で誤結合するのを防ぐ。
     小さな断片（guard_area 未満）が大きな成分につながるのは許す（途切れの修復が目的のため）。
  3. 8連結で成分化（重なりなし）。
  4. fill_holes=True なら、各成分の穴（4連結の背景で成分に囲まれた領域）を成分に加える。
     他の成分の画素は奪わない（背景だった画素のみ追加）ので、成分どうしは重ならない。
  5. 面積 min_area 以上、かつ成分内の最大確率 min_peak 以上の成分のみ残す。

t_lo=None（または thr 以上）・close_r=0・fill_holes=False のとき `postprocess.instances` と完全に一致する。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .postprocess import EIGHT, _disk

try:  # 閉じ演算は cv2 の方が速い。無ければ scipy
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

GUARD_AREA = 200  # close_guard で「大きさのある個体」とみなす閉じる前の成分面積（画素）


def check_prob(prob: np.ndarray) -> None:
    if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1 + 1e-4:
        raise ValueError("probability map has NaN/inf or is outside [0,1]")


def hysteresis_mask(prob: np.ndarray, thr: float, t_lo: float | None) -> np.ndarray:
    m = prob > thr
    if t_lo is not None and t_lo < thr:
        lab, n = ndi.label(prob > t_lo, structure=EIGHT)
        if n:
            seeded = np.zeros(n + 1, bool)
            seeded[np.unique(lab[m])] = True
            seeded[0] = False
            m = seeded[lab]
    return m


def close_mask(m: np.ndarray, r: int) -> np.ndarray:
    """半径 r の円板で閉じる。画像の外は背景扱いで端に人工的な塞がりを作らない（postprocess.binarize と同じ）。"""
    if r <= 0:
        return m
    if cv2 is not None:
        return _close_cv2(m, _disk(r).astype(np.uint8), r)
    return ndi.binary_closing(np.pad(m, r), structure=_disk(r))[r:-r, r:-r]


def _close_cv2(m: np.ndarray, k: np.ndarray, r: int) -> np.ndarray:
    # 余白 r を 0 で足してから閉じ、削る（scipy 版と同じ定義。cv2 の境界処理に依存しない）
    p = np.pad(m.astype(np.uint8), r)
    out = cv2.morphologyEx(p, cv2.MORPH_CLOSE, k)
    return out[r:-r, r:-r].astype(bool)


def guarded_close(m: np.ndarray, r: int, guard_area: int = GUARD_AREA) -> np.ndarray:
    """閉じ演算。ただし大きな成分（>= guard_area）を 2 個以上つなぐ橋は採用しない。"""
    closed = close_mask(m, r)
    if closed is m or not (closed & ~m).any():
        return closed
    lab0, n0 = ndi.label(m, structure=EIGHT)
    labc, nc = ndi.label(closed, structure=EIGHT)
    if nc == 0:
        return closed
    big = np.bincount(lab0.ravel(), minlength=n0 + 1) >= guard_area
    big[0] = False
    sel = big[lab0]  # 大きな元成分の画素
    pairs = np.unique(labc[sel].astype(np.int64) * (n0 + 1) + lab0[sel])
    cnt = np.bincount(pairs // (n0 + 1), minlength=nc + 1)
    bad = cnt >= 2
    bad[0] = False
    if not bad.any():
        return closed
    return (closed & ~bad[labc]) | m


def fill_component_holes(lab: np.ndarray, n: int) -> np.ndarray:
    """各成分の穴を埋める。背景（lab==0）だった画素だけを加え、他の成分は侵さない。lab を直接更新して返す。"""
    if n == 0:
        return lab
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        if sl is None:
            continue
        sub = lab[sl]
        h, w = sub.shape
        if h < 3 or w < 3:
            continue
        comp = sub == i
        add = ndi.binary_fill_holes(comp) & ~comp & (sub == 0)
        if add.any():
            sub[add] = i
    return lab


def components_ext(prob: np.ndarray, thr: float = 0.5, t_lo: float | None = None, close_r: int = 0,
                   fill_holes: bool = False, close_guard: bool = True,
                   guard_area: int = GUARD_AREA) -> tuple[np.ndarray, int, np.ndarray, np.ndarray]:
    """成分化まで。戻り値: (ラベル画像 int32[H,W], 成分数 n, 面積[n+1], 最大確率[n+1])。添字 0 は背景。"""
    m = hysteresis_mask(prob, thr, t_lo)
    if close_r > 0:
        m = guarded_close(m, close_r, guard_area) if close_guard else close_mask(m, close_r)
    lab, n = ndi.label(m, structure=EIGHT)
    if n == 0:
        return lab, 0, np.zeros(1, np.int64), np.zeros(1)
    if fill_holes:
        lab = fill_component_holes(lab, n)
    flat = lab.ravel()
    fg = np.flatnonzero(flat)
    labs = flat[fg]
    area = np.bincount(labs, minlength=n + 1)
    peak = np.zeros(n + 1)
    peak[1:] = ndi.maximum(prob.ravel()[fg], labs, np.arange(1, n + 1))
    return lab, n, area, peak


def instances_ext(prob: np.ndarray, params: dict) -> np.ndarray:
    """prob: float[H,W]（原寸）。params のキー: thr, t_lo, min_area, min_peak, close_r, fill_holes（close_guard, guard_area は任意）。
    戻り値は uint8[N,H,W]（互いに重ならない）。"""
    check_prob(prob)
    p = {"thr": 0.5, "t_lo": None, "min_area": 30, "min_peak": 0.0, "close_r": 0, "fill_holes": False,
         "close_guard": True, "guard_area": GUARD_AREA, **params}
    lab, n, area, peak = components_ext(prob, p["thr"], p["t_lo"], int(p["close_r"]), bool(p["fill_holes"]),
                                        bool(p["close_guard"]), int(p["guard_area"]))
    keep = [i for i in range(1, n + 1) if area[i] >= p["min_area"] and peak[i] >= p["min_peak"]]
    if not keep:
        return np.zeros((0, *prob.shape), np.uint8)
    return np.stack([(lab == i).astype(np.uint8) for i in keep])


def component_rles(lab: np.ndarray, n: int, offset: tuple[int, int] = (0, 0), full_shape: tuple[int, int] | None = None) -> list[dict]:
    """ラベル画像の各成分(1..n)を COCO RLE にする。`metrics.encode((lab==i)[None])[0]` と同じ結果を、前景画素の座標だけから高速に作る。"""
    from pycocotools import mask as mask_utils

    H, W = full_shape or lab.shape
    ys, xs = np.nonzero(lab)
    labs = lab[ys, xs].astype(np.int64)
    ys, xs = ys + offset[0], xs + offset[1]
    key = labs * (H * W) + xs.astype(np.int64) * H + ys  # 列優先の位置でラベルごとに並べる
    key.sort()
    labs_s = key // (H * W)
    f_s = key - labs_s * (H * W)
    bounds = np.searchsorted(labs_s, np.arange(1, n + 2))
    out = []
    for i in range(1, n + 1):
        f = f_s[bounds[i - 1]: bounds[i]]
        brk = np.flatnonzero(np.diff(f) > 1)
        st = f[np.concatenate(([0], brk + 1))]
        en = f[np.concatenate((brk, [len(f) - 1]))]
        counts = [int(st[0])]
        for j in range(len(st)):
            counts.append(int(en[j] - st[j] + 1))
            counts.append(int(st[j + 1] - en[j] - 1) if j + 1 < len(st) else int(H * W - 1 - en[j]))
        out.append(mask_utils.frPyObjects({"size": [H, W], "counts": counts}, H, W))
    return out


def region_crops(prob: np.ndarray, floor: float, margin: int = 3) -> list[tuple[int, int, np.ndarray]]:
    """prob > floor の画素を margin 画素ぶん膨張した領域(8連結)ごとに切り出す。(y0, x0, 切片)を返す。

    領域の外の画素は 0 にするので、floor 以上のしきい値で作る成分・ヒステリシス・閉じ(半径 <= margin)・穴埋めの結果は、
    画像全体で処理した場合と同じになる（領域どうしは margin 以上離れているため）。成分が疎な画像で数十倍速い。
    """
    reg = np.ascontiguousarray(prob > floor)
    if cv2 is not None:
        reg = cv2.dilate(reg.astype(np.uint8), np.ones((2 * margin + 1,) * 2, np.uint8)).astype(bool)
    else:  # pragma: no cover
        reg = ndi.binary_dilation(reg, structure=np.ones((2 * margin + 1,) * 2, bool))
    lab, n = ndi.label(reg, structure=EIGHT)
    out = []
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        crop = np.where(lab[sl] == i, prob[sl], 0).astype(prob.dtype)
        out.append((sl[0].start, sl[1].start, crop))
    return out


def components_regions(crops, full_shape, thr, t_lo=None, close_r=0, fill_holes=False, close_guard=True,
                       guard_area=GUARD_AREA) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """region_crops の結果から、全画像の成分の (面積, 最大確率, RLE) を返す。`components_ext` を画像全体に適用した結果と同じ成分集合。"""
    areas, peaks, rles = [], [], []
    for y0, x0, c in crops:
        lab, n, area, peak = components_ext(c, thr, t_lo, close_r, fill_holes, close_guard, guard_area)
        if n:
            areas.append(area[1:])
            peaks.append(peak[1:])
            rles += component_rles(lab, n, (y0, x0), full_shape)
    if not rles:
        return np.zeros(0, np.int64), np.zeros(0), []
    return np.concatenate(areas), np.concatenate(peaks), rles
