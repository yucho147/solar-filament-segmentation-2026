"""探索c: 確率マップからの個体分離（成分の木の分割）と、分断・結合の診断に使う関数（新規ファイル、既存コードは変更しない）。

用語:
- 動的な深さ h（dynamics）: ある極大の確率（logit）から、その極大が別の極大とつながる最も高い水準（鞍点）までの落差。
  logit = log(p/(1-p))。確率が 0.99 付近で飽和するため、差を logit で測る。
- 木による分割 (split_label): 確率 > thr の8連結成分を出発点に、水準を logit で step ずつ上げて成分が2つ以上に分かれるとき、
  各子の最大 logit が（分かれた水準 + h）以上のものが2つ以上あれば分割とみなして再帰する。得た「核」を種にして
  元の成分の画素を scipy.ndimage.watershed_ift で割り当てる（画素は重ならない）。h=None は分割なし（既存の8連結成分と同じ）。
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), int)


def logit(p: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def _cores(L: np.ndarray, region: np.ndarray, lvl: float, h: float, step: float, depth: int = 0) -> list[np.ndarray]:
    """region（bool、crop）から、動的な深さ h 以上で分かれる核の領域のリストを返す。"""
    top = float(L[region].max())
    while True:
        lvl += step
        if lvl >= top:
            return [region]
        sub = region & (L > lvl)
        lab, n = ndi.label(sub, structure=EIGHT)
        if n <= 1:
            region = sub if n == 1 else region
            if n == 0:
                return [region]
            continue
        peaks = ndi.maximum(L, lab, np.arange(1, n + 1))
        sig = [i + 1 for i, pk in enumerate(peaks) if pk - lvl >= h]
        if len(sig) >= 2 and depth < 12:
            out = []
            for i in sig:
                out += _cores(L, lab == i, lvl, h, step, depth + 1)
            return out
        if len(sig) == 1:
            region = lab == sig[0]
        elif len(sig) == 0:
            return [region]


def split_label(prob: np.ndarray, thr: float, h: float | None, step: float = 0.25, min_area: int = 1) -> np.ndarray:
    """prob > thr の8連結成分をラベル化し、h が与えられれば動的な深さ h 以上で分かれる核ごとに分割する。int32[H,W]。"""
    lab, n = ndi.label(prob > thr, structure=EIGHT)
    if h is None or n == 0:
        return lab
    out = np.zeros_like(lab, dtype=np.int32)
    nxt = 1
    for i, sl in enumerate(ndi.find_objects(lab), start=1):
        m = lab[sl] == i
        if m.sum() < max(min_area, 2):
            out[sl][m] = nxt
            nxt += 1
            continue
        L = np.where(m, logit(prob[sl]), -50.0)
        cores = _cores(L, m, float(logit(np.float32(thr))), h, step)
        if len(cores) < 2:
            out[sl][m] = nxt
            nxt += 1
            continue
        mk = np.zeros(m.shape, np.int32)
        for j, c in enumerate(cores, start=1):
            mk[c] = j
        top = float(L[m].max())
        cost = np.where(m, np.clip((top - L) / (top + 10.0), 0, 1) * 65534, 65535).astype(np.uint16)
        ws = ndi.watershed_ift(cost, mk)
        ws = np.where(m, ws, 0)
        out[sl][m] = ws[m] + nxt - 1
        nxt += len(cores)
    return out


def components_from_label(prob: np.ndarray, lab: np.ndarray, min_area: int, gt_sets: list[list[dict]]) -> dict:
    """post_pos.components と同じ形式（thr は ラベル作成時に使った値を呼び出し側で入れる）。lab は int ラベル画像。"""
    from pycocotools import mask as mask_utils

    n = int(lab.max())
    out = {"area": np.zeros(0, int), "peak": np.zeros(0), "cy": np.zeros(0), "cx": np.zeros(0), "rle": [],
           "iou": [np.zeros((0, len(g))) for g in gt_sets]}
    if n == 0:
        return out
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    peak = np.asarray(ndi.maximum(prob, lab, np.arange(1, n + 1)))
    objs = ndi.find_objects(lab)
    ids = [i for i in range(1, n + 1) if area[i - 1] >= min_area]
    if not ids:
        return out
    H, W = prob.shape
    cy, cx, rles = [], [], []
    buf = np.zeros((H, W), np.uint8, order="F")
    for i in ids:
        sl = objs[i - 1]
        m = lab[sl] == i
        yy, xx = np.nonzero(m)
        cy.append(yy.mean() + sl[0].start)
        cx.append(xx.mean() + sl[1].start)
        buf[sl] = m
        rles.append(mask_utils.encode(buf))
        buf[sl] = 0
    out.update(area=area[np.array(ids) - 1], peak=peak[np.array(ids) - 1], cy=np.array(cy), cx=np.array(cx), rle=rles, ids=np.array(ids))
    out["iou"] = [np.asarray(mask_utils.iou(rles, g, [0] * len(g))).reshape(len(rles), len(g)) if g else
                  np.zeros((len(rles), 0)) for g in gt_sets]
    return out


def saddle_logit(L: np.ndarray, a: tuple[int, int], b: tuple[int, int], lo: float, step: float = 0.1) -> float:
    """点 a, b（crop内の座標）が同じ8連結成分に入る最も高い logit 水準（鞍点）。lo 以上で一度もつながらなければ -inf。"""
    top = min(L[a], L[b])
    lvl = top - 1e-6
    while lvl > lo:
        lab, _ = ndi.label(L > lvl, structure=EIGHT)
        if lab[a] > 0 and lab[a] == lab[b]:
            return float(lvl)
        lvl -= step
    return float("-inf")


def components_fast(prob: np.ndarray, lab: np.ndarray, min_area: int, gt_masks: list[list[np.ndarray]]) -> dict:
    """components_from_label と同じ結果（IoU は画素数から直接計算。RLE を作らないので速い）。gt_masks: 注釈集合ごとの bool[H,W] のリスト。"""
    n = int(lab.max())
    out = {"area": np.zeros(0, int), "peak": np.zeros(0), "cy": np.zeros(0), "cx": np.zeros(0), "ids": np.zeros(0, int),
           "iou": [np.zeros((0, len(g))) for g in gt_masks]}
    if n == 0:
        return out
    area_all = np.bincount(lab.ravel(), minlength=n + 1)
    ids = np.array([i for i in range(1, n + 1) if area_all[i] >= min_area])
    if len(ids) == 0:
        return out
    peak_all, cy_all, cx_all = fg_stats(prob, lab, n)
    out.update(area=area_all[ids], peak=peak_all[ids], cy=cy_all[ids], cx=cx_all[ids], ids=ids)
    ious = []
    for g in gt_masks:
        M = np.zeros((len(ids), len(g)))
        for j, G in enumerate(g):
            inter = np.bincount(lab[G], minlength=n + 1)[ids]
            M[:, j] = inter / (area_all[ids] + G.sum() - inter)
        ious.append(M)
    out["iou"] = ious
    return out


def fg_stats(prob: np.ndarray, lab: np.ndarray, n: int):
    """ラベル 1..n ごとの最大確率と重心 (cy, cx)。前景画素だけを使う（ndi.maximum より速い）。添字0は未使用。"""
    ys, xs = np.nonzero(lab)
    l = lab[ys, xs]
    v = prob[ys, xs]
    cnt = np.bincount(l, minlength=n + 1).astype(float)
    cnt[cnt == 0] = 1
    cy = np.bincount(l, weights=ys, minlength=n + 1) / cnt
    cx = np.bincount(l, weights=xs, minlength=n + 1) / cnt
    order = np.argsort(l, kind="stable")
    ls, vs = l[order], v[order]
    starts = np.r_[0, np.nonzero(np.diff(ls))[0] + 1]
    peak = np.zeros(n + 1)
    peak[ls[starts]] = np.maximum.reduceat(vs, starts)
    return peak, cy, cx
