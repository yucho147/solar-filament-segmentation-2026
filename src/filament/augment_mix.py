"""画像をまたぐ拡張: Mosaic / CopyAndPaste / MaskDropout（exp024）。

AlbumentationsX にも同名の変換があるが、個体ごとのmask・bbox等のmetadataを要求し、教師が注釈集合平均のソフト値
（0〜1）である本案件では同期が複雑になる。そのため切片単位で自前実装し、**CropDatasetが持つ学習fit画像だけ**から素材を取る
（評価・holdout・近接時刻の画像と混ぜない。素材は呼び出し側が渡した data リストに限る）。
乱数は呼び出し側の Generator から取り、(seed, 標本番号)で決まる。
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi

EIGHT = np.ones((3, 3), int)
MIX_NAMES = ("mosaic", "copypaste", "maskdrop")


class MixBank:
    """fit画像の個体（教師>127の8連結成分）を前計算する。copypaste / maskdrop で使う。"""

    def __init__(self, data: list[tuple[np.ndarray, np.ndarray]], min_area: int = 150, max_side_frac: float = 0.45):
        self.data = data
        self.min_area, self.max_side_frac = min_area, max_side_frac
        self.comps: list[list[tuple[int, int, int, int, int]]] = []  # 画像ごとに (label, y0, y1, x0, x1)
        self.labels: list[np.ndarray] = []
        for _, t in data:
            fg = (t.max(0) if t.ndim == 3 else t) > 127
            lab, n = ndi.label(fg, structure=EIGHT)
            self.labels.append(lab.astype(np.int32))
            sl = ndi.find_objects(lab)
            area = ndi.sum(fg, lab, np.arange(1, n + 1)) if n else []
            cs = []
            for k, (s, a) in enumerate(zip(sl, area), 1):
                if a >= min_area and (s[0].stop - s[0].start) <= max_side_frac * fg.shape[0] and (s[1].stop - s[1].start) <= max_side_frac * fg.shape[0]:
                    cs.append((k, s[0].start, s[0].stop, s[1].start, s[1].stop))
            self.comps.append(cs)
        self.donors = [i for i, c in enumerate(self.comps) if c]


def mosaic(x, y, rng, bank, win_fn, w):
    """4分割: 位置(cy,cx)を境に、左上=元の切片、他の3領域は別のfit画像のランダム切片。境目に継ぎ目ができる。"""
    cy, cx = (int(rng.uniform(0.3, 0.7) * w) for _ in range(2))
    x, y = x.copy(), y.copy()
    for (ys, xs) in ((slice(0, cy), slice(cx, w)), (slice(cy, w), slice(0, cx)), (slice(cy, w), slice(cx, w))):
        j = int(rng.integers(len(bank.data)))
        img, t = bank.data[j]
        if t.ndim == 3:
            t = t[int(rng.integers(len(t)))]
        y0, x0 = win_fn(rng, j, w)
        x[ys, xs] = img[y0 : y0 + w, x0 : x0 + w][ys, xs]
        y[ys, xs] = t[y0 : y0 + w, x0 : x0 + w][ys, xs].astype(np.float32) / 255
    return x, y


def copypaste(x, y, rng, bank, w, n_range=(1, 3), rot=True):
    """donorの個体を、周囲の画像ごと（個体マスクを膨らませ縁をぼかした領域）別の位置へ貼る。教師は貼り付け範囲でdonorのソフト教師に置換。"""
    x, y = x.astype(np.float32).copy(), y.copy()
    if not bank.donors:
        return x.astype(np.uint8), y, 0
    n = int(rng.integers(n_range[0], n_range[1] + 1))
    done = 0
    for _ in range(n):
        j = bank.donors[int(rng.integers(len(bank.donors)))]
        k, y0, y1, x0, x1 = bank.comps[j][int(rng.integers(len(bank.comps[j])))]
        pad = 6
        y0, x0 = max(0, y0 - pad), max(0, x0 - pad)
        img, t = bank.data[j]
        y1, x1 = min(img.shape[0], y1 + pad), min(img.shape[1], x1 + pad)
        comp = bank.labels[j][y0:y1, x0:x1] == k
        patch = img[y0:y1, x0:x1].astype(np.float32)
        tp = (t.max(0) if t.ndim == 3 else t)[y0:y1, x0:x1].astype(np.float32) / 255
        core = ndi.binary_dilation(comp, iterations=2)
        alpha = cv2.GaussianBlur(ndi.binary_dilation(comp, iterations=4).astype(np.float32), (0, 0), 2.0)
        if rot:
            r, f = int(rng.integers(4)), rng.random() < 0.5
            patch, tp, core, alpha = (np.rot90(a, r) for a in (patch, tp, core, alpha))
            if f:
                patch, tp, core, alpha = (a[:, ::-1] for a in (patch, tp, core, alpha))
        h, ww = patch.shape
        if h >= w or ww >= w:
            continue
        py, px = int(rng.integers(w - h + 1)), int(rng.integers(w - ww + 1))
        # 貼り付け先は同じ太陽円盤の内側（画素が暗い円盤外の場所は避ける）
        if x[py : py + h, px : px + ww].mean() < 12:
            continue
        # 局所の背景明度を合わせる（donor周辺と貼り先周辺の平均差だけ補正）
        ring_d = patch[(alpha > 0.02) & ~core]
        dst = x[py : py + h, px : px + ww]
        ring_t = dst[(alpha > 0.02) & ~core]
        if len(ring_d) and len(ring_t):
            patch = patch + (ring_t.mean() - ring_d.mean())
        x[py : py + h, px : px + ww] = alpha * patch + (1 - alpha) * dst
        yy = y[py : py + h, px : px + ww]
        y[py : py + h, px : px + ww] = np.where(core, tp, yy * (1 - alpha) + 0 * alpha)  # 被覆された既存教師は貼り付け範囲で消える
        done += 1
    return np.clip(x + 0.5, 0, 255).astype(np.uint8), y, done


def maskdrop(x, y, rng, n_range=(1, 2)):
    """切片内の個体を1〜2本選び、画像の該当箇所を周囲から補間して消し、教師もゼロにする（見えない物を学習させない）。"""
    lab, n = ndi.label(y > 0.5, structure=EIGHT)
    if n == 0:
        return x, y, 0
    k = min(n, int(rng.integers(n_range[0], n_range[1] + 1)))
    ids = rng.choice(np.arange(1, n + 1), size=k, replace=False)
    m = np.isin(lab, ids)
    region = ndi.binary_dilation(m, iterations=3)
    x2 = cv2.inpaint(np.ascontiguousarray(x), region.astype(np.uint8), 3, cv2.INPAINT_TELEA)
    y2 = y.copy()
    y2[ndi.binary_dilation(m, iterations=2)] = 0.0
    return x2, y2, int(k)
