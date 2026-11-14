"""出力解像度が細い対象を消さないかの診断。

「完全に正しい確率マップ」(正解の注釈集合平均) を、モデルの出力格子 (入力の縮小率 factor × 出力の stride) まで
面積平均で縮め、実際の推論と同じ双一次補間で原寸へ戻し、同じ後処理で個体化して PQ を測る。
モデルがどれだけ良くても、この上限以下になる（解像度の損失だけを分離した診断で、モデルの性能ではない）。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from pycocotools import mask as mask_utils
from scipy import ndimage as ndi

from .data import SIZE


def soft_target_fullres(set_rles: list[list[dict]]) -> np.ndarray:
    """注釈集合ごとの個体RLEを集合ごとに和集合にし、集合平均した原寸の確率 float32[2048,2048]。"""
    acc = np.zeros((SIZE, SIZE), np.float32)
    for g in set_rles:
        if g:
            acc += mask_utils.decode(g).max(2).astype(np.float32)
    return acc / max(len(set_rles), 1)


def through_grid(prob: np.ndarray, cell: int) -> np.ndarray:
    """原寸確率を cell x cell 画素ごとの平均へ縮め、双一次補間で原寸へ戻す（cell=1 は何もしない）。"""
    if cell == 1:
        return prob
    t = torch.from_numpy(prob)[None, None]
    low = F.avg_pool2d(t, cell)
    return F.interpolate(low, size=(SIZE, SIZE), mode="bilinear", align_corners=False)[0, 0].numpy()


def instance_widths(rles: list[dict]) -> np.ndarray:
    """個体ごとの最大幅(原寸画素) = 2 x 内接円の半径の最大値。"""
    out = []
    for r in rles:
        m = mask_utils.decode(r)
        out.append(2 * float(ndi.distance_transform_edt(np.pad(m, 1)).max()))
    return np.asarray(out)
