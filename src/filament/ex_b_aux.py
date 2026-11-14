"""担当b 候補C: 個体の分離・軸を学習に取り込む補助教師（境界・接触・spine・中心度）と補助出力層。探索用の新規ファイル。

補助教師は **注釈集合ごとに個体マスクから作り、注釈集合で平均**（主教師と同じ「平均＝一致度」の扱い）して uint8[0..255] にする。
  boundary : 個体の輪郭（内側 width 画素の帯）。個体の形そのもの
  touch    : 同じ注釈集合内の別個体と距離 <= gap 画素まで近づいた箇所（接触・近接。誤結合の原因になる場所）
  spine    : 配布の spine（個体の軸の折れ線）を線幅 line_w で描いたもの
  center   : 個体ごとの距離変換を個体内の最大値で割った値（0..1）。個体の中心ほど1
主教師（平均マスク）は変えない。補助出力は推論に使わなくてよい（候補として: 結合の分離に使う）。
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn

from .model_dino import DinoV3Seg

AUX_NAMES = ("boundary", "touch", "spine", "center")


def _k(r: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def instance_targets(masks: np.ndarray, spines: list[np.ndarray | None], width: int = 2, gap: int = 2,
                     line_w: int = 3) -> np.ndarray:
    """1注釈集合の個体マスク uint8[N,H,W]（0/1）と spine（各 float[M,2] の (x,y) 列、無ければNone）から
    float32[4,H,W]（AUX_NAMES順、0..1）を作る。N=0なら0。"""
    n = len(masks)
    h, w = masks.shape[-2:] if n else (1, 1)
    out = np.zeros((4, h, w), np.float32)
    if n == 0:
        return out
    dil = [cv2.dilate(m, _k(gap)) for m in masks]
    cnt = np.zeros((h, w), np.int32)
    for m in masks:
        cnt += m
    for i, m in enumerate(masks):
        er = cv2.erode(m, _k(width))
        out[0] = np.maximum(out[0], (m & (1 - er)).astype(np.float32))
        others = cnt - m  # 自分以外が占める画素数
        out[1] = np.maximum(out[1], ((dil[i] > 0) & (others > 0)).astype(np.float32))
        if m.any():
            d = cv2.distanceTransform(m, cv2.DIST_L2, 3)
            out[3] = np.maximum(out[3], d / max(float(d.max()), 1e-6))
        s = spines[i] if i < len(spines) else None
        if s is not None and len(s) >= 2:
            canvas = np.zeros((h, w), np.uint8)
            cv2.polylines(canvas, [np.round(np.asarray(s)).astype(np.int32).reshape(-1, 1, 2)], False, 1, line_w)
            out[2] = np.maximum(out[2], canvas.astype(np.float32))
    return out


def mean_over_sets(per_set: list[np.ndarray]) -> np.ndarray:
    """注釈集合ごとの float32[4,H,W] を平均し uint8[4,H,W]（0..255）にする（主教師と同じ平均）。"""
    return np.clip(np.mean(per_set, 0) * 255 + 0.5, 0, 255).astype(np.uint8)


def touch_stats(masks: np.ndarray, gap: int = 2) -> dict:
    """1注釈集合内で別個体と距離<=gap画素まで近い（または重なる）個体の数。誤結合の起こりやすさの診断。"""
    n = len(masks)
    if n < 2:
        return {"n": n, "touching": 0}
    cnt = masks.sum(0, dtype=np.int32)
    touching = 0
    for m in masks:
        d = cv2.dilate(m, _k(gap)) > 0
        touching += int((d & ((cnt - m) > 0)).any())
    return {"n": n, "touching": touching}


class AuxDinoSeg(DinoV3Seg):
    """DinoV3Seg の最終 head の入力（16ch特徴）を forward_pre_hook で取り、補助出力層 aux_head を足す。
    forward は [B, 1+n_aux, H, W]（ch0=主ロジット、ch1..=AUX_NAMES[:n_aux] のロジット）。aux_head は0初期化ではなく通常初期化
    （主経路へ影響しない: 主ロジットは従来と同一の計算。補助の勾配は共有 decoder へ入る）。"""

    def __init__(self, *a, n_aux: int = 4, **kw):
        super().__init__(*a, **kw)
        self.n_aux = n_aux
        self.aux_head = nn.Conv2d(self.head.in_channels, n_aux, 1)
        self._cap: torch.Tensor | None = None
        self.head.register_forward_pre_hook(lambda mod, inp: setattr(self, "_cap", inp[0]))

    def decoder_modules(self):
        return super().decoder_modules() + [("aux", self.aux_head)]

    def forward(self, x):
        main = super().forward(x)
        return torch.cat([main, self.aux_head(self._cap)], 1)


def aux_loss(aux_logit: torch.Tensor, aux_tgt: torch.Tensor, weights: tuple[float, ...] | None = None,
             pos_weight: float = 5.0) -> torch.Tensor:
    """aux_logit/aux_tgt: [B,n_aux,H,W]（tgt 0..1）。疎な線状教師なので、各チャネルをBCE(pos_weight)＋Diceにする。
    weights はチャネルごとの重み（既定は全て1/n_aux）。"""
    n = aux_logit.shape[1]
    wts = weights or tuple(1.0 / n for _ in range(n))
    p = torch.sigmoid(aux_logit)
    pw = torch.tensor(pos_weight, device=aux_logit.device)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(aux_logit, aux_tgt, reduction="none", pos_weight=pw)
    tp = (p * aux_tgt).sum((2, 3))
    dice = 1 - (2 * tp + 1) / (p.sum((2, 3)) + aux_tgt.sum((2, 3)) + 1)  # [B,n]
    total = 0
    for c in range(n):
        total = total + wts[c] * (bce[:, c].mean() + dice[:, c].mean())
    return total
