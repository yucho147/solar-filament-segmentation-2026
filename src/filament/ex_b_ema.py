"""担当b 候補A: 重みの指数移動平均（EMA: Exponential Moving Average）。探索用の新規ファイル（既存コードは未編集）。

学習中の重み w_t から w_ema <- d*w_ema + (1-d)*w_t を更新し、検証・保存・推論にはEMA重みを使う。
目的: 内側の検証lossの揺れ（500更新ごとの連続差の標準偏差が約0.03）に左右される更新数選択と、
外側再学習の崩れ（18 run中2 run）を、推論コスト0で和らげられるか。seed平均と似た効果（出力の分散を減らす）を狙う。
BatchNorm（decoderに多数）の running_mean/var もEMAする（学習時の統計と重みの整合のため）。整数バッファ（num_batches_tracked）はコピー。
d は更新数nに応じて min(decay, (1+n)/(10+n)) と立ち上げる（初期の未学習重みを引きずらない）。
"""
from __future__ import annotations

import copy
from contextlib import contextmanager

import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: bool = True):
        if not 0 < decay < 1:
            raise ValueError("decay must be in (0, 1)")
        self.decay, self.warmup, self.n = decay, warmup, 0
        # 勾配を持たない複製。device・dtypeは元モデルに従う（学習はfp32重み＋autocast）。
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def current_decay(self) -> float:
        return min(self.decay, (1 + self.n) / (10 + self.n)) if self.warmup else self.decay

    @torch.no_grad()
    def update(self, model: nn.Module) -> float:
        """optimizer.step() の直後に呼ぶ。使った減衰率を返す。"""
        d = self.current_decay()
        self.n += 1
        src, dst = model.state_dict(), self.ema.state_dict()
        for k, v in dst.items():
            s = src[k]
            if v.dtype.is_floating_point:
                v.mul_(d).add_(s.detach(), alpha=1 - d)
            else:
                v.copy_(s)
        return d

    @contextmanager
    def swapped_into(self, model: nn.Module):
        """with 内で model の重みを EMA 重みに入れ替え、抜けたら元に戻す（検証lossの計算・保存用）。"""
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.ema.state_dict())
        try:
            yield model
        finally:
            model.load_state_dict(backup)

    def state_dict(self) -> dict:
        return {"ema": self.ema.state_dict(), "n": self.n, "decay": self.decay}

    def load_state_dict(self, sd: dict) -> None:
        self.ema.load_state_dict(sd["ema"])
        self.n, self.decay = int(sd["n"]), float(sd["decay"])


def effective_horizon(decay: float) -> float:
    """EMAが実質的に平均する更新数（1/(1-d)）。decay=0.999 で約1000更新（cosine 12000更新の約8%）。"""
    return 1.0 / (1.0 - decay)
