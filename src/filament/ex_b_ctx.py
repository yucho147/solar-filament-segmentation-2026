"""担当b 候補B: 縮小した全体画像の文脈と原寸512切片の併用（探索用の新規ファイル。既存コードは未編集）。

構成（ContextDinoSeg）:
  - 切片（原寸512）は従来どおり DinoV3Seg の経路（encoder 4層特徴 -> proj -> 高解像度decoder）。
  - 全体画像を面積平均で 1/ctx_factor に縮めたサムネイル（既定 2048 -> 512）を **同じ encoder（重み共有）** に通し、
    その 4層特徴を 1x1 畳み込みで dec_ch に射影する。切片が全体画像のどこにあるかの枠（サムネイル座標）で RoIAlign
    （枠内を双一次補間で 32x32 に切り出す）し、切片の 1/16 特徴 y に「加算」する。加算の前の畳み込みは 0 初期化なので、
    学習開始時点の出力は従来モデルと同一（初期の事前学習特徴を壊さない）。
  - 加算の出し入れは y = y + ctx_out(ReLU(BN(conv(roi)))) の形。位置チャネル(exp028)と違い画像の内容を運ぶ。
入力: x[B,1,512,512]、thumbs[T,1,G,G]（バッチ内の相異なる画像のサムネイル。同一画像の切片は共有）、
      idx[B]（各切片が使うサムネイルの番号）、boxes[B,4]（サムネイル座標の枠 x0,y0,x1,y1。小数可）。
ctx_grad=False なら全体画像の枝は no_grad（forwardのみ、活性値を保持しない）で、VRAMと時間の増分が小さい。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import roi_align

from .model_dino import DinoV3Seg, _cbr
from .native import META, NativeBatchDataset


def make_thumb(img: np.ndarray, factor: int) -> np.ndarray:
    """uint8[H,W] を factor x factor の面積平均で縮める（補間の位相ずれが無く、枠の座標は 1/factor になる）。"""
    h, w = img.shape
    if h % factor or w % factor:
        raise ValueError("image size must be divisible by factor")
    t = img.astype(np.float32).reshape(h // factor, factor, w // factor, factor).mean((1, 3))
    return t / 255.0


def d4_box(box: tuple[float, float, float, float], size: int, k: int, flip: int) -> tuple[float, float, float, float]:
    """正方形画像（一辺 size）に np.rot90(k) -> 左右反転(flip) を掛けたとき、元の座標系の枠(x0,y0,x1,y1)が移る先。
    native.forward（rot90 -> [:, ::-1]）と同じ順序。"""
    x0, y0, x1, y1 = box
    for _ in range(k % 4):  # np.rot90 は反時計回り: 点 (x, y) -> (y, size - x)
        x0, y0, x1, y1 = y0, size - x1, y1, size - x0
    if flip:
        x0, x1 = size - x1, size - x0
    return x0, y0, x1, y1


class ContextDinoSeg(DinoV3Seg):
    def __init__(self, *a, ctx_factor: int = 4, ctx_grad: bool = False, ctx_dim: int = 128, **kw):
        super().__init__(*a, **kw)
        self.ctx_factor, self.ctx_grad = ctx_factor, ctx_grad
        dec_ch = self.proj[0].out_channels
        self.ctx_proj = nn.Sequential(nn.Conv2d(self.enc.embed_dim * len(self.layers), ctx_dim, 1, bias=False),
                                      nn.BatchNorm2d(ctx_dim), nn.ReLU(inplace=True))
        self.ctx_mix = _cbr(ctx_dim, ctx_dim)
        self.ctx_out = nn.Conv2d(ctx_dim, dec_ch, 1)
        nn.init.zeros_(self.ctx_out.weight)  # 学習開始時は加算分が0（従来モデルと同じ出力）
        nn.init.zeros_(self.ctx_out.bias)

    def decoder_modules(self):
        return super().decoder_modules() + [("ctx", nn.ModuleList([self.ctx_proj, self.ctx_mix, self.ctx_out]))]

    def _features(self, x1: torch.Tensor) -> torch.Tensor:
        x3 = (x1.expand(-1, 3, -1, -1) - self.mean) / self.std
        feats = self.enc.forward_intermediates(x3, indices=self.layers, output_fmt="NCHW", intermediates_only=True)
        return torch.cat(feats, 1)

    def context_feature(self, thumbs: torch.Tensor) -> torch.Tensor:
        """thumbs[T,1,G,G] -> [T,ctx_dim,G/16,G/16]。"""
        if self.ctx_grad:
            f = self._features(thumbs)
        else:
            with torch.no_grad():
                f = self._features(thumbs)
        return self.ctx_proj(f)

    def forward(self, x, thumbs=None, idx=None, boxes=None):
        if thumbs is None:  # 文脈なし = 従来モデル
            return super().forward(x)
        crop = x[:, :1]
        feats = self._features(crop)
        y = self.proj(feats)
        g = self.context_feature(thumbs)  # [T,C,gh,gw]
        scale = 1.0 / 16  # サムネイル座標 -> 特徴格子
        rois = torch.cat([idx.view(-1, 1).to(boxes.dtype), boxes], 1)
        roi = roi_align(g, rois, output_size=y.shape[-2:], spatial_scale=scale, sampling_ratio=2, aligned=True)
        y = y + self.ctx_out(self.ctx_mix(roi))
        return self._decode(crop, y)

    def _decode(self, x, y):
        h1 = self.hr1((x - 0.45) / 0.225)
        h2 = self.hr2(h1)
        h4 = self.hr4(h2)
        h8 = self.hr8(h4)
        up = lambda t, ref: F.interpolate(t, size=ref.shape[-2:], mode="bilinear", align_corners=False)  # noqa: E731
        y = self.d8(torch.cat([up(y, h8), h8], 1))
        y = self.d4(torch.cat([up(y, h4), h4], 1))
        y = self.d2(torch.cat([up(y, h2), h2], 1))
        y = self.d1(torch.cat([up(y, h1), h1], 1))
        return self.head(y)


class NativeCtxDataset(NativeBatchDataset):
    """NativeBatchDataset の各バッチに (thumbs, idx, boxes) を足す。切片のD4に合わせてサムネイルも同じD4にする。
    返り値: (x, y, flags, meta, thumbs[T,1,G,G], idx[B], boxes[B,4])。T は相異なる画像×相異なるD4の組の数
    （同一画像の切片が同じD4なら共有、異なれば別サムネイル。最大でB）。"""

    def __init__(self, stems, cfg, length=None, ctx_factor: int = 4):
        super().__init__(stems, cfg, length)
        self.ctx_factor = ctx_factor
        self._thumbs: dict[int, np.ndarray] = {}

    def thumb(self, i: int) -> np.ndarray:
        if i not in self._thumbs:
            self._thumbs[i] = make_thumb(self.data[i][0], self.ctx_factor)
        return self._thumbs[i]

    def get_batch(self, b: int):
        x, y, fl, meta = super().get_batch(b)
        m = meta.numpy()
        f, w = self.ctx_factor, self.w
        keys, thumbs, idx, boxes = {}, [], [], []
        for r in m:
            i, y0, x0 = int(r[META.index("image")]), int(r[META.index("y0")]), int(r[META.index("x0")])
            k, fp = int(r[META.index("k")]), int(r[META.index("flip")])
            key = (i, k, fp)
            if key not in keys:
                t = self.thumb(i)
                t = np.rot90(t, k) if k else t
                t = t[:, ::-1] if fp else t
                keys[key] = len(thumbs)
                thumbs.append(np.ascontiguousarray(t, dtype=np.float32))
            G = self.thumb(i).shape[0]
            boxes.append(d4_box((x0 / f, y0 / f, (x0 + w) / f, (y0 + w) / f), G, k, fp))
            idx.append(keys[key])
        return (x, y, fl, meta, torch.from_numpy(np.stack(thumbs))[:, None], torch.tensor(idx),
                torch.tensor(boxes, dtype=torch.float32))


@torch.no_grad()
def predict_prob_ctx(model: ContextDinoSeg, img: np.ndarray, dev, tile: int = 512, overlap: float = 0.5,
                     weight: str = "uniform", bs: int = 8, tta_set: str = "flip", amp: bool = True) -> np.ndarray:
    """重なり窓＋全体サムネイル文脈の推論。TTAは「全体画像を変換 -> 窓推論 -> 逆変換」で、サムネイルと枠が自動で整合する。"""
    from . import engine

    model.eval()
    acc = 0
    variants = engine.tta_variants(True, tta_set)
    for v in variants:
        im = torch.from_numpy(img.astype(np.float32) / 255)[None, None].to(dev)
        im = engine._fwd(im, v)[0, 0].cpu().numpy()
        h = im.shape[0]
        th = torch.from_numpy(make_thumb((im * 255).round().astype(np.uint8), model.ctx_factor))[None, None].to(dev)
        ys = engine.tile_starts(h, tile, overlap)
        pos = [(a, c) for a in ys for c in ys]
        wt = engine.tile_weight(tile, weight)
        out = np.zeros((h, h), np.float32)
        wsum = np.zeros((h, h), np.float32)
        f = model.ctx_factor
        for i in range(0, len(pos), bs):
            chunk = pos[i:i + bs]
            x = torch.from_numpy(np.stack([im[a:a + tile, c:c + tile] for a, c in chunk]))[:, None].to(dev)
            boxes = torch.tensor([[c / f, a / f, (c + tile) / f, (a + tile) / f] for a, c in chunk], device=dev,
                                 dtype=torch.float32)
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
                lg = model(x, th, torch.zeros(len(chunk), dtype=torch.long, device=dev), boxes)
            p = torch.sigmoid(lg.float())[:, 0].cpu().numpy()
            for (a, c), pi in zip(chunk, p):
                out[a:a + tile, c:c + tile] += pi * wt
                wsum[a:a + tile, c:c + tile] += wt
        prob = torch.from_numpy(out / wsum)[None, None]
        acc = acc + engine._inv(prob, v)
    return (acc / len(variants))[0, 0].numpy()
