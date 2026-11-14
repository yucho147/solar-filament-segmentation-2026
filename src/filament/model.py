"""汎用エンコーダ(torchvision の EfficientNet / ResNet) + U-Net デコーダ。
1ch入力、前景ロジット1ch出力、入力と同解像度。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

_WEIGHTS = "DEFAULT"


def _conv(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
        nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
    )


class Encoder(nn.Module):
    """stride 2,4,8,16,32 の特徴を返す。EfficientNetは各strideで最後に出た特徴を採用する。"""

    def __init__(self, name: str, pretrained: bool):
        super().__init__()
        m = getattr(torchvision.models, name)(weights=_WEIGHTS if pretrained else None)
        self.resnet = name.startswith("resnet")
        if self.resnet:
            self.stem = nn.Sequential(m.conv1, m.bn1, m.relu)
            self.pool = m.maxpool
            self.layers = nn.ModuleList([m.layer1, m.layer2, m.layer3, m.layer4])
        else:
            self.blocks = m.features  # 最後のconv(1x1拡張)も含む
        with torch.no_grad():
            self.channels = [f.shape[1] for f in self._feats(torch.zeros(1, 3, 64, 64))]

    def _feats(self, x):
        if self.resnet:
            s0 = self.stem(x)
            f = [s0]
            y = self.pool(s0)
            for l in self.layers:
                y = l(y)
                f.append(y)
            return f
        out, last = {}, x.shape[-1]
        y = x
        for b in self.blocks:
            y = b(y)
            out[x.shape[-1] // y.shape[-1]] = y  # 同じstrideなら後の特徴で上書き
        return [out[s] for s in (2, 4, 8, 16, 32)]

    def forward(self, x):
        return self._feats(x)


class UNet(nn.Module):
    output_stride = 1  # 出力は入力と同解像度

    def __init__(self, encoder: str = "efficientnet_v2_s", pretrained: bool = True,
                 dec: tuple[int, ...] = (256, 128, 64, 32)):
        super().__init__()
        self.enc = Encoder(encoder, pretrained)
        ch = self.enc.channels  # [s2, s4, s8, s16, s32]
        prev = ch[4]
        self.dec = nn.ModuleList()
        for skip_c, d in zip((ch[3], ch[2], ch[1], ch[0]), dec):
            self.dec.append(_conv(prev + skip_c, d))
            prev = d
        self.up0 = _conv(prev, 32)
        self.head = nn.Conv2d(32, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def head_parameters(self):
        return list(self.dec.parameters()) + list(self.up0.parameters()) + list(self.head.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: float[B,1,H,W] 値域[0,1]。H,W は32の倍数。"""
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        f = self.enc(x)
        y = f[4]
        for blk, skip in zip(self.dec, (f[3], f[2], f[1], f[0])):
            y = F.interpolate(y, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            y = blk(torch.cat([y, skip], 1))
        y = F.interpolate(y, scale_factor=2, mode="bilinear", align_corners=False)
        return self.head(self.up0(y))


# ---- SegFormer（階層型 Transformer encoder + all-MLP decoder）-------------------------------
# 重み: NVIDIA Source Code License（非商用・研究/評価目的のみ）。コードは transformers (Apache-2.0)。
# revision を固定する。ライセンスの扱いは aidlc-docs/construction/implementation-candidates.md を参照。
SEGFORMER_VARIANTS = {
    "b0": dict(repo="nvidia/mit-b0", revision="80983a413c30d36a39c20203974ae7807835e2b4",
               hidden_sizes=[32, 64, 160, 256], depths=[2, 2, 2, 2], decoder_hidden_size=256),
    "b2": dict(repo="nvidia/mit-b2", revision="3bb39e8739149c3777d0325349b2a6c32c6413db",
               hidden_sizes=[64, 128, 320, 512], depths=[3, 4, 6, 3], decoder_hidden_size=768),
}


class SegFormer(nn.Module):
    """1ch入力、前景ロジット1ch出力。SegFormerの出力は入力の1/4なので、双一次補間で入力と同解像度へ戻す。

    出力の1セルは入力の4画素（factor=2の入力なら原寸の8画素）に相当する。細い対象が消えないかは
    `resolution.py` の上限診断で確認する。
    """

    output_stride = 4  # 仕上げ層(refine)ありでは入力と同解像度の格子で出力（実効の出力 stride は 1）

    def __init__(self, variant: str = "b2", pretrained: bool = True, refine: bool = False):
        super().__init__()
        self.refine = refine
        if refine:
            self.output_stride = 1
            self.stem = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                      nn.Conv2d(16, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
            self.fuse = nn.Sequential(nn.Conv2d(17, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(inplace=True),
                                      nn.Conv2d(16, 1, 1))
            nn.init.zeros_(self.fuse[-1].weight)  # 初期状態は元のSegFormerの出力と同じ
            nn.init.zeros_(self.fuse[-1].bias)
        from transformers import SegformerConfig, SegformerForSemanticSegmentation

        v = SEGFORMER_VARIANTS[variant]
        if pretrained:
            self.net = SegformerForSemanticSegmentation.from_pretrained(v["repo"], revision=v["revision"], num_labels=1)
        else:  # 重みを読み込む用。configはネットワークなしで組み立てる
            cfg = SegformerConfig(num_labels=1, num_encoder_blocks=4, depths=v["depths"], sr_ratios=[8, 4, 2, 1],
                                  hidden_sizes=v["hidden_sizes"], patch_sizes=[7, 3, 3, 3], strides=[4, 2, 2, 2],
                                  num_attention_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
                                  decoder_hidden_size=v["decoder_hidden_size"])
            self.net = SegformerForSemanticSegmentation(cfg)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: float[B,1,H,W] 値域[0,1]。H,W は32の倍数。"""
        x = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        lg = self.net(pixel_values=x).logits
        up = F.interpolate(lg, size=x.shape[-2:], mode="bilinear", align_corners=False)
        if not self.refine:
            return up
        return up + self.fuse(torch.cat([up, self.stem(x)], 1))  # 原寸の画素から、1/4格子で失われた細部を補う

    def head_parameters(self):
        extra = list(self.stem.parameters()) + list(self.fuse.parameters()) if self.refine else []
        return list(self.net.decode_head.parameters()) + extra


def build_model(arch: str, encoder: str, pretrained: bool, opts: dict | None = None) -> nn.Module:
    """arch: 'unet'（encoderはtorchvision名）| 'segformer_b0' | 'segformer_b2' | 'dinov3_seg'（encoder=large|base|small）。"""
    if arch == "dinov3_seg":
        from .model_dino import DinoV3Seg

        return DinoV3Seg(encoder, pretrained, **(opts or {}))
    if arch == "unet":
        return UNet(encoder, pretrained)
    if arch.startswith("segformer_"):  # segformer_b2 | segformer_b2_hr（原寸の仕上げ層つき）
        v = arch.split("_", 1)[1]
        return SegFormer(v.removesuffix("_hr"), pretrained, refine=v.endswith("_hr"))
    raise ValueError(f"unknown arch {arch}")
