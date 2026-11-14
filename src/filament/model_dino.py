"""DINOv3 ViT encoder（timm配布）＋高解像度の画素分岐を持つU-Net型decoder。

入力は512×512のグレースケール切片（0〜1）。出力は同じ512×512の前景ロジット。
入力画像・教師のリサイズは行わない。ViTの特徴格子は1/16（512入力で32×32）と粗いので、
(1) 4つの中間層の特徴を連結して1/16の特徴にし、
(2) 元画素から作る小さなCNN分岐（1/1, 1/2, 1/4, 1/8）をskip接続としてdecoderに渡し、
(3) decoderで 32→64→128→256→512 と段階的に拡大する（拡大は双一次補間＋学習する畳み込み。
    これはモデル内部の特徴の復元で、入力画像・教師・最終出力のリサイズではない）。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 公開（ゲートなし）のtimm配布。重みのrevision/sha256は provenance.dinov3_provenance で記録する。
TIMM_NAMES = {
    "large": "vit_large_patch16_dinov3.lvd1689m",
    "base": "vit_base_patch16_dinov3.lvd1689m",
    "small": "vit_small_patch16_dinov3.lvd1689m",
}
PATCH = 16
POS_MEAN, POS_STD = 0.5, 0.3  # 位置チャネル（rho, lat）をモデル内で標準化する定数（posch.py）


def _cbr(i: int, o: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(i, o, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(o),
                         nn.ReLU(inplace=True))


def _block(i: int, o: int) -> nn.Sequential:
    return nn.Sequential(_cbr(i, o), _cbr(o, o))


def _down(i: int, o: int) -> nn.Sequential:
    """stride-2の畳み込みで1/2へ（入力画素を平均で縮めず、学習する畳み込みで特徴化する）。"""
    return nn.Sequential(_cbr(i, o, stride=2), _cbr(o, o))


def default_layers(depth: int) -> list[int]:
    return [depth // 4 - 1, depth // 2 - 1, 3 * depth // 4 - 1, depth - 1]


def check_param_groups(model: nn.Module, groups: list[dict]) -> None:
    """学習対象（requires_grad）のparameterが、全groupで重複・欠落なく一度ずつ登録されていることを確認する。"""
    assigned = [id(p) for g in groups for p in g["params"]]
    if len(assigned) != len(set(assigned)):
        raise ValueError("optimizer groups register a parameter more than once")
    expected = {id(p) for p in model.parameters() if p.requires_grad}
    if set(assigned) != expected:
        raise ValueError(f"optimizer groups miss {len(expected - set(assigned))} trainable parameters "
                         f"or include {len(set(assigned) - expected)} frozen ones")
    if any(len(g["params"]) == 0 for g in groups):
        raise ValueError("empty optimizer group")


class DinoV3Seg(nn.Module):
    output_stride = 1  # 出力は入力と同じ画素格子

    def __init__(self, variant: str = "large", pretrained: bool = True, layers: list[int] | None = None,
                 dec_ch: int = 256, hr_ch: tuple[int, ...] = (16, 32, 64, 128), freeze_blocks: int = 0,
                 grad_ckpt: bool = False, timm_kwargs: dict | None = None, pos_channels: int = 0):
        super().__init__()
        import timm

        self.variant = variant
        self.enc = timm.create_model(TIMM_NAMES[variant], pretrained=pretrained, num_classes=0,
                                     **(timm_kwargs or {}))
        depth = len(self.enc.blocks)
        self.layers = list(layers) if layers is not None else default_layers(depth)
        if any(not 0 <= i < depth for i in self.layers) or len(set(self.layers)) != len(self.layers):
            raise ValueError(f"invalid intermediate layers {self.layers} for depth {depth}")
        if not 0 <= freeze_blocks <= depth:
            raise ValueError("freeze_blocks out of range")
        self.freeze_blocks = freeze_blocks
        if grad_ckpt:
            self.enc.set_grad_checkpointing(True)
        d = self.enc.embed_dim
        # forward_intermediates は最終のnormを通さないので、normと使わない最終層以降のparameterは学習対象から外す
        last = max(self.layers)
        self.unused_prefix = [f"blocks.{i}." for i in range(last + 1, depth)] + ["norm.", "fc_norm.", "head."]
        for n, p in self.enc.named_parameters():
            if any(n.startswith(u) for u in self.unused_prefix):
                p.requires_grad = False
        frozen = [f"blocks.{i}." for i in range(freeze_blocks)]
        if freeze_blocks:
            frozen += ["patch_embed.", "cls_token", "reg_token", "storage_tokens", "mask_token", "pos_embed"]
        for n, p in self.enc.named_parameters():
            if any(n.startswith(f) for f in frozen):
                p.requires_grad = False
        c1, c2, c4, c8 = hr_ch
        self.hr1 = _block(1, c1)  # 512 (元画素のまま)
        self.hr2 = _down(c1, c2)  # 256
        self.hr4 = _down(c2, c4)  # 128
        self.hr8 = _down(c4, c8)  # 64
        self.proj = nn.Sequential(nn.Conv2d(d * len(self.layers), dec_ch, 1, bias=False), nn.BatchNorm2d(dec_ch),
                                  nn.ReLU(inplace=True))
        self.d8 = _block(dec_ch + c8, 128)
        self.d4 = _block(128 + c4, 64)
        self.d2 = _block(64 + c2, 32)
        self.d1 = _block(32 + c1, 16)
        self.head = nn.Conv2d(16, 1, 1)
        # 位置チャネル（exp028）: 高解像度分岐の最初の畳み込みへ足す入力層。他の層より後に作り（乱数の消費順を変えない）、
        # 0で初期化する（学習開始時点の出力は位置チャネル無しと同じ）。encoderには入れない。
        self.pos_channels = pos_channels
        if pos_channels:
            self.pos_in = nn.Conv2d(pos_channels, c1, 3, padding=1, bias=False)
            nn.init.zeros_(self.pos_in.weight)
        self.register_buffer("mean", torch.tensor(self.enc.pretrained_cfg.get("mean", (0.485, 0.456, 0.406))).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(self.enc.pretrained_cfg.get("std", (0.229, 0.224, 0.225))).view(1, 3, 1, 1))

    def decoder_modules(self) -> list[tuple[str, nn.Module]]:
        mods = [("hr", nn.ModuleList([self.hr1, self.hr2, self.hr4, self.hr8])),
                ("proj", self.proj), ("dec", nn.ModuleList([self.d8, self.d4, self.d2, self.d1, self.head]))]
        return mods + ([("pos", self.pos_in)] if self.pos_channels else [])

    def param_groups(self, lr_opts: dict) -> list[dict]:
        """encoder（層別の学習率減衰つき）と新規decoder（hr分岐・proj・dec）で別の学習率にする。

        lr_opts: encoder_lr, decoder_lr, layer_decay。層iのlr = encoder_lr * layer_decay**(depth-1-i)、
        patch_embed等は最も浅い扱い（layer_decay**depth）。
        """
        enc_lr, dec_lr = float(lr_opts["encoder_lr"]), float(lr_opts["decoder_lr"])
        decay = float(lr_opts.get("layer_decay", 1.0))
        if not 0 < decay <= 1:
            raise ValueError("layer_decay must be in (0, 1]")
        depth = len(self.enc.blocks)
        groups: list[dict] = []
        in_block = set()
        for i, blk in enumerate(self.enc.blocks):
            ps = [p for p in blk.parameters() if p.requires_grad]
            in_block.update(id(p) for p in blk.parameters())
            if ps:
                groups.append({"params": ps, "lr": enc_lr * decay ** (depth - 1 - i), "name": f"enc_block{i:02d}",
                               "role": "encoder"})
        rest = [p for p in self.enc.parameters() if p.requires_grad and id(p) not in in_block]
        if rest:
            groups.insert(0, {"params": rest, "lr": enc_lr * decay ** depth, "name": "enc_embed", "role": "encoder"})
        for name, mod in self.decoder_modules():
            groups.append({"params": list(mod.parameters()), "lr": dec_lr, "name": f"dec_{name}", "role": "decoder"})
        check_param_groups(self, groups)
        return groups

    def head_parameters(self):  # 旧経路との互換（新規初期化されるdecoder側）
        return [p for _, m in self.decoder_modules() for p in m.parameters()]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: float[B,1+k,H,W]。ch0=画像（値域[0,1]）、ch1以降=位置チャネル（k=pos_channels、0なら従来と同じ1ch）。
        H,Wは16の倍数かつ64の倍数（512）。encoderへは画像chだけを3複製して渡す。"""
        if x.shape[1] != 1 + self.pos_channels:
            raise ValueError(f"expected {1 + self.pos_channels} input channels, got {x.shape[1]}")
        pos = x[:, 1:]
        x = x[:, :1]
        if x.shape[-1] % 64 or x.shape[-2] % 64:
            raise ValueError(f"input size must be a multiple of 64, got {tuple(x.shape[-2:])}")
        x3 = (x.expand(-1, 3, -1, -1) - self.mean) / self.std
        feats = self.enc.forward_intermediates(x3, indices=self.layers, output_fmt="NCHW", intermediates_only=True)
        y = self.proj(torch.cat(feats, 1))  # 1/16
        if self.pos_channels:  # 最初の畳み込みの出力へ位置チャネルの畳み込みを足す（入力チャネルを連結した畳み込みと等価）
            (conv, bn, act), tail = self.hr1[0], self.hr1[1]
            h1 = tail(act(bn(conv((x - 0.45) / 0.225) + self.pos_in((pos - POS_MEAN) / POS_STD))))
        else:
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
