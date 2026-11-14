"""学習・推論・評価の中核。入力は縮小(factor)画像、評価は原寸2048で行う。"""
from __future__ import annotations

import pickle
import random
import time
import csv
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger

from . import cache as cache_mod
from . import posch
from .data import SIZE
from .metrics import PQStats, encode, pq_stats
from .model import UNet
from .postprocess import instances


@dataclass
class Config:
    factor: int = 2  # 2048 -> 1024
    crop: int = 512
    arch: str = "unet"  # unet（encoderで指定）| segformer_b0 | segformer_b2
    encoder: str = "efficientnet_v2_s"  # arch=unet のときだけ使う
    pretrained: bool = True
    head_lr_mult: float = 1.0  # decoderのlr倍率（SegFormerは新規初期化のheadを10倍にするのが原論文の設定）
    steps: int = 3000
    batch: int = 8
    lr: float = 3e-4
    wd: float = 1e-4
    seed: int = 42
    pos_crop_prob: float = 0.7  # 前景を含む位置を優先して切り出す確率
    eval_every: int = 1000
    thr: float = 0.5
    min_area: int = 30
    tta: bool = True
    tta_set: str = "flip"  # flip=元/水平/垂直の3通り（従来）| d4=90度回転×反転の8通り | none
    target_mode: str = "mean"  # mean=注釈集合の平均ソフト教師 | random_set=毎回1集合を乱数で選ぶ(契約のbaseline)
    amp: bool = True  # CUDAのときbf16 autocast
    loss: str = "bce_dice"  # bce_dice | focal_dice | focal_tversky
    focal_gamma: float = 2.0
    focal_alpha: float = 0.5  # 0.5で前景/背景の重みが等しい。小さくすると前景を軽く見る
    tversky_alpha: float = 0.3  # FPの重み（小さいほどFPに寛容）
    tversky_beta: float = 0.7  # FNの重み
    # 学習時拡張: preset名（augment.PRESETS）か {preset, off, set}。sampling: augment.SAMPLING のキーか dict
    aug: object = "legacy"
    sampling: object = "legacy"
    workers: int = 0  # DataLoader worker数。乱数は (seed, 標本番号) で決まり、worker数に依らない
    # 推論: mode=full は画像全体を一度に、tile はsliding window。weight=uniform|cos|gauss
    infer: dict = field(default_factory=lambda: {"mode": "full", "tile": 512, "overlap": 0.5, "weight": "uniform"})
    # 画像をまたぐ拡張（augment_mix）: {mosaic: {p}, copypaste: {p, n: [1,3]}, maskdrop: {p, n: [1,2]}}。素材は学習fit画像のみ
    mix: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)
    early_stopping: dict = field(default_factory=dict)
    # 原寸（factor=1）の512切片で学習する設定（native.DEFAULT参照）。空なら従来のCropDataset
    native: dict = field(default_factory=dict)
    # arch固有の構成（dinov3_seg: layers, dec_ch, freeze_blocks, grad_ckpt, timm_kwargs）
    model_opts: dict = field(default_factory=dict)
    # 層別学習率（modelがparam_groupsを持つとき）: encoder_lr, decoder_lr, layer_decay
    lr_opts: dict = field(default_factory=dict)
    lr_schedule_steps: int = 0  # 0: OneCycleLRの長さ=steps。正値: その更新数で最小学習率に達し、以後保持
    pct_start: float = 0.05  # OneCycleLRのwarmup割合（scheduler長に対する）
    lr_scheduler: str = "onecycle"  # onecycle | cosine（線形warmup→cosine減衰→最小学習率を保持）
    min_lr_ratio: float = 0.01  # cosineの最終学習率 = max_lr × この比（onecycleでは使わない）

    def __post_init__(self):  # -o infer={mode: tile} のように一部だけ上書きできるようにする
        self.infer = {**{"mode": "full", "tile": 512, "overlap": 0.5, "weight": "uniform"}, **self.infer}


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def reset_peak_memory(dev: torch.device) -> None:
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb(dev: torch.device) -> float | None:
    """CUDAは学習中の最大確保量。MPSは現在のドライバ確保量（最大値ではない）。CPUはNone。"""
    if dev.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if dev.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ema_settings(cfg: Config) -> dict | None:
    """Config.extra["ema"] から重みのEMA（指数移動平均）の仕様を返す。無効（未指定・空）なら None（従来動作）。

    例: {decay: 0.999, extra_decays: [0.9998]}。decays は主と追加を合わせた重複なしの昇順なしリスト（指定順）。
    warmup=true なら開始直後の減衰率を min(decay, (1+n)/(10+n)) に立ち上げる（ModelEMAの既定）。注意: この式では
    n=5000更新で約0.9982、n=8991更新でやっと0.999に達する。よって約9000更新未満の学習では、0.9982以上のdecayは
    全て同一のEMAになる（例: 5000更新の再学習で 0.999 と 0.9998 は同じ重み）。異なる時定数を比べるなら
    warmup=false か、0.9982未満のdecayを使う。実際に使った最終decayは .aug.json の ema.last_decay_used に残る。
    validate_decay: 内側の検証loss・最良チェックポイント選択にそのdecayのEMA重みを使う（既定 null=ライブ重み。
    early_stopping が有効な内側選択だけに影響し、外側再学習には影響しない）。
    """
    e = (cfg.extra or {}).get("ema")
    if not e:
        return None
    unknown = set(e) - {"decay", "extra_decays", "warmup", "validate_decay"}
    if unknown:
        raise ValueError(f"unknown ema keys {sorted(unknown)}")
    decays = [float(e.get("decay", 0.999)), *[float(d) for d in e.get("extra_decays", [])]]
    if len(set(decays)) != len(decays) or not all(0 < d < 1 for d in decays):
        raise ValueError(f"ema decays must be distinct values in (0, 1): {decays}")
    vd = e.get("validate_decay")
    if vd is not None and float(vd) not in decays:
        raise ValueError(f"ema.validate_decay {vd} must be one of {decays}")
    return {"decays": decays, "warmup": bool(e.get("warmup", True)), "validate_decay": None if vd is None else float(vd)}


def ema_tag(decay: float) -> str:
    return f"{decay:g}"


def ema_weights_path(weights: Path, decay: float) -> Path:
    """weights.pt -> weights_ema_0.999.pt（同じディレクトリ）。"""
    return weights.with_name(f"{weights.stem}_ema_{ema_tag(decay)}{weights.suffix}")


class CropDataset(torch.utils.data.Dataset):
    """標本番号 i → (画像crop, ソフト教師crop, 適用フラグ)。乱数は (seed, i) だけで決まる。

    切片の選び方: fg（前景を含む）／disc（太陽円盤内の任意位置）／uniform（画像全体で一様）。
    窓を大きめに取り、幾何変換後に中央から crop する（変換の境界埋めが入力に入らないようにする）。
    """

    def __init__(self, stems: list[str], cfg: Config, length: int | None = None):
        from . import augment as au

        self.cfg = cfg
        self.spec = au.resolve(cfg.aug)
        self.samp = dict(au.SAMPLING[cfg.sampling]) if isinstance(cfg.sampling, str) else dict(cfg.sampling)
        self.pipe = au.Pipeline(self.spec, cfg.crop)
        self.length = length or cfg.steps * cfg.batch
        if cfg.target_mode == "random_set":
            self.data = [cache_mod.load_sets_cfg(s, cfg) for s in stems]  # tgt: [S,H,W]
            self.fg = [np.argwhere(t.max(0) > 127) for _, t in self.data]
        elif cfg.target_mode == "mean":
            self.data = [cache_mod.load_cfg(s, cfg) for s in stems]
            self.fg = [np.argwhere(t > 127) for _, t in self.data]
        else:
            raise ValueError(f"unknown target_mode {cfg.target_mode}")
        self.samp.setdefault("bg", 0.0)
        need_disc = self.samp.get("disc", 0) > 0 or self.samp["bg"] > 0
        self.disc = [au.find_disc(img) for img, _ in self.data] if need_disc else None
        self.fgmask = [(t.max(0) if t.ndim == 3 else t) > 127 for _, t in self.data] if self.samp["bg"] > 0 else None
        self.stems = stems
        self.mix = cfg.mix or {}
        if self.mix:
            from . import augment_mix as am

            if cfg.target_mode != "mean":
                raise ValueError("mix requires target_mode=mean")
            bad = set(self.mix) - set(am.MIX_NAMES)
            if bad:
                raise ValueError(f"unknown mix keys {sorted(bad)}")
            self.bank = am.MixBank(self.data)

    def __len__(self):
        return self.length

    def window(self, rng, i: int, w: int, samp: dict | None = None) -> tuple[int, int, str]:
        """画像iの窓(w×w)の左上と、選んだ方式。sampは切片の選び方（省略時はself.samp）。"""
        img = self.data[i][0]
        h = img.shape[0]
        u = rng.random()
        s = samp or self.samp
        has_disc = self.disc is not None and len(self.disc[i].centers) > 0
        if u < s["fg"] and len(self.fg[i]):
            mode = "fg"
        elif u < s["fg"] + s["disc"] and has_disc:
            mode = "disc"
        elif u < s["fg"] + s["disc"] + s["bg"] and has_disc:
            mode = "bg"
        else:
            mode = "uniform"
        if mode == "bg":  # 円盤内で前景を含まない窓を探す。見つからなければ円盤内の窓で代用
            for _ in range(64):
                cy, cx = self.disc[i].centers[int(rng.integers(len(self.disc[i].centers)))]
                y0, x0 = int(np.clip(cy - w // 2, 0, h - w)), int(np.clip(cx - w // 2, 0, h - w))
                c = min(w, self.cfg.crop)  # 幾何変換後に残る中央部だけが前景ゼロであればよい（拡大縮小で端の前景が少し入りうる）
                o = (w - c) // 2
                if not self.fgmask[i][y0 + o : y0 + o + c, x0 + o : x0 + o + c].any():
                    break
            else:
                mode = "disc"
        elif mode == "fg":
            cy, cx = self.fg[i][int(rng.integers(len(self.fg[i])))]
            y0 = int(np.clip(cy - rng.integers(w), 0, h - w))
            x0 = int(np.clip(cx - rng.integers(w), 0, h - w))
        elif mode == "disc":
            cy, cx = self.disc[i].centers[int(rng.integers(len(self.disc[i].centers)))]
            y0, x0 = int(np.clip(cy - w // 2, 0, h - w)), int(np.clip(cx - w // 2, 0, h - w))
        else:
            y0, x0 = (int(rng.integers(h - w + 1)) for _ in range(2))
        return y0, x0, mode

    def get(self, idx: int):
        from . import augment as au

        rng = np.random.default_rng([self.cfg.seed, idx])
        i = int(rng.integers(len(self.data)))
        img, tgt = self.data[i]
        w = min(self.pipe.win, img.shape[0])
        for _ in range(8):  # 円盤が少ししか入らない窓は捨てる（mixed のとき）
            y0, x0, mode = self.window(rng, i, w)
            if mode in ("fg", "bg") or self.disc is None or self.disc[i].mask[y0 : y0 + w, x0 : x0 + w].mean() >= self.samp["min_disc"]:
                break
        if tgt.ndim == 3:  # random_set: 1集合を選ぶ
            tgt = tgt[int(rng.integers(len(tgt)))]
        x = img[y0 : y0 + w, x0 : x0 + w]
        y = tgt[y0 : y0 + w, x0 : x0 + w].astype(np.float32) / 255
        mixed = dict.fromkeys(("mosaic", "copypaste", "maskdrop"), False)
        if self.mix:
            from . import augment_mix as am

            if "mosaic" in self.mix and rng.random() < self.mix["mosaic"].get("p", 0.5):
                x, y = am.mosaic(x, y, rng, self.bank, lambda r, j, ww: self.window(r, j, ww)[:2], w)
                mixed["mosaic"] = True
            if "copypaste" in self.mix and rng.random() < self.mix["copypaste"].get("p", 0.5):
                x, y, n = am.copypaste(x, y, rng, self.bank, w, tuple(self.mix["copypaste"].get("n", (1, 3))))
                mixed["copypaste"] = n > 0
            if "maskdrop" in self.mix and rng.random() < self.mix["maskdrop"].get("p", 0.3):
                x, y, n = am.maskdrop(x, y, rng, tuple(self.mix["maskdrop"].get("n", (1, 2))))
                mixed["maskdrop"] = n > 0
        pipe = self.pipe if w == self.pipe.win else au.Pipeline(self.spec, self.cfg.crop)
        x, y, applied = pipe(x, y, rng)
        applied.update(mixed)
        return x, y, i, mode, applied

    def __getitem__(self, idx: int):
        from . import augment as au

        x, y, i, mode, applied = self.get(idx)
        flags = [applied[n] for n in au.NAMES] + [applied.get(n, False) for n in ("mosaic", "copypaste", "maskdrop")] + [mode == m for m in ("fg", "disc", "bg", "uniform")]
        return (torch.from_numpy(x.astype(np.float32) / 255)[None], torch.from_numpy(y)[None],
                torch.tensor(flags, dtype=torch.int32))


def _flag_names() -> list[str]:
    from . import augment as au

    return list(au.NAMES) + ["mosaic", "copypaste", "maskdrop", "crop_fg", "crop_disc", "crop_bg", "crop_uniform"]


def loss_fn(logit: torch.Tensor, y: torch.Tensor, cfg: Config | None = None) -> torch.Tensor:
    """y はソフト教師[0,1]。focal項は |y-p|^gamma を画素のBCEに掛ける（ソフト教師にも使える形）。"""
    cfg = cfg or Config()
    p = torch.sigmoid(logit)
    bce = F.binary_cross_entropy_with_logits(logit, y, reduction="none")
    if cfg.loss.startswith("focal"):
        w = (cfg.focal_alpha * y + (1 - cfg.focal_alpha) * (1 - y)) * 2  # alpha=0.5で係数1
        bce = bce * w * (y - p).abs().pow(cfg.focal_gamma)
    elif cfg.loss != "bce_dice":
        raise ValueError(f"unknown loss {cfg.loss}")
    bce = bce.mean()
    tp = (p * y).sum((1, 2, 3))
    if cfg.loss == "focal_tversky":
        fp, fn = (p * (1 - y)).sum((1, 2, 3)), ((1 - p) * y).sum((1, 2, 3))
        tv = (tp + 1) / (tp + cfg.tversky_alpha * fp + cfg.tversky_beta * fn + 1)
        return bce + (1 - tv).pow(0.75).mean()  # focal Tversky: (1-TI)^(1/gamma), gamma=4/3
    dice = 1 - (2 * tp + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)
    return bce + dice.mean()


def tta_variants(tta: bool, kind: str) -> list[tuple[int, bool]]:
    """(90度回転数k, 水平反転) の列。入力へ k回転→反転を適用し、出力へ逆変換を適用して平均する。"""
    if not tta or kind == "none":
        return [(0, False)]
    if kind == "d4":
        return [(k, f) for k in range(4) for f in (False, True)]
    if kind == "flip":
        return [(0, False), (0, True), (2, True)]  # (2,True)=垂直反転（180度回転+水平反転）
    raise ValueError(f"unknown tta_set {kind}")


def _fwd(x: torch.Tensor, v: tuple[int, bool]) -> torch.Tensor:
    k, f = v
    x = torch.rot90(x, k, (2, 3)) if k else x
    return x.flip(3) if f else x


def _inv(x: torch.Tensor, v: tuple[int, bool]) -> torch.Tensor:
    k, f = v
    x = x.flip(3) if f else x
    return torch.rot90(x, -k, (2, 3)) if k else x


@torch.no_grad()
def predict_prob(models: list[UNet], img: np.ndarray, dev, tta: bool = True, amp: bool = True,
                 tta_set: str = "flip", pos: np.ndarray | None = None) -> np.ndarray:
    """img: uint8[H,H] (縮小済み)。モデル・TTA平均の確率 float32[H,H]。"""
    x = torch.from_numpy(img.astype(np.float32) / 255)[None, None].to(dev)
    if pos is not None:  # 位置チャネル（exp028）: [k,H,W]。TTAの変換は全チャネルに掛かる
        x = torch.cat([x, torch.from_numpy(pos)[None].to(dev)], 1)
    acc, n = 0, 0
    for m in models:
        m.eval()
        for v in tta_variants(tta, tta_set):
            with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
                lg = m(_fwd(x, v))
            acc = acc + _inv(torch.sigmoid(lg.float()), v)
            n += 1
    return (acc / n)[0, 0].float().cpu().numpy()


def tile_weight(tile: int, kind: str) -> np.ndarray:
    """tile内の画素の重み。中心ほど文脈が十分なので大きくする（cos/gauss）。端でも0にはしない。"""
    if kind == "uniform":
        return np.ones((tile, tile), np.float32)
    t = (np.arange(tile) + 0.5) / tile * 2 - 1  # -1..1
    if kind == "cos":
        w1 = 0.5 * (1 + np.cos(np.pi * t)) + 0.02
    elif kind == "gauss":
        w1 = np.exp(-0.5 * (t / 0.5) ** 2) + 0.02
    else:
        raise ValueError(f"unknown weight {kind}")
    return np.outer(w1, w1).astype(np.float32)


def tile_starts(n: int, tile: int, overlap: float) -> list[int]:
    if n <= tile:
        return [0]
    stride = max(1, int(round(tile * (1 - overlap))))
    xs = list(range(0, n - tile + 1, stride))
    if xs[-1] != n - tile:
        xs.append(n - tile)  # 端まで必ず覆う
    return xs


@torch.no_grad()
def predict_prob_tiled(models, img: np.ndarray, dev, tta: bool = True, amp: bool = True, tile: int = 512,
                       overlap: float = 0.5, weight: str = "uniform", bs: int = 8,
                       tta_set: str = "flip", posmap: np.ndarray | None = None) -> np.ndarray:
    """sliding window。posmap（位置チャネル[k,H,W]、exp028）があれば、窓ごとに同じ座標で切り出して画像の後ろに連結する。tileごとにモデル・TTAの確率を出し、重み付き平均で統合してからしきい値処理へ渡す。"""
    h, w = img.shape
    ys, xs = tile_starts(h, tile, overlap), tile_starts(w, tile, overlap)
    pos = [(y, x) for y in ys for x in xs]
    wt = tile_weight(tile, weight)
    acc = np.zeros((h, w), np.float32)
    wsum = np.zeros((h, w), np.float32)
    for i in range(0, len(pos), bs):
        chunk = pos[i : i + bs]
        x = torch.from_numpy(np.stack([img[y : y + tile, x0 : x0 + tile] for y, x0 in chunk]).astype(np.float32)
                             / 255)[:, None].to(dev)
        if posmap is not None:
            x = torch.cat([x, torch.from_numpy(np.stack([posmap[:, y:y + tile, x0:x0 + tile] for y, x0 in chunk])).to(dev)], 1)
        tot, n = 0, 0
        for m in models:
            m.eval()
            for v in tta_variants(tta, tta_set):
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
                    lg = m(_fwd(x, v))
                tot = tot + _inv(torch.sigmoid(lg.float()), v)
                n += 1
        p = (tot / n)[:, 0].cpu().numpy()
        for (y, x0), pi in zip(chunk, p):
            acc[y : y + tile, x0 : x0 + tile] += pi * wt
            wsum[y : y + tile, x0 : x0 + tile] += wt
    if (wsum <= 0).any():
        raise RuntimeError("tiles do not cover the image")
    return acc / wsum


def predict(models, img: np.ndarray, dev, cfg: "Config") -> np.ndarray:
    """cfg.infer に従う確率（縮小解像度）。"""
    inf = cfg.infer
    if inf["mode"] == "full":
        return predict_prob(models, img, dev, cfg.tta, cfg.amp, cfg.tta_set, pos=posch.full_map(cfg, img.shape))
    return predict_prob_tiled(models, img, dev, cfg.tta, cfg.amp, inf["tile"], inf["overlap"], inf["weight"],
                              tta_set=cfg.tta_set, posmap=posch.full_map(cfg, img.shape))


def upsample(prob: np.ndarray, size: int = SIZE) -> np.ndarray:
    if prob.shape[-2:] == (size, size):  # 原寸の確率はそのまま返す（補間を通さない）
        return prob
    t = torch.from_numpy(prob)[None, None]
    return F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()


def build(cfg: Config, pretrained: bool | None = None):
    from .model import build_model

    return build_model(cfg.arch, cfg.encoder, cfg.pretrained if pretrained is None else pretrained,
                       {**cfg.model_opts, **posch.model_opts(cfg)})


def load_models(paths: list[Path], cfg: Config, dev) -> list:
    ms = []
    for p in paths:
        m = build(cfg, pretrained=False)
        state = torch.load(p, map_location="cpu")
        posch.check_state(state, cfg)  # 位置チャネルの有無・本数が設定と違う重みは拒否
        m.load_state_dict(state)
        ms.append(m.to(dev))
    return ms


def prob_loss(prob: torch.Tensor, y: torch.Tensor, cfg: Config) -> torch.Tensor:
    """確率(0〜1)からの損失。logit(clip(p))を損失関数へ渡す（窓の確率平均を原寸で再構成したあとに使う）。"""
    p = prob.clamp(1e-6, 1 - 1e-6)
    return loss_fn(torch.log(p) - torch.log1p(-p), y, cfg)


def eval_tiles(cfg: Config) -> dict | None:
    """検証損失の評価に使うsliding windowの設定。未指定なら従来の画像一括forward。"""
    t = cfg.early_stopping.get("eval_tiles")
    return {"tile": cfg.crop, "overlap": 0.0, "batch": 8, "tta": False, **t} if t else None


def validation_loss(stems: list[str], model, cfg: Config, dev) -> float:
    """無拡張の画像全体と注釈集合の平均教師を、画像単位で平均する。

    early_stopping.eval_tiles が指定されていれば、原寸画像をsliding window（既定512・重なりなし・TTAなし）で
    推論し、窓ごとの確率を画素単位で重み1の平均にして原寸の確率を作ってから損失を計算する（2048全体の一括forwardはしない）。
    内側fitの固定画像（train側）と内側tuneは同じ関数・同じ設定で評価する。
    """
    if not stems or len(stems) != len(set(stems)):
        raise ValueError("validation IDs must be nonempty and unique")
    values = []
    tiles = eval_tiles(cfg)
    model.eval()
    with torch.inference_mode():
        for stem in stems:
            image, target = cache_mod.load_cfg(stem, cfg)
            y = torch.from_numpy(target.astype(np.float32) / 255)[None, None]
            if tiles:
                prob = predict_prob_tiled([model], image, dev, tiles["tta"], cfg.amp, tiles["tile"], tiles["overlap"],
                                          "uniform", tiles["batch"], posmap=posch.full_map(cfg, image.shape))
                if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
                    raise FloatingPointError(f"non-finite or out-of-range probability for {stem}")
                value = float(prob_loss(torch.from_numpy(prob)[None, None], y, cfg).item())
            else:
                x = torch.from_numpy(image.astype(np.float32) / 255)[None, None].to(dev)
                if posch.full_map(cfg, image.shape) is not None:
                    x = torch.cat([x, torch.from_numpy(posch.full_map(cfg, image.shape))[None].to(dev)], 1)
                with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cfg.amp and dev.type == "cuda"):
                    logits = model(x)
                value = float(loss_fn(logits.float(), y.to(dev), cfg).item())
            if not math.isfinite(value):
                raise FloatingPointError(f"non-finite validation loss for {stem}")
            values.append(value)
    return float(np.mean(values))


def fixed_subset(ids: list[str], n: int, seed: int, salt: int) -> list[str]:
    """n>0なら、IDの集合から決定的にn枚を選ぶ（順序に依存しない）。0または全数以上なら全体。"""
    ids = sorted(ids)
    if n <= 0 or n >= len(ids):
        return ids
    pick = np.random.default_rng([seed, salt]).choice(len(ids), size=n, replace=False)
    return [ids[i] for i in sorted(pick)]


def make_optimizer(model, cfg: Config):
    """(optimizer, max_lr)。modelがparam_groupsを持ち lr_opts があれば encoder/decoder 別・層別の学習率。"""
    from .model_dino import check_param_groups

    if cfg.lr_opts:
        if not hasattr(model, "param_groups"):
            raise ValueError(f"arch {cfg.arch} does not support lr_opts")
        groups = model.param_groups(cfg.lr_opts)
    elif cfg.head_lr_mult != 1.0:  # headだけ学習率を変える（SegFormer: encoder lr, head lr x10）
        head = list(model.head_parameters())
        ids = {id(p) for p in head}
        groups = [{"params": [p for p in model.parameters() if id(p) not in ids], "lr": cfg.lr, "name": "encoder"},
                  {"params": head, "lr": cfg.lr * cfg.head_lr_mult, "name": "head"}]
    else:
        groups = [{"params": list(model.parameters()), "lr": cfg.lr, "name": "all"}]
    check_param_groups(model, groups)
    return torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.wd), [g["lr"] for g in groups]


def cosine_factor(i: int, total: int, warmup: int, min_ratio: float, start_ratio: float = 0.04) -> float:
    """cosineスケジュールの倍率（最大学習率に対する比）。iは0始まりの更新番号（i番目の更新で使う学習率）。
    0〜warmup-1: start_ratioから1へ線形、以後 total-1 まで1→min_ratioのcosine減衰、total以降は最終値を保持。"""
    i = min(i, total - 1)
    if i < warmup:
        return start_ratio + (1 - start_ratio) * i / warmup
    t = (i - warmup) / max(total - 1 - warmup, 1)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * t))


def make_scheduler(opt, max_lr: list[float], total: int, cfg: Config):
    """学習率scheduler。onecycle=従来（OneCycleLR）、cosine=線形warmup＋cosine減衰（確定した新しい基準）。
    どちらも total 更新で最小値に達し、以後は保持する（呼び出し側がlast_epoch<total-1の間だけstepする）。"""
    if cfg.lr_scheduler == "onecycle":
        return torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=max_lr, total_steps=total, pct_start=cfg.pct_start)
    if cfg.lr_scheduler != "cosine":
        raise ValueError(f"unknown lr_scheduler {cfg.lr_scheduler}")
    if not 0 <= cfg.min_lr_ratio < 1:
        raise ValueError("min_lr_ratio must be in [0, 1)")
    warmup = int(cfg.pct_start * total)
    for g, m in zip(opt.param_groups, max_lr):
        g["lr"] = m  # LambdaLRは初期lrを基準に倍率をかける
        g["initial_lr"] = m
    return torch.optim.lr_scheduler.LambdaLR(
        opt, [lambda i: cosine_factor(i, total, warmup, cfg.min_lr_ratio)] * len(opt.param_groups))


def train(stems: list[str], cfg: Config, out: Path, val: list[str] | None = None,
          log=lambda step, d: None, fixed_steps: int | None = None) -> Path:
    """内側検証による早期停止、または固定更新数で学習する。

    更新数の用語: cfg.steps=予定する最大更新数、lr_schedule_steps=OneCycleLRの長さ（0なら cfg.steps）、
    start_step=検証を始める更新数、selection_start=最良値を選ぶ対象にする最初の更新数、
    min_steps=停止を許す最低更新数（それ以前は patience を満たしても止めない）。
    """
    es = {"enabled": False, "monitor": False, "eval_every": 500, "patience": 3, "min_delta": 0.0,
          "start_step": 500, "min_steps": 0, "train_subset": 0, "val_max": 0, "history_every": 50,
          **cfg.early_stopping}
    es.setdefault("selection_start", es["start_step"])
    active = bool(es["enabled"]) and fixed_steps is None
    monitoring = (bool(es["enabled"]) or bool(es["monitor"])) and fixed_steps is None
    if fixed_steps is not None and (fixed_steps < 1 or fixed_steps > cfg.steps):
        raise ValueError("fixed_steps must be within the planned scheduler horizon")
    sched_total = cfg.lr_schedule_steps or cfg.steps
    if not 1 <= sched_total <= cfg.steps:
        raise ValueError("lr_schedule_steps must be within 1..steps")
    if monitoring:
        if not val or len(val) != len(set(val)) or set(stems) & set(val):
            raise ValueError("early stopping requires unique validation IDs disjoint from fit")
        if (es["eval_every"] < 1 or es["patience"] < 1 or not math.isfinite(es["min_delta"])
                or es["min_delta"] < 0 or not 1 <= es["start_step"] <= cfg.steps
                or es["selection_start"] < es["start_step"] or es["min_steps"] < 0):
            raise ValueError("invalid early stopping configuration")
    limit = fixed_steps if fixed_steps is not None else cfg.steps
    seed_all(cfg.seed)
    dev = device()
    model = build(cfg).to(dev)
    opt, max_lr = make_optimizer(model, cfg)
    group_names = [g.get("name", str(i)) for i, g in enumerate(opt.param_groups)]
    sched = make_scheduler(opt, max_lr, sched_total, cfg)
    ema_cfg = ema_settings(cfg)
    emas: dict[float, object] = {}
    last_decay: dict[float, float] = {}
    if ema_cfg:  # 乱数・学習の軌道に触れない（ライブ重みの更新には使わない）。モデル構築・seed_allの後に複製する
        from .ex_b_ema import ModelEMA

        emas = {d: ModelEMA(model, d, ema_cfg["warmup"]) for d in ema_cfg["decays"]}
    t_start = time.perf_counter()
    reset_peak_memory(dev)
    if cfg.workers > 0:  # 長時間学習でworkerとの共有tensorがfdを使い果たす（Too many open files）のを避ける
        torch.multiprocessing.set_sharing_strategy("file_system")
    native = bool(cfg.native)
    if native:
        from .native import META, NativeBatchDataset, batch_summary

        ds = NativeBatchDataset(stems, cfg)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=None, shuffle=False, num_workers=cfg.workers, persistent_workers=cfg.workers > 0,
            prefetch_factor=4 if cfg.workers else None)
    else:
        ds = CropDataset(stems, cfg)
        loader = torch.utils.data.DataLoader(
            ds, batch_size=cfg.batch, shuffle=False, num_workers=cfg.workers, drop_last=True,
            persistent_workers=cfg.workers > 0, prefetch_factor=4 if cfg.workers else None)
    batches = iter(loader)
    stats = {"names": _flag_names(), "counts": np.zeros(len(_flag_names()), np.int64), "crops": 0, "steps": 0}
    metas: list[np.ndarray] = []
    out.parent.mkdir(parents=True, exist_ok=True)
    val_ids = fixed_subset(val, es["val_max"], cfg.seed, 11) if monitoring else []
    train_ids = fixed_subset(stems, es["train_subset"], cfg.seed, 13) if monitoring and es["train_subset"] else []
    run = 0.0
    history = []
    best_loss, best_step, bad_checks = float("inf"), None, 0
    global_best_loss, global_best_step = float("inf"), None
    significant_loss = float("inf")
    stopped_early, stop_reason, eval_seconds, n_evals = False, "max_steps", 0.0, 0
    lr_used = [g["lr"] for g in opt.param_groups]
    # 検証loss・最良チェックポイントに使うモデル。既定はライブ重み。ema.validate_decay 指定時（内側選択のみ）はそのEMA重み
    vmodel = emas[ema_cfg["validate_decay"]].ema if emas and ema_cfg["validate_decay"] is not None else model

    def save_checkpoints() -> None:
        torch.save(vmodel.state_dict(), out)
        for d, e in emas.items():
            torch.save(e.ema.state_dict(), ema_weights_path(out, d))

    for step in range(1, limit + 1):
        model.train()
        batch = next(batches)
        x, y, fl = batch[:3]
        if native:
            metas.append(batch[3].numpy())
        stats["counts"] += fl.sum(0).numpy()
        stats["crops"] += len(x)
        stats["steps"] += 1
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=cfg.amp and dev.type == "cuda"):
            logit = model(x.to(dev))
        loss = loss_fn(logit.float(), y.to(dev), cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        lr_used = [g["lr"] for g in opt.param_groups]  # この更新(step)で実際に使う学習率
        opt.step()
        for d, e in emas.items():
            last_decay[d] = e.update(model)
        if sched.last_epoch < sched_total - 1:  # スケジュール終端以降は最小学習率を保持（last_epoch=total に進めない）
            sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 50 == 0:
            logger.info("step {}/{} loss {:.4f}", step, cfg.steps, run)
            log(step, {"train_loss": run})
        row = {"step": step, "train_loss": run, "train_fixed_loss": "", "validation_loss": "", "lr": lr_used[0],
               **{f"lr_{n}": v for n, v in zip(group_names, lr_used)}}
        if monitoring and step >= es["start_step"] and (step % es["eval_every"] == 0 or step == limit):
            t0 = time.perf_counter()
            value = validation_loss(val_ids, vmodel, cfg, dev)
            row["validation_loss"] = value
            metrics = {"validation_loss": value, "learning_rate": lr_used[0],
                       **{f"learning_rate/{n}": v for n, v in zip(group_names, lr_used)}}
            if train_ids:
                row["train_fixed_loss"] = validation_loss(train_ids, vmodel, cfg, dev)
                metrics["train_fixed_loss"] = row["train_fixed_loss"]
            eval_seconds += time.perf_counter() - t0
            n_evals += 1
            log(step, metrics)
            logger.info("step {} validation loss {:.6f}", step, value)
            if value < global_best_loss:
                global_best_loss, global_best_step = value, step
            if step >= es["selection_start"]:
                if value < best_loss:
                    best_loss, best_step = value, step
                    if active:
                        save_checkpoints()
                if value < significant_loss - es["min_delta"]:
                    significant_loss, bad_checks = value, 0
                else:
                    bad_checks += 1
                if active and bad_checks >= es["patience"] and step >= es["min_steps"]:
                    stopped_early = step < limit
                    stop_reason = "patience"
        if step % es["history_every"] == 0 or row["validation_loss"] != "" or step == limit or stop_reason == "patience":
            history.append(row)
            with out.with_suffix(".history.csv").open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(row))
                writer.writeheader()
                writer.writerows(history)
        if stop_reason == "patience":
            break
        if not monitoring and fixed_steps is None and val and step % cfg.eval_every == 0 and step != cfg.steps:
            s = evaluate(val, [model], cfg)
            logger.info("step {} diag val PQ {:.4f}", step, s.pq)
            log(step, {"val_pq_diag": s.pq})
    if fixed_steps is not None:
        stop_reason = "fixed_steps"
    if not active:
        torch.save(model.state_dict(), out)  # 最終のライブ重み（EMAを使っても従来と同じ）
        for d, e in emas.items():
            torch.save(e.ema.state_dict(), ema_weights_path(out, d))
    elif best_step is None:
        raise RuntimeError("early stopping produced no validation checkpoint")
    summary = {"optimizer_updates": stats["steps"], "crops_presented": stats["crops"],
               "fit_images": len(stems),
               "resources": {"train_seconds": time.perf_counter() - t_start, "device": dev.type,
                             "params_million": sum(p.numel() for p in model.parameters()) / 1e6,
                             "trainable_params_million": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
                             "peak_memory_mb": peak_memory_mb(dev), "output_stride": getattr(model, "output_stride", 1),
                             "input_downscale_factor": cfg.factor, "train_crop": cfg.crop, "batch": cfg.batch,
                             "amp_bf16": bool(cfg.amp and dev.type == "cuda"),
                             "eval_seconds": eval_seconds, "n_evals": n_evals},
               "applied": {n: int(c) for n, c in zip(stats["names"], stats["counts"])},
               "lr": {"groups": group_names, "max_lr": max_lr, "last_lr_used": lr_used,
                      "scheduler_total_steps": sched_total, "planned_max_steps": cfg.steps,
                      "warmup_steps": int(cfg.pct_start * sched_total), "pct_start": cfg.pct_start,
                      "scheduler": cfg.lr_scheduler, "min_lr_ratio": cfg.min_lr_ratio if cfg.lr_scheduler == "cosine" else None,
                      "holds_final_lr_after": sched_total if sched_total < cfg.steps else None},
               "early_stopping": {"enabled": active, "best_step": best_step,
                                  "best_validation_loss": best_loss if monitoring else None,
                                  "global_best_step": global_best_step if monitoring else None,
                                  "global_best_validation_loss": global_best_loss if monitoring else None,
                                  "stopped_early": stopped_early, "stop_reason": stop_reason,
                                  "stopped_step": stats["steps"], "scheduler_total_steps": sched_total,
                                  "requested_fixed_steps": fixed_steps, "settings": es,
                                  "validation_ids": val_ids, "train_monitor_ids": train_ids,
                                  "eval_tiles": eval_tiles(cfg)}}
    if emas:
        summary["ema"] = {**ema_cfg, "n_updates": {ema_tag(d): e.n for d, e in emas.items()},
                          "last_decay_used": {ema_tag(d): last_decay.get(d) for d in emas},
                          "weights": {ema_tag(d): ema_weights_path(out, d).name for d in emas},
                          "validated_with": "ema" if vmodel is not model else "live",
                          "saved_at": "best_step" if active else "final_step",
                          "main_weights_are": "ema(validate_decay)" if vmodel is not model and active else "live"}
    if posch.record(cfg) is not None:
        summary["position_channels"] = posch.record(cfg)
    if native and metas:
        summary["native_crops"] = batch_summary(np.concatenate(metas), cfg.crop)
        summary["native_crops"]["meta_columns"] = list(META)
        np.save(out.with_suffix(".crops.npy"), np.concatenate(metas).astype(np.float32))
    out.with_suffix(".aug.json").write_text(__import__("json").dumps(summary, indent=1))
    logger.info("aug summary {}", summary)
    return out


_GT: dict[str, list[list[dict]]] = {}
GT_CACHE = Path("data/processed/filament-cache/gt_rle.pkl")


def gt_rles(stems: list[str]) -> dict[str, list[list[dict]]]:
    """物理画像stem -> 注釈集合ごとの個体RLE。初回に原寸ポリゴンからRLEを作って保存する。"""
    global _GT
    if not _GT and GT_CACHE.exists():
        _GT = pickle.loads(GT_CACHE.read_bytes())
    missing = [s for s in stems if s not in _GT]
    if missing:
        from .data import load_train

        by_stem = {p.stem: p for p in load_train().values()}
        for s in missing:
            _GT[s] = [encode(a.instance_masks()) for a in by_stem[s].sets]
        GT_CACHE.parent.mkdir(parents=True, exist_ok=True)
        GT_CACHE.write_bytes(pickle.dumps(_GT))
    return {s: _GT[s] for s in stems}


def score_probs(probs: dict[str, np.ndarray], thr: float, min_area: int, min_peak: float = 0.0) -> PQStats:
    """probs: stem -> 原寸確率。注釈集合ごとの統計量を合算したPQ。"""
    gts = gt_rles(list(probs))
    total = PQStats()
    for s, p in probs.items():
        pred = encode(instances(p, thr, min_area, min_peak))
        for g in gts[s]:
            total.add(pq_stats(g, pred))
    return total


def score_grid(probs, thrs, min_areas) -> dict[tuple[float, int], PQStats]:
    """probsは.items()で原寸確率を1枚ずつ返す辞書。成分ごとのRLEをしきい値ごとに一度だけ作り、最小面積は絞り込みだけで評価する。"""
    from scipy import ndimage as ndi

    from .postprocess import EIGHT

    gts = gt_rles(list(probs))
    out = {(t, a): PQStats() for t in thrs for a in min_areas}
    for s, p in probs.items():
        for t in thrs:
            lab, n = ndi.label(p > t, structure=EIGHT)
            idx = np.arange(1, n + 1)
            area = ndi.sum(np.ones_like(lab), lab, idx) if n else np.array([])
            rle = {}
            for a in min_areas:
                keep = [int(i) for i, ar in zip(idx, area) if ar >= a]
                for i in keep:
                    if i not in rle:
                        rle[i] = encode((lab == i)[None])[0]
                pred = [rle[i] for i in keep]
                for g in gts[s]:
                    out[(t, a)].add(pq_stats(g, pred))
    return out


def score_grid_peak(probs, thrs, min_areas, min_peaks=(0.0,)) -> dict[tuple[float, int, float], PQStats]:
    """score_grid に成分の最大確率の下限(min_peak)を加えた版。キーは (thr, min_area, min_peak)。
    probsは.items()で原寸確率を1枚ずつ返す辞書。成分ごとのRLEは (画像, thr) で一度だけ作る。"""
    from scipy import ndimage as ndi

    from .postprocess import EIGHT

    gts = gt_rles(list(probs))
    out = {(t, a, q): PQStats() for t in thrs for a in min_areas for q in min_peaks}
    for s, p in probs.items():
        for t in thrs:
            lab, n = ndi.label(p > t, structure=EIGHT)
            idx = np.arange(1, n + 1)
            area = ndi.sum(np.ones_like(lab), lab, idx) if n else np.array([])
            peak = ndi.maximum(p, lab, idx) if n else np.array([])
            rle = {}
            for a in min_areas:
                for q in min_peaks:
                    keep = [int(i) for i, ar, pk in zip(idx, area, peak) if ar >= a and pk >= q]
                    for i in keep:
                        if i not in rle:
                            rle[i] = encode((lab == i)[None])[0]
                    pred = [rle[i] for i in keep]
                    for g in gts[s]:
                        out[(t, a, q)].add(pq_stats(g, pred))
    return out


def evaluate(stems: list[str], models: list[UNet], cfg: Config) -> PQStats:
    dev = device()
    probs = {}
    for s in stems:
        img, _ = cache_mod.load_cfg(s, cfg)
        probs[s] = upsample(predict(models, img, dev, cfg))
    return score_probs(probs, cfg.thr, cfg.min_area)


def cfg_dict(cfg: Config) -> dict:
    return asdict(cfg)
