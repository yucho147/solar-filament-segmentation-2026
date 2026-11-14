"""原寸（2048×2048）のまま512×512の切片で学習するためのデータ供給と検査。

- 入力画像・教師は縮小・拡大・補間をしない（cache の factor=1 を使う）。
- 1バッチ = 同じ画像から重なりのある複数切片（基準切片＋近傍切片）＋ほかの画像の切片（前景・円盤内背景・一様）。
- 各切片に独立な補間なしの拡張（90度回転・反転、画像のみの弱い明度/コントラスト）を、画像と教師で同期して適用する。
- 乱数は (seed, バッチ番号) だけで決まり、worker数に依らない。切片の座標・拡張の値は meta に返す。
"""
from __future__ import annotations

import numpy as np
import torch

from . import posch
from .engine import Config, CropDataset, _flag_names

DEFAULT = {
    "same_image_crops": 3,  # 1バッチ内の同一画像の切片数（基準1＋近傍 same_image_crops-1）
    "overlap": [0.25, 0.75],  # 近傍切片と基準切片の重なり面積率（画素面積の比）の目標範囲
    "max_retry": 32,
    "min_fg_px": 64,  # 前景モードの切片・近傍切片が含むべき正解画素数の下限（満たす切片を再抽選で優先）
    "fg_retry": 16,
}
META = ("image", "y0", "x0", "role", "mode", "k", "flip", "contrast", "brightness", "overlap", "fg_frac", "edge_cut")
ROLE = {"anchor": 0, "neighbor": 1, "other": 2}
MODE = {"fg": 0, "disc": 1, "bg": 2, "uniform": 3}
ANCHOR_SAMPLING = {"fg": 1.0, "disc": 0.0, "bg": 0.0, "min_disc": 0.0}
ALLOWED_AUG = {"d4", "bc"}  # 初期条件: 補間のない幾何変換と画像だけの明度/コントラスト


def parse(native: dict, batch: int) -> dict:
    spec = {**DEFAULT, **native}
    lo, hi = spec["overlap"]
    if not (0 < lo <= hi < 1):
        raise ValueError("overlap range must satisfy 0 < lo <= hi < 1")
    if not 1 <= spec["same_image_crops"] <= batch:
        raise ValueError("same_image_crops must be within 1..batch")
    return spec


def overlap_ratio(y0: int, x0: int, y1: int, x1: int, w: int) -> float:
    return max(0, w - abs(y0 - y1)) * max(0, w - abs(x0 - x1)) / (w * w)


def sample_neighbor(rng: np.random.Generator, y0: int, x0: int, w: int, h: int, lo: float, hi: float,
                    max_retry: int = 32) -> tuple[int, int, float]:
    """基準切片(y0,x0)に対し、重なり面積率が[lo,hi]の近傍切片の左上と実際の重なり率を返す。
    画像端で窓が切り詰められて範囲外になる場合は再抽選し、尽きたら範囲内に最も近い案を返す。"""
    best = None
    for _ in range(max_retry):
        r = rng.uniform(lo, hi)
        u = rng.uniform(r, 1.0)  # (w-|dy|)/w
        v = r / u  # (w-|dx|)/w
        dy, dx = round(w * (1 - u)), round(w * (1 - v))
        if rng.random() < 0.5:
            dy, dx = dx, dy
        dy, dx = dy * (1 if rng.random() < 0.5 else -1), dx * (1 if rng.random() < 0.5 else -1)
        ny, nx = int(np.clip(y0 + dy, 0, h - w)), int(np.clip(x0 + dx, 0, h - w))
        ov = overlap_ratio(y0, x0, ny, nx, w)
        if (ny, nx) != (y0, x0) and lo <= ov <= hi:
            return ny, nx, ov
        gap = 0.0 if lo <= ov <= hi else min(abs(ov - lo), abs(ov - hi))
        if best is None or gap < best[0]:
            best = (gap, ny, nx, ov)
    return best[1], best[2], best[3]


def apply_aug(img: np.ndarray, tgt: np.ndarray, rng: np.random.Generator, spec: dict) -> tuple:
    """切片(uint8, float32)に d4（回転k×90度＋左右反転）と bc（画像のみ）を適用。値は返して記録する。"""
    k, flip, contrast, brightness = 0, 0, 1.0, 0.0
    if "d4" in spec and rng.random() < spec["d4"]["p"]:
        k = int(rng.integers(4))
        flip = int(rng.random() < 0.5)
    if "bc" in spec and rng.random() < spec["bc"]["p"]:
        contrast = float(rng.uniform(*spec["bc"]["contrast"]))
        brightness = float(rng.uniform(*spec["bc"]["brightness"]))
    return (*forward(img, tgt, k, flip, contrast, brightness), k, flip, contrast, brightness)


def forward(img: np.ndarray, tgt: np.ndarray, k: int, flip: int, contrast: float, brightness: float):
    """記録した値から決定的に変換を再現する（検査でも使う）。"""
    if k:
        img, tgt = np.rot90(img, k), np.rot90(tgt, k)
    if flip:
        img, tgt = img[:, ::-1], tgt[:, ::-1]
    if contrast != 1.0 or brightness != 0.0:
        x = img.astype(np.float32) / 255
        x = (x - 0.5) * contrast + 0.5 + brightness
        img = np.clip(x * 255 + 0.5, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img, dtype=np.uint8), np.ascontiguousarray(tgt, dtype=np.float32)


class NativeBatchDataset(CropDataset):
    """バッチ番号 b → (x[B,1,w,w], y[B,1,w,w], flags[B,F], meta[B,len(META)])。cfg.steps 個のバッチ。
    cfg.sampling は「ほかの画像の切片」の選び方（augment.SAMPLING のキー）。基準切片は前景を含む切片。"""

    def __init__(self, stems: list[str], cfg: Config, length: int | None = None):
        if cfg.factor != 1:
            raise ValueError("native crops require factor=1 (no resize of image or target)")
        if cfg.target_mode != "mean":
            raise ValueError("native crops currently require target_mode=mean")
        if cfg.mix:
            raise ValueError("native crops do not support mix augmentations")
        super().__init__(stems, cfg, length)
        self.nspec = parse(cfg.native, cfg.batch)
        bad = set(self.spec) - ALLOWED_AUG
        if bad:
            raise ValueError(f"native augmentations are limited to {sorted(ALLOWED_AUG)}; got {sorted(bad)}")
        self.w = cfg.crop
        if self.pipe.win != self.w:
            raise ValueError("native augmentation must not enlarge the crop window")
        self.length = length or cfg.steps

    def _other_window(self, rng, j: int):
        w = self.w
        for _ in range(8):  # 円盤が少ししか入らない窓は捨てる（従来のCropDatasetと同じ）
            y0, x0, mode = self.window(rng, j, w)
            if mode in ("fg", "bg") or self.disc is None or self.disc[j].mask[y0:y0 + w, x0:x0 + w].mean() >= self.samp["min_disc"]:
                break
        return y0, x0, mode

    def _fg_px(self, i: int, y0: int, x0: int) -> int:
        return int((self.data[i][1][y0:y0 + self.w, x0:x0 + self.w] > 127).sum())

    def plan(self, b: int) -> list[dict]:
        """バッチbの切片計画（画像番号・座標・役割・重なり）。画素は読まない。"""
        rng = np.random.default_rng([self.cfg.seed, b])
        B, n_same = self.cfg.batch, self.nspec["same_image_crops"]
        n_img, w = len(self.data), self.w
        pick = rng.choice(n_img, size=B - n_same + 1, replace=(B - n_same + 1) > n_img)
        a = int(pick[0])
        h = self.data[a][0].shape[0]
        for _ in range(self.nspec["fg_retry"]):
            y0, x0, mode = self.window(rng, a, w, ANCHOR_SAMPLING)
            if self._fg_px(a, y0, x0) >= self.nspec["min_fg_px"]:
                break
        out = [dict(image=a, y0=y0, x0=x0, role="anchor", mode=mode, overlap=1.0)]
        lo, hi = self.nspec["overlap"]
        for _ in range(n_same - 1):
            for _ in range(self.nspec["fg_retry"]):  # 重なり範囲を守りつつ、前景を含む近傍切片を優先
                ny, nx, ov = sample_neighbor(rng, y0, x0, w, h, lo, hi, self.nspec["max_retry"])
                if self._fg_px(a, ny, nx) >= self.nspec["min_fg_px"]:
                    break
            out.append(dict(image=a, y0=ny, x0=nx, role="neighbor", mode=mode, overlap=ov))
        for j in pick[1:]:
            y, x, m = self._other_window(rng, int(j))
            for _ in range(self.nspec["fg_retry"] if m == "fg" else 0):
                if self._fg_px(int(j), y, x) >= self.nspec["min_fg_px"]:
                    break
                y, x, m = self._other_window(rng, int(j))
                if m != "fg":
                    break
            out.append(dict(image=int(j), y0=y, x0=x, role="other", mode=m, overlap=0.0))
        return out

    def get_batch(self, b: int):
        plan = self.plan(b)
        w = self.w
        xs, ys, flags, metas, ps = [], [], [], [], []
        for j, c in enumerate(plan):
            img, tgt = self.data[c["image"]]
            x = img[c["y0"]:c["y0"] + w, c["x0"]:c["x0"] + w]
            y = tgt[c["y0"]:c["y0"] + w, c["x0"]:c["x0"] + w].astype(np.float32) / 255
            crng = np.random.default_rng([self.cfg.seed, b, j, 1])
            xa, ya, k, flip, contrast, brightness = apply_aug(x, y, crng, self.spec)
            border = np.concatenate([y[0], y[-1], y[:, 0], y[:, -1]])
            fl = dict.fromkeys(_flag_names(), 0)
            fl["d4"], fl["bc"] = int(k > 0 or flip > 0), int(contrast != 1.0 or brightness != 0.0)
            fl[f"crop_{c['mode']}"] = 1
            xs.append(xa)
            pm = posch.full_map(self.cfg, img.shape)  # 位置チャネル（無効ならNone）。切片切出し前の全体座標から作り、画像・教師と同じD4変換
            if pm is not None:
                ps.append(posch.crop_d4(pm, c["y0"], c["x0"], w, k, flip))
            ys.append(ya)
            flags.append([fl[n] for n in _flag_names()])
            metas.append([c["image"], c["y0"], c["x0"], ROLE[c["role"]], MODE[c["mode"]], k, flip, contrast, brightness,
                          c["overlap"], float((y > 0.5).mean()), float((border > 0.5).any())])
        x = torch.from_numpy(np.stack(xs).astype(np.float32) / 255)[:, None]
        if ps:
            x = torch.cat([x, torch.from_numpy(np.stack(ps))], 1)
        return (x, torch.from_numpy(np.stack(ys))[:, None],
                torch.tensor(flags, dtype=torch.int32), torch.tensor(metas, dtype=torch.float64))

    def __getitem__(self, b: int):
        return self.get_batch(b)


def window_starts(n: int, tile: int, stride: int) -> list[int]:
    xs = list(range(0, n - tile + 1, stride))
    if xs[-1] != n - tile:
        xs.append(n - tile)
    return xs


def coverage(h: int, w: int, tile: int, stride: int) -> np.ndarray:
    """各画素を覆う窓の数。1以上でなければ被覆漏れ。"""
    cnt = np.zeros((h, w), np.int32)
    for y in window_starts(h, tile, stride):
        for x in window_starts(w, tile, stride):
            cnt[y:y + tile, x:x + tile] += 1
    return cnt


def batch_summary(meta: np.ndarray, w: int) -> dict:
    """学習全体の切片統計（meta は [N,len(META)]）。設定・実測値の記録用。"""
    m = {n: meta[:, i] for i, n in enumerate(META)}
    return {"crops": int(len(meta)), "role_counts": {r: int((m["role"] == v).sum()) for r, v in ROLE.items()},
            "mode_counts": {r: int((m["mode"] == v).sum()) for r, v in MODE.items()},
            "contains_foreground_frac": float((m["fg_frac"] > 0).mean()),
            "background_only_frac": float((m["fg_frac"] == 0).mean()),
            "edge_cut_frac": float(m["edge_cut"].mean()),
            "neighbor_overlap_mean": float(m["overlap"][m["role"] == 1].mean()) if (m["role"] == 1).any() else None,
            "neighbor_overlap_min": float(m["overlap"][m["role"] == 1].min()) if (m["role"] == 1).any() else None,
            "neighbor_overlap_max": float(m["overlap"][m["role"] == 1].max()) if (m["role"] == 1).any() else None,
            "unique_images": int(len(np.unique(m["image"]))), "crop": w}
