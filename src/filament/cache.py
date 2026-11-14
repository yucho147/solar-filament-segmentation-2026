"""学習用キャッシュ。画像を 1/scale に縮小したuint8と、注釈集合平均のソフト教師を保存する。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from loguru import logger

from .data import ROOT, SIZE, PhysicalImage, load_train, read_gray

CACHE = Path("data/processed/filament-cache")


def downscale(arr: np.ndarray, factor: int) -> np.ndarray:
    if factor == 1:
        return arr
    h = arr.shape[0] // factor
    return np.asarray(Image.fromarray(arr).resize((h, h), Image.Resampling.BOX))


def build(factor: int = 2, root: Path = ROOT, out: Path = CACHE, limit: int | None = None,
          stems: set[str] | None = None, with_sets: bool = True) -> Path:
    """stems を指定すればその画像だけ、with_sets=False なら注釈集合ごとの教師（sets）を保存しない（原寸学習は平均教師だけ使う）。
    既存ファイルは、必要な配列が揃っていれば作り直さない。"""
    images = load_train(root)
    out = out / f"x{factor}"
    out.mkdir(parents=True, exist_ok=True)
    for n, (name, p) in enumerate(sorted(images.items())):
        if limit and n >= limit:
            break
        if stems is not None and p.stem not in stems:
            continue
        dst = out / f"{p.stem}.npz"
        if dst.exists() and (not with_sets or "sets" in np.load(dst).files):
            continue
        img = read_gray(root / "train/train_images" / name)
        per_set = []
        for s in p.sets:
            m = s.instance_masks()
            per_set.append(m.max(0) if len(m) else np.zeros((SIZE, SIZE), np.uint8))
        soft = np.mean(per_set, 0)
        arrays = {"image": downscale(img, factor), "target": downscale((soft * 255).astype(np.uint8), factor)}
        if with_sets:
            arrays["sets"] = np.stack([downscale((x * 255).astype(np.uint8), factor) for x in per_set])
        np.savez(dst, **arrays)
        if n % 100 == 0:
            logger.info("cached {}/{}", n, len(images))
    return out


def verify_native(stem: str, root: Path = ROOT, out: Path = CACHE) -> dict:
    """factor=1 のキャッシュが元画像・元ポリゴンと画素単位で一致することを検査する（不一致は例外）。
    - 画像: JPEGをデコードした値と一致、形状2048×2048、dtype uint8
    - 教師: 保存済みの注釈集合ごとの個体RLE（gt_rles、原寸ポリゴンから生成）の和集合を平均した教師と一致"""
    from .engine import gt_rles
    from pycocotools import mask as mu

    by_stem = {p.stem: p for p in load_train(root).values()}
    p = by_stem[stem]
    img, tgt = load(stem, 1, out)
    if img.shape != (SIZE, SIZE) or tgt.shape != (SIZE, SIZE) or img.dtype != np.uint8 or tgt.dtype != np.uint8:
        raise AssertionError(f"{stem}: cached arrays are not native 2048x2048 uint8")
    raw = read_gray(root / "train/train_images" / p.file_name)
    if not np.array_equal(raw, img):
        raise AssertionError(f"{stem}: cached image differs from decoded source image")
    per_set = []
    for rles in gt_rles([stem])[stem]:
        m = mu.decode(rles) if rles else np.zeros((SIZE, SIZE, 0), np.uint8)
        per_set.append(m.max(2) if m.shape[2] else np.zeros((SIZE, SIZE), np.uint8))
    expect = (np.mean(per_set, 0) * 255).astype(np.uint8)
    if not np.array_equal(expect, tgt):
        raise AssertionError(f"{stem}: cached target differs from the native polygon rasterization")
    return {"stem": stem, "n_sets": len(per_set), "fg_pixels": int((tgt > 127).sum())}


def load(stem: str, factor: int = 2, out: Path = CACHE) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(out / f"x{factor}" / f"{stem}.npz")
    return z["image"], z["target"]


def load_sets(stem: str, factor: int = 2, out: Path = CACHE) -> tuple[np.ndarray, np.ndarray]:
    """画像と、注釈集合ごとの教師uint8[S,H,W]。"""
    z = np.load(out / f"x{factor}" / f"{stem}.npz")
    return z["image"], z["sets"]


def _image_cache_name(cfg) -> str | None:
    """Config.extra["image_cache"]（例 "x1n" = 物理正規化した原寸画像）。未指定なら None（従来動作）。"""
    name = (getattr(cfg, "extra", None) or {}).get("image_cache")
    if not name:
        return None
    if cfg.factor != 1 or name != "x1n":
        raise ValueError(f"image_cache={name!r} は factor=1 の 'x1n' のみ対応（factor={cfg.factor}）")
    return name


def load_cfg(stem: str, cfg, out: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """画像と教師を読む。cfg.extra["image_cache"] があれば画像だけをそのキャッシュから、教師は従来の x{factor} から読む。"""
    name = _image_cache_name(cfg)
    if name is None:
        return load(stem, cfg.factor) if out is None else load(stem, cfg.factor, out)
    return np.load((out or CACHE) / name / f"{stem}.npz")["image"], (load(stem, cfg.factor) if out is None else load(stem, cfg.factor, out))[1]


def load_sets_cfg(stem: str, cfg, out: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    name = _image_cache_name(cfg)
    if name is None:
        return load_sets(stem, cfg.factor) if out is None else load_sets(stem, cfg.factor, out)
    return np.load((out or CACHE) / name / f"{stem}.npz")["image"], (load_sets(stem, cfg.factor) if out is None else load_sets(stem, cfg.factor, out))[1]
