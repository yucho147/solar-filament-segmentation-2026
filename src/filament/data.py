"""公式配布データの読み取りと原寸マスク生成。欠落は例外にする（合成データへ切り替えない）。"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

SIZE = 2048
ROOT = Path("data/raw/filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")


@dataclass
class AnnotationSet:
    set_id: str  # images[].id（注釈集合ID。観測画像IDではない）
    polygons: list[list[float]] = field(default_factory=list)  # 1個体=1ポリゴン

    def instance_masks(self) -> np.ndarray:
        """uint8[N,2048,2048]。個体ごとの原寸マスク。"""
        if not self.polygons:
            return np.zeros((0, SIZE, SIZE), np.uint8)
        rles = [mask_utils.merge(mask_utils.frPyObjects([p], SIZE, SIZE)) for p in self.polygons]
        return np.ascontiguousarray(mask_utils.decode(rles).transpose(2, 0, 1))


@dataclass
class PhysicalImage:
    file_name: str
    captured: datetime
    sets: list[AnnotationSet] = field(default_factory=list)

    @property
    def stem(self) -> str:
        return Path(self.file_name).stem

    @property
    def site(self) -> str:
        return self.stem[-2:]


def load_train(root: Path = ROOT, check_images: bool = True) -> dict[str, PhysicalImage]:
    """物理画像(file_name)をキーに、同じ画像の全注釈集合をまとめて返す。

    check_images=False は、画像そのものを読まない処理（撮影時刻・注釈集合の参照）だけで、GPU機へ元画像を送らずに済ませるための指定。
    画素を読む処理（キャッシュ生成など）は既定の True のまま、欠落を例外にする。
    """
    meta = json.loads((root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").read_text())
    sets: dict[str, AnnotationSet] = {}
    by_file: dict[str, PhysicalImage] = {}
    for im in meta["images"]:
        if (im["height"], im["width"]) != (SIZE, SIZE):
            raise ValueError(f"unexpected size {im['file_name']}")
        phys = by_file.setdefault(
            im["file_name"],
            PhysicalImage(im["file_name"], datetime.strptime(im["date_captured"], "%Y-%m-%d %H:%M:%S")),
        )
        s = AnnotationSet(str(im["id"]))
        sets[s.set_id] = s
        phys.sets.append(s)
    for a in meta["annotations"]:
        if str(a["image_id"]) not in sets:
            raise KeyError(f"annotation refers to missing set {a['image_id']}")
        seg = a["segmentation"]
        if not isinstance(seg, list) or len(seg) != 1:
            raise ValueError("expected a single polygon per annotation")
        sets[str(a["image_id"])].polygons.append(seg[0])
    for p in by_file.values() if check_images else ():
        if not (root / "train/train_images" / p.file_name).exists():
            raise FileNotFoundError(p.file_name)
    return by_file


def list_test_images(root: Path = ROOT) -> list[Path]:
    files = sorted((root / "test/test_images").glob("*.jpeg"))
    if not files:
        raise FileNotFoundError("no test images")
    return files


def read_gray(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("L"))
    if arr.shape != (SIZE, SIZE):
        raise ValueError(f"{path}: shape {arr.shape}")
    return arr


def group_sets_by_file(images: dict[str, PhysicalImage]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for k, v in images.items():
        out[k] = len(v.sets)
    return dict(out)
