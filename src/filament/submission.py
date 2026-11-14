"""提出CSV（filament_id, segmentation_rle）。圧縮RLEのcounts文字列のみ。"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

from .postprocess import check_disjoint


def rle_string(mask: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def write(rows: dict[str, np.ndarray], path: Path, test_ids: list[str]) -> int:
    """全画像分を辞書で受け取る版（小規模・テスト用）。"""
    missing = set(test_ids) - set(rows)
    if missing:
        raise ValueError(f"{len(missing)} test images not processed")
    return write_stream(((s, rows[s]) for s in test_ids), path, test_ids)


def write_stream(items, path: Path, test_ids: list[str]) -> int:
    """(stem, bool[N,2048,2048]) を1画像ずつ書く。全テスト画像が一度ずつ処理されることを要求する。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    n = 0
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filament_id", "segmentation_rle"])
        for stem, m in items:
            if stem in done or stem not in test_ids:
                raise ValueError(f"duplicate/unknown image {stem}")
            done.add(stem)
            if m.ndim != 3 or m.shape[1:] != (2048, 2048) or m.dtype != bool:
                raise ValueError(f"{stem}: masks must be bool[N,2048,2048], got {m.dtype}{m.shape}")
            if len(m) and (m.reshape(len(m), -1).sum(1) == 0).any():
                raise ValueError(f"{stem}: empty mask")
            check_disjoint(m)
            for k, one in enumerate(m, 1):
                w.writerow([f"{stem}_{k}", rle_string(one)])
                n += 1
    if done != set(test_ids):
        raise ValueError(f"{len(set(test_ids) - done)} test images not processed")
    return n


def verify(path: Path, test_ids: list[str]) -> dict:
    """読み戻し検査: 列名・ID一意・原寸デコード・二値・テスト画像ID対応。"""
    seen, stems = set(), set()
    cur, acc, closed = None, None, set()
    with path.open() as f:
        r = csv.DictReader(f)
        assert r.fieldnames == ["filament_id", "segmentation_rle"], r.fieldnames
        for row in r:
            fid = row["filament_id"]
            assert fid not in seen, f"dup {fid}"
            seen.add(fid)
            stem = fid.rsplit("_", 1)[0]
            assert stem in test_ids, f"unknown image {fid}"
            stems.add(stem)
            m = mask_utils.decode({"size": [2048, 2048], "counts": row["segmentation_rle"].encode()})
            assert m.shape == (2048, 2048) and m.sum() > 0, f"empty/bad mask {fid}"
            if stem != cur:  # CSVは画像ごとに連続しているので、1画像分のcoverageだけ保持する
                assert stem not in closed, f"{stem} rows are not contiguous"
                closed.add(cur)
                cur, acc = stem, np.zeros((2048, 2048), np.uint8)
            assert not (acc & m).any(), f"overlapping masks in {stem}"
            acc |= m
    return {"rows": len(seen), "images_with_rows": len(stems), "images": len(test_ids)}
