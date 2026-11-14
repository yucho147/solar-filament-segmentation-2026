"""公開 Self Evaluation Notebook と同じ規則の PQ（IoU>0.5 を対応、集合単位の合算）。"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pycocotools import mask as mask_utils


@dataclass
class PQStats:
    iou_sum: float = 0.0
    tp: int = 0
    fp: int = 0
    fn: int = 0

    def add(self, other: "PQStats") -> None:
        self.iou_sum += other.iou_sum
        self.tp += other.tp
        self.fp += other.fp
        self.fn += other.fn

    @property
    def pq(self) -> float:
        den = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.iou_sum / den if den > 0 else 0.0

    @property
    def sq(self) -> float:
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def rq(self) -> float:
        den = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / den if den > 0 else 0.0


def encode(masks: np.ndarray) -> list[dict]:
    """bool/uint8[N,H,W] -> COCO RLE（列優先）。"""
    return [mask_utils.encode(np.asfortranarray(m.astype(np.uint8))) for m in masks]


def pq_stats(gt: list[dict], pred: list[dict]) -> PQStats:
    """一つの注釈集合に対する予測の統計量。公開コード同様、IoU>0.5 の要素を全てTPとして数える。"""
    if not gt and not pred:
        return PQStats()
    if not gt:
        return PQStats(fp=len(pred))
    if not pred:
        return PQStats(fn=len(gt))
    iou = np.asarray(mask_utils.iou(pred, gt, [0] * len(gt)))  # [P,G]
    hit = iou > 0.5
    # 公開Notebook: TPのIoUは0.5超の要素の合計、FPは対応のない予測列、FNは対応のない正解
    return PQStats(
        iou_sum=float(iou[hit].sum()),
        tp=int(hit.sum()),
        fp=int((~hit.any(axis=1)).sum()),
        fn=int((~hit.any(axis=0)).sum()),
    )
