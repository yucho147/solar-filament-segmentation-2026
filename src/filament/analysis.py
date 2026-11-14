"""誤り分析: 面積別の検出率、見逃し/誤検出、一本の分断（split）と複数本の誤結合（merge）、画像ごとのPQ。"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

from .metrics import PQStats, encode, pq_stats
from .postprocess import instances

AREA_BINS = (0, 100, 400, 1600, 6400, 10**9)  # <100, 100-400, 400-1600, 1600-6400, >6400 画素（原寸）
BIN_NAMES = ("<100", "100-400", "400-1600", "1600-6400", ">6400")


def detail(gt: list[dict], pred: list[dict]) -> dict:
    """1注釈集合に対する詳細。

    matched: IoU>0.5で対応した正解の数（面積別）。split: IoU>0.5の対応がなく、予測2個以上が正解の大半を占めて分断された正解。
    merge: 正解2個以上の大半を覆う予測（誤結合）。fp_area: 対応のない予測の面積別個数。
    """
    ga = np.asarray(mask_utils.area(gt)) if gt else np.zeros(0)
    pa = np.asarray(mask_utils.area(pred)) if pred else np.zeros(0)
    gb = np.digitize(ga, AREA_BINS[1:-1])
    pb = np.digitize(pa, AREA_BINS[1:-1])
    n_gt, matched, fn = np.zeros(5, int), np.zeros(5, int), np.zeros(5, int)
    fp_bins = np.zeros(5, int)
    split_gt = merge_pred = 0
    if gt:
        for b in gb:
            n_gt[b] += 1
    if gt and pred:
        iou = np.asarray(mask_utils.iou(pred, gt, [0] * len(gt)))
        inter = np.asarray(mask_utils.iou(pred, gt, [1] * len(gt))) * pa[:, None]  # crowd: inter/area_pred
        hit = iou > 0.5
        for j, b in enumerate(gb):
            if hit[:, j].any():
                matched[b] += 1
            else:
                fn[b] += 1
                cover = inter[:, j] / max(ga[j], 1)
                if (inter[:, j] / np.maximum(pa, 1) > 0.5).sum() >= 2 and cover.sum() > 0.5:
                    split_gt += 1
        for i, b in enumerate(pb):
            if not hit[i].any():
                fp_bins[b] += 1
        covered = (inter / np.maximum(ga, 1)[None, :]) > 0.5  # 予測iが正解jの大半を覆う
        merge_pred = int((covered.sum(1) >= 2).sum())
    elif gt:
        for b in gb:
            fn[b] += 1
    elif pred:
        for b in pb:
            fp_bins[b] += 1
    return {"n_gt": n_gt, "matched": matched, "fn": fn, "fp_bins": fp_bins, "split_gt": split_gt, "merge_pred": merge_pred}


def analyze(probs, gts, post: dict) -> dict:
    """probs: stem->原寸確率（.items()で1枚ずつ）。gts: stem->注釈集合ごとのRLE。postは instances の引数。

    注釈集合ごとに集計（公開評価と同じ単位）。画像ごとのPQ統計も返す。
    """
    tot = {k: np.zeros(5, int) for k in ("n_gt", "matched", "fn", "fp_bins")}
    split = merge = 0
    pq = PQStats()
    per_image = {}
    for s, p in probs.items():
        pred = encode(instances(p, **post))
        ps = PQStats()
        for g in gts[s]:
            st = pq_stats(g, pred)
            pq.add(st)
            ps.add(st)
            d = detail(g, pred)
            for k in tot:
                tot[k] += d[k]
            split += d["split_gt"]
            merge += d["merge_pred"]
        per_image[s] = {"pq": ps.pq, "tp": ps.tp, "fp": ps.fp, "fn": ps.fn, "n_pred": len(pred)}
    return {
        "post": post, "pq": pq.pq, "sq": pq.sq, "rq": pq.rq, "tp": pq.tp, "fp": pq.fp, "fn": pq.fn,
        "by_area": {n: {"gt": int(tot["n_gt"][i]), "detected": int(tot["matched"][i]),
                        "detection_rate": float(tot["matched"][i] / tot["n_gt"][i]) if tot["n_gt"][i] else None,
                        "false_positive_pred": int(tot["fp_bins"][i])} for i, n in enumerate(BIN_NAMES)},
        "split_gt": split, "merge_pred": merge, "per_image": per_image,
    }
