"""exp025: 固定した候補を同じ確率で注釈集合ごとに評価する。"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils

from .analysis import BIN_NAMES, detail
from .metrics import PQStats, encode, pq_stats
from .postprocess import instances


def candidates(baseline: dict) -> list[dict]:
    """既存対照の画素閾値を固定し、事前に決めた4条件を返す。"""
    return [{"thr": baseline["thr"], "min_area": area, "min_peak": peak}
            for area, peak in ((100, 0.95), (100, 0.99), (200, 0.95), (200, 0.99))]


def score(probs, gts: dict, posts: list[dict], blocks: dict, times: dict) -> list[dict]:
    """画像と注釈集合ごとの十分統計量、面積別統計、注釈間の対応を返す。"""
    if set(probs) != set(gts):
        raise ValueError("probability and ground-truth image IDs differ")
    summaries = [{"post": post, "rows": [], "sets": [],
                  "annotation_matches": {"all": 0, "some": 0, "none": 0}} for post in posts]
    for stem, prob in probs.items():
        for post, out in zip(posts, summaries):
            pred = encode(instances(prob, **post))
            total = PQStats()
            hit_counts = np.zeros(len(pred), dtype=int)
            for set_index, gt in enumerate(gts[stem]):
                stat = pq_stats(gt, pred)
                total.add(stat)
                d = detail(gt, pred)
                if gt and pred:
                    hit_counts += (np.asarray(mask_utils.iou(pred, gt, [0] * len(gt))) > 0.5).any(axis=1)
                out["sets"].append({"stem": stem, "set_index": set_index,
                                    "iou_sum": stat.iou_sum, "tp": stat.tp, "fp": stat.fp, "fn": stat.fn,
                                    "n_gt": d["n_gt"].tolist(), "matched": d["matched"].tolist(),
                                    "fp_bins": d["fp_bins"].tolist()})
            n_sets = len(gts[stem])
            if n_sets > 1:
                out["annotation_matches"]["all"] += int((hit_counts == n_sets).sum())
                out["annotation_matches"]["some"] += int(((hit_counts > 0) & (hit_counts < n_sets)).sum())
                out["annotation_matches"]["none"] += int((hit_counts == 0).sum())
            out["rows"].append({"stem": stem, "block": int(blocks[stem]), "year": times[stem].year,
                                "site": stem[-2:], "n_sets": n_sets,
                                "n_gt": sum(len(g) for g in gts[stem]), "n_pred": len(pred),
                                "iou_sum": total.iou_sum, "tp": total.tp, "fp": total.fp, "fn": total.fn})
    for out in summaries:
        total = PQStats()
        for row in out["rows"]:
            total.add(PQStats(row["iou_sum"], row["tp"], row["fp"], row["fn"]))
        out.update({"pq": total.pq, "tp": total.tp, "fp": total.fp, "fn": total.fn})
        out["by_area"] = {name: {key: sum(row[key][i] for row in out["sets"])
                                for key in ("n_gt", "matched", "fp_bins")}
                          for i, name in enumerate(BIN_NAMES)}
    return summaries


def choose(summaries: list[dict]) -> int:
    """事前条件を満たす最大PQの候補の添字。なければ対照0。"""
    baseline = summaries[0]
    small = baseline["by_area"]["100-400"]
    if not small["n_gt"]:
        return 0
    eligible = [i for i, candidate in enumerate(summaries[1:], 1)
                if candidate["by_area"]["100-400"]["matched"] >= 1.05 * small["matched"]
                and candidate["by_area"]["100-400"]["matched"] > small["matched"]
                and candidate["fp"] <= baseline["fp"] and candidate["pq"] >= baseline["pq"]]
    return min(eligible, key=lambda i: (-summaries[i]["pq"], summaries[i]["fp"], i)) if eligible else 0
