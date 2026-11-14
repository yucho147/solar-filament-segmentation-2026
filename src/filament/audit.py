"""既存 split・学習済みモデルの manifest・保存済み予測の対応を検査する。

検査は「記録の整合性」の確認で、学習過程そのものの証明ではない（manifest は学習コードの自己申告）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .nested import LeakageError, check_clean_evaluation, min_gap_days
from .provenance import ids_hash, sha256_file
from .split import PURGE_DAYS


def audit_split(split: dict, times: dict, sets_per_stem: dict[str, list[str]]) -> dict:
    """split の被覆・重複・時間分離・複数注釈の同居を検査する。違反は例外ではなく結果に載せる。"""
    res: dict = {"hash": split["hash"], "name": split["name"]}
    part = {"holdout": split["holdout"], "buffer": split["holdout_buffer"], "development": split["development"]}
    allids = [i for v in part.values() for i in v]
    res["coverage"] = {"n_images": len(times), "n_in_split": len(allids), "duplicates": len(allids) - len(set(allids)),
                       "missing": sorted(set(times) - set(allids))[:5], "unknown": sorted(set(allids) - set(times))[:5],
                       "sizes": {k: len(v) for k, v in part.items()}}
    # 複数注釈: 1枚の物理画像に複数の注釈集合がある。split の単位は物理画像なので、同じ画像の集合は同じ側にある
    cnt = {}
    for s in sets_per_stem.values():
        cnt[len(s)] = cnt.get(len(s), 0) + 1
    res["annotation_sets_per_image"] = {str(k): v for k, v in sorted(cnt.items())}
    folds = []
    dev_eval_seen: list[str] = []
    for k, f in enumerate(split["folds"]):
        fit, ev, pu = set(f["fit"]), set(f["eval"]), set(f["purge"])
        dev_eval_seen += f["eval"]
        sets_eval = sum(len(sets_per_stem[i]) for i in f["eval"])
        # 注釈集合IDから画像stemを逆引きした側が、fit/eval で食い違わないこと
        split_sets = {sid for i in f["fit"] for sid in sets_per_stem[i]} & {sid for i in f["eval"] for sid in sets_per_stem[i]}
        folds.append({"fold": k, "n_fit": len(fit), "n_eval": len(ev), "n_purge": len(pu), "n_eval_sets": sets_eval,
                      "fit_eval_overlap": len(fit & ev), "set_id_in_both_sides": len(split_sets),
                      "min_gap_fit_eval_days": round(min_gap_days(times, f["fit"], f["eval"]), 4),
                      "years_eval": _year_counts(f["eval"], times)})
    res["folds"] = folds
    res["development_each_image_evaluated_once"] = sorted(dev_eval_seen) == sorted(split["development"])
    res["min_gap_development_holdout_days"] = round(min_gap_days(times, split["development"], split["holdout"]), 4)
    res["purge_days"] = PURGE_DAYS
    res["time_check_uses"] = "配布 JSON の date_captured（タイムゾーン未指定のまま比較）"
    res["ok"] = (res["coverage"]["duplicates"] == 0 and not res["coverage"]["missing"] and not res["coverage"]["unknown"]
                 and all(f["fit_eval_overlap"] == 0 and f["set_id_in_both_sides"] == 0 and f["min_gap_fit_eval_days"] > PURGE_DAYS for f in folds)
                 and res["development_each_image_evaluated_once"] and res["min_gap_development_holdout_days"] > PURGE_DAYS)
    return res


def _year_counts(ids, times):
    out: dict[int, int] = {}
    for i in ids:
        out[times[i].year] = out.get(times[i].year, 0) + 1
    return {str(k): v for k, v in sorted(out.items())}


def audit_manifests(out_root: Path, split: dict, times: dict) -> list[dict]:
    """outputs/filament/*/foldK.manifest.json を、現在の split と照合する（旧形式は重みのハッシュがない）。"""
    rows = []
    for p in sorted(out_root.glob("*/fold[0-9].manifest.json")):
        man = json.loads(p.read_text())
        k = int(p.name[4])
        f = split["folds"][k]
        row = {"run": p.parent.name, "fold": k, "manifest": str(p), "split_match": man.get("split") == split["hash"],
               "fit_equals_split_fit": sorted(man["fit"]) == sorted(f["fit"]), "weights_found": [str(q) for q in (p.parent / f"fold{k}.pt", p.parent / "fp16" / f"fold{k}.pt") if q.exists()],
               "has_weights_hash": "weights_sha256" in man}
        try:
            row["checks_passed"] = check_clean_evaluation(man, f["eval"], split, times, require_hash=False, require_scratch=False)
            row["clean"] = True
        except LeakageError as e:
            row["clean"] = False
            row["error"] = str(e)
        # 保存済み確率が fold の評価画像と一致するか（あれば）
        probs = sorted(p.parent.glob(f"fold{k}_probs*.npz"))
        row["saved_probs"] = [{"file": q.name, "ids_match_eval": sorted(np.load(q).files) == sorted(f["eval"])} for q in probs]
        rows.append(row)
    return rows


def audit_gt_cache(split: dict, gt_cache: dict[str, list]) -> dict:
    ev = set(split["holdout"]) | set(split["development"])
    return {"cached_images": len(gt_cache), "development_holdout_images": len(ev),
            "missing_in_cache": len(ev - set(gt_cache)), "buffer_cached": len(set(split["holdout_buffer"]) & set(gt_cache))}


def stamp(split_path: Path) -> dict:
    return {"split_file": str(split_path), "split_file_sha256": sha256_file(split_path),
            "development_ids_hash": ids_hash(json.loads(split_path.read_text())["development"])}
