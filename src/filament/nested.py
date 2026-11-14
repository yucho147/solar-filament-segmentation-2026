"""条件選択（inner）と最終比較（outer）を分ける入れ子の評価。

流れ（外側 fold ごと）:
  1. 外側の学習側 (outer_fit) を、27日ブロック単位で inner_fit / inner_tune に分ける（前後14日は inner_fit から除外）。
  2. 候補（学習更新数など）ごとに inner_fit だけで学習し、inner_tune の確率でしきい値・最小面積・min_peak を含めて選ぶ。
     外側の評価画像（outer_eval）は 1. 2. のどこにも現れない。
  3. 選んだ条件を固定し、outer_fit 全体で事前学習の重みから最初から学習し直して outer_eval を評価する。
  4. 検査 `check_clean_evaluation` が、学習画像と評価画像の重複・時間除外の違反・評価画像の混入を拒否する。

重要: この枠組みを使っても、既存の外側 fold と最終 holdout が過去の判断（拡張・後処理の選択、holdout の採点）に
使われた事実は消えない。結果には `usage_history` を添え、未使用の独立評価とは書かない。
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
from loguru import logger

from . import cache as cache_mod
from . import engine
from .evalrecord import per_image_stats, write_rows
from .provenance import env_state, git_state, ids_hash, sha256_file, sha256_obj
from .split import PURGE_DAYS, _near

SCHEMA = "nested_v1"


class LeakageError(AssertionError):
    """評価の独立性を壊す条件。"""


# ------------------------------------------------------------------ inner split
def make_inner(outer_fit: list[str], times: dict, blocks: dict[str, int], seed: int, n_inner: int = 1,
               tune_frac: float = 0.25, purge_days: int = PURGE_DAYS) -> list[dict]:
    """outer_fit を inner_fit / inner_tune / inner_purge に分ける。tune は27日ブロック単位で、n_inner 個は互いに素。"""
    if n_inner * tune_frac > 0.8:
        raise ValueError("n_inner * tune_frac is too large")
    rng = np.random.default_rng(seed)
    by_block: dict[int, list[str]] = {}
    for i in sorted(outer_fit):
        by_block.setdefault(blocks[i], []).append(i)
    order = [int(b) for b in rng.permutation(sorted(by_block))]
    out, pos = [], 0
    for _ in range(n_inner):
        tune: list[str] = []
        while pos < len(order) and len(tune) < tune_frac * len(outer_fit):
            tune += by_block[order[pos]]
            pos += 1
        ts = set(tune)
        rest = [i for i in sorted(outer_fit) if i not in ts]
        purge = set(_near(times, rest, tune, purge_days))
        out.append({"tune": sorted(tune), "purge": sorted(purge), "fit": [i for i in rest if i not in purge]})
    return out


def make_nested(split: dict, times: dict, blocks: dict[str, int], seed: int = 0, n_inner: int = 1,
                tune_frac: float = 0.25) -> dict:
    folds = []
    for k, f in enumerate(split["folds"]):
        folds.append({"outer_fit": f["fit"], "outer_eval": f["eval"], "outer_purge": f["purge"],
                      "inner": make_inner(f["fit"], times, blocks, seed + 1000 * k, n_inner, tune_frac)})
    body = {"schema": SCHEMA, "outer_split_hash": split["hash"], "outer_split_name": split["name"], "seed": seed,
            "n_inner": n_inner, "tune_frac": tune_frac, "purge_days": PURGE_DAYS, "folds": folds}
    return {**body, "hash": sha256_obj(body)[:16]}


def min_gap_days(times: dict, ids_a, ids_b) -> float:
    """ids_a と ids_b の撮影時刻の最短間隔（日）。"""
    a = np.sort([times[i].timestamp() for i in ids_a])
    b = np.array([times[i].timestamp() for i in ids_b])
    if not len(a) or not len(b):
        return float("inf")
    pos = np.searchsorted(a, b)
    lo = np.abs(b - a[np.clip(pos - 1, 0, len(a) - 1)])
    hi = np.abs(b - a[np.clip(pos, 0, len(a) - 1)])
    return float(np.minimum(lo, hi).min() / 86400)


def validate_nested(nested: dict, split: dict, times: dict) -> None:
    body = {k: v for k, v in nested.items() if k != "hash"}
    if nested["hash"] != sha256_obj(body)[:16]:
        raise LeakageError("nested split hash does not match contents")
    if nested["outer_split_hash"] != split["hash"]:
        raise LeakageError("nested split was made from a different outer split")
    forbidden = set(split["holdout"]) | set(split["holdout_buffer"])
    for k, f in enumerate(nested["folds"]):
        o = split["folds"][k]
        if f["outer_fit"] != o["fit"] or f["outer_eval"] != o["eval"]:
            raise LeakageError(f"fold{k}: outer sets differ from the outer split")
        tunes = []
        for j, inner in enumerate(f["inner"]):
            fit, tune, purge = set(inner["fit"]), set(inner["tune"]), set(inner["purge"])
            if not fit or not tune:
                raise LeakageError(f"fold{k}/inner{j}: empty fit or tune")
            if fit & tune or fit & purge or tune & purge:
                raise LeakageError(f"fold{k}/inner{j}: inner sets overlap")
            if (fit | tune | purge) != set(f["outer_fit"]):
                raise LeakageError(f"fold{k}/inner{j}: inner sets do not partition outer_fit")
            if (fit | tune | purge) & (set(f["outer_eval"]) | forbidden):
                raise LeakageError(f"fold{k}/inner{j}: outer_eval/holdout/buffer image in inner sets")
            if min_gap_days(times, inner["fit"], inner["tune"]) <= PURGE_DAYS:
                raise LeakageError(f"fold{k}/inner{j}: inner_fit within {PURGE_DAYS} days of inner_tune")
            if min_gap_days(times, inner["fit"] + inner["tune"], f["outer_eval"]) <= PURGE_DAYS:
                raise LeakageError(f"fold{k}/inner{j}: inner image within {PURGE_DAYS} days of outer_eval")
            tunes.append(tune)
        for a in range(len(tunes)):
            for b in range(a + 1, len(tunes)):
                if tunes[a] & tunes[b]:
                    raise LeakageError(f"fold{k}: inner tune sets overlap")


# ------------------------------------------------------------------ 検査
def check_clean_evaluation(manifest: dict, eval_ids: list[str], split: dict, times: dict,
                           weights: Path | None = None, pred_ids: list[str] | None = None,
                           require_hash: bool = True, require_scratch: bool = True,
                           allow_holdout_eval: bool = False) -> list[str]:
    """学習側 (manifest['fit']) と評価画像の関係を検査する。違反は LeakageError、成功した検査名を返す。"""
    passed: list[str] = []

    def need(cond: bool, msg: str, name: str):
        if not cond:
            raise LeakageError(msg)
        passed.append(name)

    fit, ev = manifest["fit"], list(eval_ids)
    need(manifest.get("split") == split["hash"], f"manifest split {manifest.get('split')} != {split['hash']}", "split_hash")
    need(len(set(fit)) == len(fit), "duplicate image ids in fit", "fit_ids_unique")
    need(len(set(ev)) == len(ev), "duplicate image ids in eval", "eval_ids_unique")
    inter = set(fit) & set(ev)
    need(not inter, f"{len(inter)} evaluation images were used for training (e.g. {sorted(inter)[:3]})", "fit_eval_disjoint")
    hb = set(split["holdout"]) | set(split["holdout_buffer"])
    need(not set(fit) & hb, "training set contains holdout/buffer images", "fit_excludes_holdout")
    if not allow_holdout_eval:
        need(not set(ev) & hb, "evaluation set contains holdout/buffer images", "eval_excludes_holdout")
    gap = min_gap_days(times, fit, ev)
    need(gap > PURGE_DAYS, f"training image within {gap:.2f} days of an evaluation image (limit {PURGE_DAYS})", "time_purge")
    if "fit_ids_hash" in manifest:
        need(manifest["fit_ids_hash"] == ids_hash(fit), "manifest fit_ids_hash does not match fit", "fit_hash")
    if require_scratch:
        need(manifest.get("init_checkpoint") is None, "weights were initialised from a previous checkpoint", "from_scratch")
    if require_hash:
        need(weights is not None and manifest.get("weights_sha256") == sha256_file(weights),
             "weights file hash does not match manifest", "weights_hash")
    if pred_ids is not None:
        need(sorted(pred_ids) == sorted(ev), "predicted image ids differ from evaluation ids (missing/duplicate/extra)", "pred_ids")
    return passed


def check_selection(sel: dict, nested: dict, fold: int, inner_idx: int = 0) -> None:
    """選択記録が、この外側 fold の inner_tune だけから作られたことを検査する。"""
    body = {k: v for k, v in sel.items() if k != "content_hash"}
    if sel["content_hash"] != sha256_obj(body)[:16]:
        raise LeakageError("selection content hash mismatch")
    f = nested["folds"][fold]
    inner = f["inner"][inner_idx]
    if sel["nested_hash"] != nested["hash"] or sel["fold"] != fold:
        raise LeakageError("selection belongs to a different nested split/fold")
    if sel["tune_ids_hash"] != ids_hash(inner["tune"]):
        raise LeakageError("selection was scored on different tune images")
    if set(inner["tune"]) & set(f["outer_eval"]):
        raise LeakageError("tune images overlap outer_eval")
    for c in sel["candidates"]:
        if not set(c["fit_ids"]) <= set(inner["fit"]):
            raise LeakageError(f"candidate {c['name']} was trained on images outside inner_fit")


# ------------------------------------------------------------------ 実行
class LazyUp(dict):
    """.items() で原寸へ補間した確率を1枚ずつ返す辞書（メモリ節約）。"""

    def __init__(self, low):
        super().__init__({k: None for k in low})
        self.low = low

    def items(self):
        return ((k, engine.upsample(v)) for k, v in self.low.items())


def apply_overrides(cfg: engine.Config, overrides: dict) -> engine.Config:
    unknown = set(overrides) - set(engine.Config.__dataclass_fields__)
    if unknown:
        raise ValueError(f"unknown config keys {sorted(unknown)}")
    return dataclasses.replace(cfg, **overrides)


def predict_lowres(models, stems: list[str], cfg: engine.Config, dev) -> tuple[dict, dict]:
    """評価画像の縮小解像度の確率と、推論時間の記録。"""
    t0 = time.perf_counter()
    engine.reset_peak_memory(dev)
    probs = {s: engine.predict(models, cache_mod.load_cfg(s, cfg)[0], dev, cfg).astype(np.float32) for s in stems}
    dt = time.perf_counter() - t0
    return probs, {"infer_seconds_total": dt, "infer_seconds_per_image": dt / max(len(stems), 1),
                   "infer_peak_memory_mb": engine.peak_memory_mb(dev), "n_images": len(stems),
                   "tta_variants": len(engine.tta_variants(cfg.tta, cfg.tta_set)), "infer": cfg.infer}


def write_manifest(path: Path, kind: str, fit: list[str], weights: Path, cfg: engine.Config, split_hash: str,
                   extra: dict | None = None, summary_path: Path | None = None) -> dict:
    summary = json.loads((summary_path or weights.with_suffix(".aug.json")).read_text())
    res = summary.get("resources", {})
    weights_info = None
    if cfg.arch == "dinov3_seg" and cfg.pretrained:
        from .provenance import dinov3_provenance

        weights_info = dinov3_provenance(cfg.encoder)
    man = {"pretrained_weights": weights_info, "kind": kind, "split": split_hash, "fit": fit, "fit_ids_hash": ids_hash(fit), "config": engine.cfg_dict(cfg),
           "config_hash": sha256_obj(engine.cfg_dict(cfg))[:16], "weights": str(weights),
           "weights_sha256": sha256_file(weights), "init_checkpoint": None,
           "pretrained_init": cfg.pretrained, "resources": res, "training_summary": summary,
           "git": git_state(), "env": env_state(),
           **({"position_channels": summary["position_channels"]} if "position_channels" in summary else {}),
           **(extra or {})}
    path.write_text(json.dumps(man, indent=1))
    return man


def run_select(cfg_base: engine.Config, nested: dict, fold: int, select_cfg: dict, out_dir: Path, inner_idx: int = 0,
               max_images: dict | None = None, log=lambda *a, **k: None) -> dict:
    """inner_fit で候補ごとに学習し、inner_tune のみで条件（候補・しきい値・最小面積・min_peak）を選ぶ。"""
    f = nested["folds"][fold]
    inner = f["inner"][inner_idx]
    fit, tune = inner["fit"], inner["tune"]
    if max_images:  # smoke用: 実画像の先頭n枚（決定的）。本番では使わない
        fit, tune = fit[: max_images.get("fit", len(fit))], tune[: max_images.get("tune", len(tune))]
    if set(tune) & set(f["outer_eval"]) or set(fit) & set(f["outer_eval"]):
        raise LeakageError("inner sets contain outer_eval images")
    dev = engine.device()
    run_dir = out_dir / f"fold{fold}" / f"inner{inner_idx}"
    run_dir.mkdir(parents=True, exist_ok=True)
    grid = select_cfg["post_grid"]
    cands = []
    for cand in select_cfg["candidates"]:
        cfg = apply_overrides(cfg_base, cand["overrides"])
        cdir = run_dir / cand["name"]
        cdir.mkdir(exist_ok=True)
        early = bool(cfg.early_stopping.get("enabled", False))
        monitor = early or bool(cfg.early_stopping.get("monitor", False))
        w = engine.train(fit, cfg, cdir / "weights.pt", tune if monitor else None,
                         log=lambda step, metrics: log(step, {f"{cand['name']}.{k}": v for k, v in metrics.items()}))
        man = write_manifest(cdir / "manifest.json", "inner_fit", fit, w, cfg, nested["outer_split_hash"],
                             {"nested_hash": nested["hash"], "fold": fold, "inner_idx": inner_idx})
        models = engine.load_models([w], cfg, dev)
        probs, inf = predict_lowres(models, tune, cfg, dev)
        np.savez_compressed(cdir / "tune_probs.npz", **probs)
        sc = engine.score_grid_peak(LazyUp(probs), grid["thr"], grid["min_area"], grid["min_peak"])
        rows = [{"thr": t, "min_area": a, "min_peak": q, "pq": st.pq, "tp": st.tp, "fp": st.fp, "fn": st.fn}
                for (t, a, q), st in sc.items()]
        best = min(rows, key=lambda r: (-r["pq"], r["thr"], r["min_area"], r["min_peak"]))  # 同点は小さい値を優先
        logger.info("fold{} {} inner_tune best PQ {:.4f} @ {}", fold, cand["name"], best["pq"], best)
        train_info = json.loads(w.with_suffix(".aug.json").read_text())
        best_step = train_info.get("early_stopping", {}).get("best_step")
        cands.append({"name": cand["name"], "overrides": cand["overrides"], "best_post": best, "grid": rows,
                      "refit_steps": best_step if early else None,
                      **({"ema": train_info["ema"]} if "ema" in train_info else {}),
                      "fit_ids": fit, "weights_sha256": man["weights_sha256"], "config_hash": man["config_hash"],
                      "resources": {"train": man["resources"], "infer": inf}})
    pick = min(cands, key=lambda c: (-c["best_post"]["pq"], c["overrides"].get("steps", 0), c["name"]))
    edge = [k for k, v in (("thr", grid["thr"]), ("min_area", grid["min_area"]), ("min_peak", grid["min_peak"]))
            if len(v) > 1 and pick["best_post"][k] in (min(v), max(v))]
    body = {"schema": SCHEMA, "kind": "inner_selection", "nested_hash": nested["hash"],
            "outer_split_hash": nested["outer_split_hash"], "fold": fold, "inner_idx": inner_idx,
            "tune_ids": tune, "tune_ids_hash": ids_hash(inner["tune"]), "tune_subset": bool(max_images),
            "smoke": bool(max_images), "select_cfg_hash": sha256_obj(select_cfg)[:16],
            "base_config_hash": sha256_obj(engine.cfg_dict(cfg_base))[:16], "candidates": cands,
            "selected": {"candidate": pick["name"], "overrides": pick["overrides"],
                         "refit_steps": pick.get("refit_steps"),
                         "post": {k: pick["best_post"][k] for k in ("thr", "min_area", "min_peak")},
                         "inner_tune_pq": pick["best_post"]["pq"], "grid_edge_params": edge},
            "selection_data": "inner_tune only (outer_eval was not read)", "git": git_state(), "env": env_state()}
    sel = {**body, "content_hash": sha256_obj(body)[:16]}
    (run_dir / "selection.json").write_text(json.dumps(sel, indent=1))
    return sel


def run_refit(cfg_base: engine.Config, nested: dict, split: dict, fold: int, selection_path: Path, out_dir: Path,
              times: dict, blocks: dict[str, int], inner_idx: int = 0, max_images: dict | None = None,
              usage_history: dict | None = None, log=lambda *a, **k: None, refit_steps_override: int | None = None) -> dict:
    """選んだ条件を固定し、outer_fit 全体で最初から学習し直して outer_eval を評価する。

    refit_steps_override: 診断実験用。内側で選んだ更新数の代わりに使う更新数（明示的な指定のみ。外側の結果から決めない。
    値は manifest・評価記録に selection_refit_steps と並べて残す）。None なら選択記録の更新数をそのまま使う。"""
    sel = json.loads(Path(selection_path).read_text())
    check_selection(sel, nested, fold, inner_idx)
    f = nested["folds"][fold]
    fit, ev = f["outer_fit"], f["outer_eval"]
    if max_images:
        fit, ev = fit[: max_images.get("fit", len(fit))], ev[: max_images.get("eval", len(ev))]
    cfg = apply_overrides(cfg_base, sel["selected"]["overrides"])
    post = sel["selected"]["post"]
    rdir = out_dir / f"fold{fold}"
    rdir.mkdir(parents=True, exist_ok=True)
    selection_steps = sel["selected"].get("refit_steps")
    selected_steps = selection_steps
    if refit_steps_override is not None:
        if not 1 <= int(refit_steps_override) <= cfg.steps:
            raise ValueError(f"refit_steps_override {refit_steps_override} must be within 1..cfg.steps ({cfg.steps})")
        selected_steps = int(refit_steps_override)
        logger.warning("refit steps overridden: selection {} -> {} (diagnostic run; not chosen from outer results)", selection_steps, selected_steps)
    if cfg.early_stopping.get("enabled", False) and selected_steps is None:
        raise LeakageError("early stopping selection lacks a fixed refit step")
    if selected_steps is None and cfg.early_stopping.get("monitor", False):
        selected_steps = cfg.steps
    ema_cfg = engine.ema_settings(cfg)
    w = engine.train(fit, cfg, rdir / "weights.pt", None, log=log, fixed_steps=selected_steps)
    man = write_manifest(rdir / "manifest.json", "outer_refit", fit, w, cfg, nested["outer_split_hash"],
                         {"nested_hash": nested["hash"], "fold": fold, "selection_content_hash": sel["content_hash"],
                          "selection_file_sha256": sha256_file(selection_path), "refit_steps": selected_steps,
                          "selection_refit_steps": selection_steps, "refit_steps_overridden": refit_steps_override is not None,
                          "smoke": bool(max_images), **({"ema": ema_cfg} if ema_cfg else {})})  # ライブ重みはEMAなしの学習と同じ
    rec = _evaluate_weights(w, rdir, man, cfg, ev, split, times, blocks, post, sel, nested, fold, fit,
                            selected_steps=selected_steps, selection_steps=selection_steps,
                            override=refit_steps_override is not None, max_images=max_images, usage_history=usage_history,
                            ema_note=ema_cfg and {"variant": "live", "ema": ema_cfg})
    if ema_cfg:
        # 同じ1回の学習で得たEMA重みを、同じ外側評価画像・同じ後処理（内側tuneで選択済み）で評価する。
        # 更新数・decay・後処理は外側の結果で選ばない。fit idsはライブ重みと同一（check_clean_evaluationを各変種で通す）
        variants = {}
        for d in ema_cfg["decays"]:
            tag = engine.ema_tag(d)
            ew = engine.ema_weights_path(w, d)
            vdir = rdir / f"ema_{tag}"
            vdir.mkdir(exist_ok=True)
            vman = write_manifest(
                vdir / "manifest.json", "outer_refit", fit, ew, cfg, nested["outer_split_hash"],
                {"nested_hash": nested["hash"], "fold": fold, "selection_content_hash": sel["content_hash"],
                 "selection_file_sha256": sha256_file(selection_path), "refit_steps": selected_steps,
                 "selection_refit_steps": selection_steps, "refit_steps_overridden": refit_steps_override is not None,
                 "smoke": bool(max_images), "ema": {**ema_cfg, "variant_decay": d, "n_updates": man["training_summary"]["ema"]["n_updates"][tag],
                                                   "live_weights": str(w), "live_weights_sha256": man["weights_sha256"],
                                                   "fit_ids_hash_equals_live": True}},
                summary_path=w.with_suffix(".aug.json"))
            if vman["fit_ids_hash"] != man["fit_ids_hash"]:
                raise LeakageError("EMA variant fit ids differ from the live weights")
            vrec = _evaluate_weights(ew, vdir, vman, cfg, ev, split, times, blocks, post, sel, nested, fold, fit,
                                     selected_steps=selected_steps, selection_steps=selection_steps,
                                     override=refit_steps_override is not None, max_images=max_images,
                                     usage_history=usage_history, ema_note={"variant": f"ema_{tag}", "decay": d, "ema": ema_cfg,
                                                                            "paired_live_weights_sha256": man["weights_sha256"]})
            variants[tag] = {"pq": vrec["pq"], "sq": vrec["sq"], "rq": vrec["rq"], "tp": vrec["tp"], "fp": vrec["fp"],
                             "fn": vrec["fn"], "dir": str(vdir)}
        rec["ema_variants"] = variants
        (rdir / "evaluation_record.json").write_text(json.dumps(rec, indent=1, default=str))
        logger.info("fold{} outer PQ live {:.4f} / {}", fold, rec["pq"], {t: round(v["pq"], 4) for t, v in variants.items()})
    return rec


def _evaluate_weights(w: Path, odir: Path, man: dict, cfg: engine.Config, ev: list[str], split: dict, times: dict,
                      blocks: dict, post: dict, sel: dict, nested: dict, fold: int, fit: list[str],
                      selected_steps, selection_steps, override: bool, max_images, usage_history, ema_note=None) -> dict:
    """1つの重みファイルを外側評価画像で評価し、per_image.csv・eval_probs.npz・evaluation_record.json を odir に書く。"""
    check_clean_evaluation(man, ev, split, times, w)  # 予測の前
    dev = engine.device()
    models = engine.load_models([w], cfg, dev)
    probs, inf = predict_lowres(models, ev, cfg, dev)
    del models
    passed = check_clean_evaluation(man, ev, split, times, w, pred_ids=list(probs))  # 予測の後
    np.savez_compressed(odir / "eval_probs.npz", **probs)
    rows = per_image_stats(LazyUp(probs), engine.gt_rles(ev), post, blocks, times)
    write_rows(rows, odir / "per_image.csv")
    tot = engine.PQStats()
    for r in rows:
        tot.add(engine.PQStats(r["iou_sum"], r["tp"], r["fp"], r["fn"]))
    rec = {"schema": SCHEMA, "kind": "outer_evaluation", "fold": fold, "nested_hash": nested["hash"],
           "outer_split_hash": nested["outer_split_hash"], "selection_content_hash": sel["content_hash"],
           "selected": sel["selected"], "weights_sha256": man["weights_sha256"], "config_hash": man["config_hash"],
           "fit_ids_hash": man["fit_ids_hash"], "eval_ids": ev, "eval_ids_hash": ids_hash(ev),
           "n_fit": len(fit), "n_eval": len(ev), "checks_passed": passed,
           "pq": tot.pq, "sq": tot.sq, "rq": tot.rq, "tp": tot.tp, "fp": tot.fp, "fn": tot.fn,
           "iou_sum": tot.iou_sum, "resources": {"train": man["resources"], "infer": inf},
           "refit_steps": selected_steps, "selection_refit_steps": selection_steps, "refit_steps_overridden": override,
           "smoke": bool(max_images), "usage_history": usage_history or "not attached",
           "independence_note": "外側 fold/holdout は過去の判断に使用済み。この値は新しい手順の外側評価であり、未使用の独立評価ではない",
           **({"ema_eval": ema_note, "post_selected_with": "live weights on inner_tune (same post for live and EMA)"} if ema_note else {}),
           "git": git_state(), "env": env_state()}
    (odir / "evaluation_record.json").write_text(json.dumps(rec, indent=1, default=str))
    logger.info("fold{} outer PQ {:.4f} (tp={} fp={} fn={}) checks={} [{}]", fold, tot.pq, tot.tp, tot.fp, tot.fn, len(passed), odir.name)
    return rec
