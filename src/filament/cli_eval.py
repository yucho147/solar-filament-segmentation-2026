"""評価の信頼性とモデル比較の入口。
使い方: uv run --group vision --group research python -m filament.cli_eval <command> ...
  audit / neardup / selfeval / ceiling / inner-split / select / refit / compare
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from . import cache as cache_mod
from . import engine
from .data import ROOT, load_train
from .provenance import env_state, git_state

OUT = Path("outputs/filament")
SPLIT = OUT / "split.json"
NESTED = OUT / "nested" / "nested_split.json"
USAGE = Path("configs/filament/usage_history.yaml")


def _times_sets():
    # 撮影時刻と注釈集合の参照だけを使うので、元画像が無いGPU機でも動くよう画像の存在確認は省く（学習・推論はキャッシュを使う）
    imgs = load_train(check_images=(ROOT / "train/train_images").exists())
    by_stem = {p.stem: p for p in imgs.values()}
    return by_stem, {s: p.captured for s, p in by_stem.items()}


def _split() -> dict:
    from .split import validate_split

    s = json.loads(SPLIT.read_text())
    validate_split(s)
    return s


def _cfg(path: str, overrides: list[str]) -> engine.Config:
    d = yaml.safe_load(Path(path).read_text())
    for o in overrides:
        k, v = o.split("=", 1)
        d[k] = yaml.safe_load(v)
    return engine.Config(**d)


def _write(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str))
    logger.info("wrote {}", path)


# ---------------------------------------------------------------- audit
def cmd_audit(a):
    from . import audit

    by_stem, times = _times_sets()
    sp = _split()
    sets = {s: [x.set_id for x in p.sets] for s, p in by_stem.items()}
    rep = audit.audit_split(sp, times, sets)
    rep["manifests"] = audit.audit_manifests(OUT, sp, times)
    gt = engine.gt_rles(sorted(set(sp["development"]) | set(sp["holdout"])))
    rep["gt_cache"] = audit.audit_gt_cache(sp, gt)
    rep["file_stamp"] = audit.stamp(SPLIT)
    rep["usage_history"] = yaml.safe_load(USAGE.read_text())
    rep["git"], rep["env"] = git_state(), env_state()
    _write(OUT / "audit" / "split_audit.json", rep)
    bad = [m for m in rep["manifests"] if not m["clean"] or not m["split_match"] or not m["fit_equals_split_fit"]]
    logger.info("split ok={} manifests={} problems={}", rep["ok"], len(rep["manifests"]), len(bad))
    for m in bad:
        logger.warning("{} fold{}: {}", m["run"], m["fold"], m.get("error", "fit differs from split"))


# ---------------------------------------------------------------- neardup
def cmd_neardup(a):
    from . import neardup as nd

    by_stem, times = _times_sets()
    sp = _split()
    paths = {s: ROOT / "train/train_images" / p.file_name for s, p in by_stem.items()}
    sites = {s: s[-2:] for s in paths}
    test = {f.stem: f for f in sorted((ROOT / "test/test_images").glob("*.jpeg"))}
    from datetime import datetime

    for s, f in test.items():  # テスト画像の時刻はファイル名 YYYYMMDDHHMMSS から（配布メタデータに無い）
        times[s] = datetime.strptime(s[:14], "%Y%m%d%H%M%S")
        sites[s] = s[-2:]
    paths.update(test)
    logger.info("embedding {} images", len(paths))
    stems, v = nd.embeddings(paths, sites)
    corr = v @ v.T
    t = np.array([times[s].timestamp() for s in stems])
    days = np.abs(t[:, None] - t[None]) / 86400
    ss = np.array([[sites[x] == sites[y] for y in stems] for x in stems])
    np.fill_diagonal(corr, -1)
    # どの境界をまたぐ組か
    fold_of = []
    for k, f in enumerate(sp["folds"]):
        fold_of.append((k, set(f["fit"]), set(f["eval"])))
    ho, dev = set(sp["holdout"]), set(sp["development"])

    def relation(x, y):
        rel = []
        for k, fit, ev in fold_of:
            if (x in fit and y in ev) or (y in fit and x in ev):
                rel.append(f"fold{k}:fit-eval")
        if (x in ho and y in dev) or (y in ho and x in dev):
            rel.append("dev-holdout")
        if (x in test) != (y in test):
            rel.append("train-test")
        return rel

    ref = nd.reference_distributions(corr, days, ss)
    rows = nd.candidates(stems, corr, days, ss, relation, top=a.top)
    d = OUT / "audit" / "neardup"
    nd.write_csv(rows, d / "candidates.csv")
    nd.contact_sheet(rows, paths, times, d / "candidates_top.png", n=a.sheet)
    only_train = [r for r in rows if "train-test" not in r["boundaries"]]
    _write(d / "summary.json", {"n_images": len(stems), "reference": ref, "n_candidates": len(rows),
                                "top_train_internal": only_train[:5], "note": "候補は目視確認用。相関だけで重複の有無を断定しない",
                                "git": git_state()})
    logger.info("reference {}", json.dumps(ref)[:600])
    logger.info("top candidates:")
    for r in rows[:8]:
        logger.info("  {a} {b} corr={corr:.3f} gap={gap_days:.0f}d {boundaries}", **r)


# ---------------------------------------------------------------- selfeval
def _instances_of(stem_set) -> list[np.ndarray]:
    return [m.astype(bool) for m in stem_set.instance_masks()]


def _half_mask(m: np.ndarray, extra: int = 0) -> np.ndarray:
    ys, xs = np.nonzero(m)
    k = len(ys) // 2 + extra
    out = np.zeros_like(m)
    out[ys[:k], xs[:k]] = True
    return out


def cmd_selfeval(a):
    from . import selfeval_check as se
    from .postprocess import instances
    from scipy import ndimage as ndi

    ref = se.load_reference()
    by_stem, times = _times_sets()
    sp = _split()
    ev = sp["folds"][0]["eval"]
    pick = {}
    for n in (3, 2, 1):  # 注釈集合が3/2/1つの実画像を2枚ずつ
        pick[n] = [s for s in ev if len(by_stem[s].sets) == n][:2]
    stems = [s for n in (3, 2, 1) for s in pick[n]]
    results = {}

    def gt_of(sts):
        return {x.set_id: _instances_of(x) for s in sts for x in by_stem[s].sets}

    def run(name, sts, pred, note=""):
        gt_df = se.make_gt_df(gt_of(sts))
        pred_df = se.make_pred_df(pred)
        r = se.compare(gt_df, pred_df, ref)
        r["note"] = note
        results[name] = r
        logger.info("{:<28s} ref {:.6f} local {:.6f} diff {:.1e} sets={} mismatches={}", name, r["ref_pq"], r["local_pq"],
                    r["abs_diff"], r["n_sets"], len(r["set_level_mismatches"]))

    s0 = {s: _instances_of(by_stem[s].sets[0]) for s in stems}
    run("real_gt_set0_as_pred", stems, s0, "実画像6枚（注釈集合3/2/1）、集合0を予測として全集合に当てる")
    run("real_shift4px", stems, {s: [np.roll(m, 4, axis=1) for m in ms] for s, ms in s0.items()}, "集合0を横に4画素ずらした予測")
    run("real_dilate3", stems, {s: [ndi.binary_dilation(m, iterations=3) for m in ms] for s, ms in s0.items()}, "集合0を3画素膨張")
    run("empty_prediction", stems, {}, "予測0件（全て見逃し）")
    run("pred_for_unknown_image_only", stems, {"20000101000000Xh": [np.ones((2048, 2048), bool)]}, "GTに無い画像の予測だけ（無視されるはず）")
    # IoU 0.5 ちょうど / 直上（実在する偶数面積の個体）
    one = next(s for s in stems if any(m.sum() % 2 == 0 and m.sum() >= 200 for m in s0[s]))
    m = next(m for m in s0[one] if m.sum() % 2 == 0 and m.sum() >= 200)
    sid = by_stem[one].sets[0].set_id
    single = lambda mask: ({sid: [m]}, {one: [mask]})  # noqa: E731
    for nm, mask, note in (("iou_exactly_0.5", _half_mask(m), "正解の半分の画素（IoU=0.5ちょうど）"),
                           ("iou_just_above_0.5", _half_mask(m, 1), "半分+1画素（IoU>0.5）")):
        g, p = single(mask)
        gt_df, pred_df = se.make_gt_df(g), se.make_pred_df(p)
        r = se.compare(gt_df, pred_df, ref)
        r["note"] = note
        results[nm] = r
        logger.info("{:<28s} ref {:.6f} local {:.6f} tp={}", nm, r["ref_pq"], r["local_pq"], r["local"]["tp"])
    # 重複: 同じ予測を2回、同じ正解を2回
    g, p = single(m)
    for nm, gg, pp, note in (("duplicate_prediction", g, {one: [m, m]}, "同じ予測を2つ出す（公開コードは両方TP）"),
                             ("duplicate_gt", {sid: [m, m]}, {one: [m]}, "同じ正解が2つ（予測1つが両方に対応）")):
        gt_df, pred_df = se.make_gt_df(gg), se.make_pred_df(pp)
        r = se.compare(gt_df, pred_df, ref)
        r["note"] = note
        results[nm] = r
        logger.info("{:<28s} ref {:.6f} local {:.6f} tp={} fp={} fn={}", nm, r["ref_pq"], r["local_pq"], r["local"]["tp"], r["local"]["fp"], r["local"]["fn"])
    # GTが空の画像: 公開コードは gt_df に行のない画像を数えない。ローカルは予測を FP と数える
    from .metrics import encode, pq_stats

    fp_local = pq_stats([], encode(np.stack([m])[:1])).fp
    results["gt_empty_image_semantics"] = {"local_counts_fp": fp_local, "reference_counts_fp": 0,
                                          "note": "GTが空の画像は gt_df に行が無く、公開コードは予測を数えない。学習データに空集合は0件なので学習側の評価に影響しないが、テストの扱いは不明"}
    # 大きい面積で float32 の境界: GT=上半分(2,097,152画素)、予測=その中の 1,048,577 画素 (IoU=0.5000005)
    big = np.zeros((2048, 2048), bool)
    big[:1024] = True
    pred_big = np.zeros_like(big)
    pred_big.reshape(-1)[:1048577] = True
    g_df = se.make_gt_df({"000000-20000101000000Xh": [big]})
    r = se.compare(g_df, se.make_pred_df({"20000101000000Xh": [pred_big]}), ref)
    r["note"] = "面積約2百万画素、IoU=0.5000005（float32の丸め差が出うる大きさ）"
    results["large_area_iou_0.5000005"] = r
    logger.info("{:<28s} ref {:.6f} local {:.6f} tp={}", "large_area_iou_0.5000005", r["ref_pq"], r["local_pq"], r["local"]["tp"])
    # 実モデルの予測
    if a.weights:
        cfg = _cfg(a.config, [])
        dev = engine.device()
        models = engine.load_models([Path(a.weights)], cfg, dev)
        post = {"thr": 0.5, "min_area": 400, "min_peak": 0.8}
        sts = ev[: a.n_model_images]
        pred = {}
        for s in sts:
            p = engine.upsample(engine.predict(models, cache_mod.load_cfg(s, cfg)[0], dev, cfg))
            pred[s] = [x for x in instances(p, **post)]
        run("real_model_predictions", sts, pred, f"{a.weights} の fold0 評価画像 {len(sts)} 枚、後処理 {post}")
    out = {"notebook_sha256": se.NOTEBOOK_SHA256, "cases": results, "git": git_state(), "env": env_state(),
           "scope": "公開 Self Evaluation Notebook の関数定義との比較。Kaggle の非公開採点処理との一致は未確認"}
    _write(OUT / "audit" / "selfeval_check.json", out)


# ---------------------------------------------------------------- ceiling
def cmd_ceiling(a):
    from . import analysis as an
    from . import resolution as rs

    sp = _split()
    stems = sp["folds"][a.fold]["eval"][: a.n]
    gts = engine.gt_rles(stems)
    post = {"thr": 0.5, "min_area": a.min_area}
    grids = {"fullres(1x)": 1, "unet@1024 (cell 2px)": 2, "segformer@1024 (cell 8px)": 8, "segformer@2048 (cell 4px)": 4,
             "unet@512 (cell 4px)": 4, "segformer@512 (cell 16px)": 16}
    grids = {k: v for k, v in grids.items() if k in a.grids} if a.grids else grids
    widths = np.concatenate([rs.instance_widths(g) for s in stems for g in gts[s][:1]])
    res = {"fold": a.fold, "n_images": len(stems), "post": post,
           "gt_instance_width_px": {"n": int(len(widths)), **{f"p{p}": float(np.percentile(widths, p)) for p in (5, 25, 50, 75, 95)}}}
    soft = {s: rs.soft_target_fullres(gts[s]) for s in stems}
    out = {}
    for name, cell in grids.items():
        class P(dict):
            def items(self_, cell=cell):  # noqa: N805
                return ((s, rs.through_grid(soft[s], cell)) for s in stems)

        r = an.analyze(P({s: None for s in stems}), gts, post)
        r.pop("per_image")
        out[name] = {"cell_px": cell, **{k: r[k] for k in ("pq", "sq", "rq", "tp", "fp", "fn", "split_gt", "merge_pred")},
                     "by_area": r["by_area"]}
        logger.info("{:<28s} oracle PQ {:.4f} tp={} fp={} fn={}", name, r["pq"], r["tp"], r["fp"], r["fn"])
    res["oracle"] = out
    res["interpretation"] = "正解の注釈集合平均を各格子まで縮めて戻した上限。モデル性能ではなく解像度による損失の目安"
    _write(OUT / "audit" / f"resolution_ceiling_fold{a.fold}.json", res)


# ---------------------------------------------------------------- nested
def cmd_inner_split(a):
    from . import nested as nt
    from .evalrecord import block_table

    _, times = _times_sets()
    sp = _split()
    blocks = block_table(times)
    ns = nt.make_nested(sp, times, blocks, seed=a.seed, n_inner=a.n_inner, tune_frac=a.tune_frac)
    nt.validate_nested(ns, sp, times)
    _write(NESTED, ns)
    for k, f in enumerate(ns["folds"]):
        for j, i in enumerate(f["inner"]):
            logger.info("fold{} inner{}: fit={} tune={} purge={} (outer_fit={} outer_eval={})", k, j, len(i["fit"]), len(i["tune"]),
                        len(i["purge"]), len(f["outer_fit"]), len(f["outer_eval"]))
    logger.info("nested hash {}", ns["hash"])


def _load_nested():
    from . import nested as nt

    _, times = _times_sets()
    sp = _split()
    ns = json.loads(NESTED.read_text())
    nt.validate_nested(ns, sp, times)
    return ns, sp, times


def _mlflow(name: str, cfg: engine.Config, tags: dict, smoke: bool):
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("filament-nested-smoke" if smoke else "filament-nested")
    mlflow.start_run(run_name=name, tags=tags)
    mlflow.log_params({k: str(v) for k, v in engine.cfg_dict(cfg).items()})
    return mlflow


def cmd_select(a):
    from . import nested as nt

    ns, sp, times = _load_nested()
    cfg = _cfg(a.config, a.override)
    sel_cfg = yaml.safe_load(Path(a.select_config).read_text())
    if a.steps_scale != 1.0:  # smoke/見積り用。本番では使わない
        for c in sel_cfg["candidates"]:
            c["overrides"]["steps"] = max(2, int(c["overrides"]["steps"] * a.steps_scale))
    mx = {"fit": a.max_fit, "tune": a.max_tune} if a.smoke else None
    out = OUT / "nested" / (a.name + ("_smoke" if a.smoke else ""))
    mlf = _mlflow(f"{a.name}-select-fold{a.fold}", cfg, {"phase": "inner_select", "fold": str(a.fold), "nested": ns["hash"]}, a.smoke)
    try:
        sel = nt.run_select(cfg, ns, a.fold, sel_cfg, out, 0, mx, log=lambda s, d: mlf.log_metrics(d, step=s))
        mlf.log_metrics({"inner_tune_pq": sel["selected"]["inner_tune_pq"]})
        mlf.log_artifact(str(out / f"fold{a.fold}" / "inner0" / "selection.json"))
        histories = sorted((out / f"fold{a.fold}" / "inner0").glob("*/weights.history.csv"))
        for path in histories:
            mlf.log_artifact(str(path), artifact_path=f"learning_curves/{path.parent.name}")
            summary = path.with_name("weights.aug.json")
            if summary.exists():
                mlf.log_artifact(str(summary), artifact_path=f"learning_curves/{path.parent.name}")
        if histories:
            from .learning_curves import plot_fold_curves, plot_histories

            image = out / f"fold{a.fold}" / "inner0" / "learning_curves.png"
            aggregate = out / "learning_curves_by_fold.png"
            all_histories = sorted(out.glob("fold*/inner*/*/weights.history.csv"))
            if "train_fixed_loss" in histories[0].read_text().splitlines()[0]:  # 同一定義のtrain/validation損失と学習率つき
                def runs(paths):
                    return [{"label": "/".join(p.parts[-4:-1]), "history": p, "summary": p.with_name("weights.aug.json")}
                            for p in paths]

                plot_fold_curves(runs(histories), image)
                plot_fold_curves(runs(all_histories), aggregate)
            else:
                plot_histories(histories, image)
                plot_histories(all_histories, aggregate)
            mlf.log_artifact(str(image))
            mlf.log_artifact(str(aggregate))
    finally:
        mlf.end_run()
    logger.info("selected {}", sel["selected"])


def cmd_refit(a):
    from . import nested as nt
    from .evalrecord import block_table

    ns, sp, times = _load_nested()
    cfg = _cfg(a.config, a.override)
    out = OUT / "nested" / (a.name + ("_smoke" if a.smoke else ""))
    sel_path = out / f"fold{a.fold}" / "inner0" / "selection.json"
    mx = {"fit": a.max_fit, "eval": a.max_eval} if a.smoke else None
    usage = yaml.safe_load(USAGE.read_text())
    mlf = _mlflow(f"{a.name}-refit-fold{a.fold}", cfg, {"phase": "outer_refit", "fold": str(a.fold), "nested": ns["hash"]}, a.smoke)
    try:
        rec = nt.run_refit(cfg, ns, sp, a.fold, sel_path, out / "outer", times, block_table(times), 0, mx, usage,
                           log=lambda s, d: mlf.log_metrics(d, step=s), refit_steps_override=a.refit_steps)
        mlf.log_metrics({"outer_pq": rec["pq"], "outer_sq": rec["sq"], "outer_rq": rec["rq"]})
        for tag, v in rec.get("ema_variants", {}).items():  # EMA有効時: ライブ重みと同じ学習・同じ外側評価画像の対比較
            mlf.log_metrics({f"outer_pq_ema_{tag}": v["pq"]})
        mlf.log_artifact(str(out / "outer" / f"fold{a.fold}" / "evaluation_record.json"))
        history = out / "outer" / f"fold{a.fold}" / "weights.history.csv"
        if history.exists():
            mlf.log_artifact(str(history), artifact_path="learning_curves")
    finally:
        mlf.end_run()


def cmd_compare(a):
    from . import evalrecord as er

    rows = {}
    for tag, runs in (("a", a.a), ("b", a.b)):
        rows[tag] = [r for p in runs for r in er.read_rows(Path(p))]
    res = er.paired_block_bootstrap(rows["a"], rows["b"], a.n_boot, a.seed)
    res["a"], res["b"] = a.a, a.b
    res["by_year_a"], res["by_year_b"] = er.subgroup_pq(rows["a"], "year"), er.subgroup_pq(rows["b"], "year")
    res["by_site_a"], res["by_site_b"] = er.subgroup_pq(rows["a"], "site"), er.subgroup_pq(rows["b"], "site")
    _write(Path(a.out), res)
    logger.info("PQ a {:.4f} b {:.4f} diff {:+.4f} CI {} P(b>a)={:.3f} blocks={}", res["pq_a"], res["pq_b"], res["diff_b_minus_a"],
                [round(x, 4) for x in res["diff_ci"]], res["prob_b_better"], res["n_blocks"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("audit")
    p = sub.add_parser("neardup")
    p.add_argument("--top", type=int, default=60)
    p.add_argument("--sheet", type=int, default=12)
    p = sub.add_parser("selfeval")
    p.add_argument("--weights")
    p.add_argument("--config", default="configs/filament/nested/unet_effv2s.yaml")
    p.add_argument("--n-model-images", type=int, default=10)
    p = sub.add_parser("ceiling")
    p.add_argument("--fold", type=int, default=0)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--min-area", type=int, default=400)
    p.add_argument("--grids", nargs="*")
    p = sub.add_parser("inner-split")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-inner", type=int, default=1)
    p.add_argument("--tune-frac", type=float, default=0.25)
    for n in ("select", "refit"):
        p = sub.add_parser(n)
        p.add_argument("--config", required=True)
        p.add_argument("--name", required=True)
        p.add_argument("--fold", type=int, required=True)
        p.add_argument("-o", "--override", action="append", default=[],
                       help="設定の上書き。重みのEMA(refit/select): -o 'extra={ema: {decay: 0.999, extra_decays: [0.995]}}'")
        p.add_argument("--smoke", action="store_true", help="実画像の先頭n枚だけで動作確認（結果は評価に使わない）")
        p.add_argument("--max-fit", type=int, default=16)
        p.add_argument("--max-tune", type=int, default=6)
        p.add_argument("--max-eval", type=int, default=6)
        if n == "select":
            p.add_argument("--select-config", default="configs/filament/nested/select_default.yaml")
            p.add_argument("--steps-scale", type=float, default=1.0)
        else:
            p.add_argument("--refit-steps", type=int, default=None,
                           help="診断用: 内側で選んだ再学習更新数の代わりに使う更新数。事前に固定した値のみ（外側の結果から決めない）。manifest・評価記録に残る")
    p = sub.add_parser("compare")
    p.add_argument("--a", nargs="+", required=True, help="基準モデルの per_image.csv")
    p.add_argument("--b", nargs="+", required=True, help="比較モデルの per_image.csv")
    p.add_argument("--out", required=True)
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    globals()[f"cmd_{a.command.replace('-', '_')}"](a)


if __name__ == "__main__":
    main()
