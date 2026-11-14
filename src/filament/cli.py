"""入口。使い方: uv run --group vision --group research python -m filament.cli <command> ..."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from loguru import logger

from . import cache as cache_mod
from . import engine, submission
from .data import ROOT, list_test_images, load_train, read_gray
from .split import make_split, split_hash, validate_split

OUT = Path("outputs/filament")
SPLIT = OUT / "split.json"


def _cfg(path: str | None, overrides: list[str]) -> engine.Config:
    d = yaml.safe_load(Path(path).read_text()) if path else {}
    for o in overrides:
        k, v = o.split("=", 1)
        d[k] = yaml.safe_load(v)
    unknown = set(d) - set(engine.Config.__dataclass_fields__)
    if unknown:
        raise SystemExit(f"unknown config keys: {sorted(unknown)}")
    return engine.Config(**d)


def _tracker(name: str, cfg: engine.Config, tags: dict):
    import mlflow

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.set_experiment("filament-segmentation-2026")
    run = mlflow.start_run(run_name=name, tags=tags)
    mlflow.log_params({k: str(v) for k, v in engine.cfg_dict(cfg).items()})
    return mlflow, run


def cmd_split(a, cfg):
    imgs = load_train()
    times = {k: v.captured for k, v in imgs.items()}
    s = make_split(times, cfg.seed)
    validate_split(s, times)  # 画像名(file_name)のまま被覆・距離を検査してから、拡張子なしstemへ変換
    strip = lambda ids: [x[:-5] for x in ids]
    s = {**s, "folds": [{k: strip(v) for k, v in f.items()} for f in s["folds"]],
         **{k: strip(s[k]) for k in ("holdout", "holdout_buffer", "development")}}
    s["hash"] = split_hash(s)
    validate_split(s)
    OUT.mkdir(parents=True, exist_ok=True)
    SPLIT.write_text(json.dumps(s, indent=1))
    logger.info("split {} holdout={} buffer={} dev={} folds={}", s["hash"], len(s["holdout"]),
                len(s["holdout_buffer"]), len(s["development"]), [(len(f["fit"]), len(f["eval"])) for f in s["folds"]])


def cmd_cache(a, cfg):
    cache_mod.build(cfg.factor)


def _split():
    s = json.loads(SPLIT.read_text())
    validate_split(s)  # 読込のたびにhash・集合関係を検査
    return s


def cmd_train(a, cfg):
    """--target fold0..fold4 | holdout_check | full"""
    if cfg.early_stopping.get("enabled", False) or cfg.early_stopping.get("monitor", False):
        raise ValueError("validation loss monitoring requires nested select; this command uses outer evaluation images")
    sp = _split()
    t = a.target
    if t.startswith("fold"):
        f = sp["folds"][int(t[4:])]
        fit, val = f["fit"], f["eval"]
    elif t == "holdout_check":
        fit, val = sp["development"], None  # holdoutは学習中に見ない。採点は predict --split holdout のみ
    elif t == "full":
        fit, val = sp["holdout"] + sp["holdout_buffer"] + sp["development"], None
    else:
        raise SystemExit("bad target")
    mlflow, run = _tracker(f"{a.name}-{t}", cfg, {"target": t, "split": sp["hash"]})
    try:
        path = engine.train(fit, cfg, OUT / a.name / f"{t}.pt", val,
                            log=lambda s, d: mlflow.log_metrics(d, step=s))
        # 学習後は評価idを一度だけ採点し、確率も保存（しきい値調整用）
        if val:
            dev = engine.device()
            models = engine.load_models([path], cfg, dev)
            probs = {}
            for s in val:
                img, _ = cache_mod.load_cfg(s, cfg)
                probs[s] = engine.predict(models, img, dev, cfg).astype(np.float32)
            np.savez_compressed(OUT / a.name / f"{t}_probs.npz", **probs)
            st = engine.score_probs({k: engine.upsample(v) for k, v in probs.items()},
                                    cfg.thr, cfg.min_area)
            logger.info("{} eval PQ {:.4f} SQ {:.4f} RQ {:.4f} (tp={} fp={} fn={})", t, st.pq, st.sq, st.rq, st.tp, st.fp, st.fn)
            mlflow.log_metrics({"eval_pq": st.pq, "eval_sq": st.sq, "eval_rq": st.rq})
        (OUT / a.name / f"{t}.manifest.json").write_text(json.dumps(
            {"target": t, "split": sp["hash"], "fit": fit, "config": engine.cfg_dict(cfg)}))
        mlflow.log_artifact(str(path))
    finally:
        mlflow.end_run()


class _LazyUp(dict):
    """score_probsへ渡す辞書。値は参照時に原寸へ補間し保持しない。"""

    def __init__(self, low):
        super().__init__({k: None for k in low})
        self.low = low

    def items(self):
        return ((k, engine.upsample(v)) for k, v in self.low.items())


def cmd_tune(a, cfg):
    """fold評価の確率(OOF)だけを使い、しきい値/最小面積を探索して保存する。holdoutは使わない。"""
    sp = _split()
    bad = [t for t in a.targets if not (t.startswith("fold") and t[4:].isdigit())]
    if bad:
        raise SystemExit(f"tune only accepts dev fold OOF probabilities, got {bad}")
    lowres = {}  # 縮小解像度のまま保持(約2 MiB/枚)し、採点時に1枚ずつ原寸へ補間する
    for t in a.targets:
        man = json.loads((OUT / a.name / f"{t}.manifest.json").read_text())
        assert man["split"] == sp["hash"], f"{t}: split hash mismatch"
        for k in ("factor", "encoder", "tta"):
            assert man["config"][k] == getattr(cfg, k), f"{t}: config {k} differs"
        z = np.load(OUT / a.name / f"{t}_probs.npz")
        lowres.update({k: z[k] for k in z.files})
    probs = _LazyUp(lowres)
    best = None
    grid = engine.score_grid(probs, (0.4, 0.5, 0.6, 0.7, 0.8, 0.9), (50, 100, 200, 400, 800, 1200))
    for (thr, ma), st in grid.items():
        logger.info("thr={} min_area={} PQ={:.4f}", thr, ma, st.pq)
        if best is None or st.pq > best[0]:
            best = (st.pq, thr, ma)
    if best[1] in (0.4, 0.9) or best[2] in (50, 1200):
        logger.warning("best is on the grid edge; widen the search range")
    (OUT / a.name / "post.json").write_text(json.dumps(
        {"pq": best[0], "thr": best[1], "min_area": best[2], "split": sp["hash"], "folds": a.targets,
         "tuned_on_outer_folds": True}))
    logger.info("best {}", best)


def _probs_path(a, t):
    return OUT / a.name / (f"{t}_probs.npz" if not a.tag else f"{t}_probs_{a.tag}.npz")


def _load_lowres(a):
    lowres = {}
    for t in a.targets:
        z = np.load(_probs_path(a, t))
        lowres.update({k: z[k] for k in z.files})
    return lowres


def cmd_report(a, cfg):
    """fold評価の確率から、固定した後処理のPQと、グリッド探索の最良PQ（楽観的）をJSONへ出す。"""
    lowres = _load_lowres(a)
    probs = _LazyUp(lowres)
    grid = engine.score_grid(probs, (0.4, 0.5, 0.6, 0.7, 0.8), (100, 200, 400, 800))
    fixed = grid[(0.5, 400)]
    best = max(grid.items(), key=lambda kv: kv[1].pq)
    out = {"name": a.name, "targets": a.targets, "n_images": len(lowres),
           "fixed_thr0.5_area400": {"pq": fixed.pq, "sq": fixed.sq, "rq": fixed.rq, "tp": fixed.tp, "fp": fixed.fp, "fn": fixed.fn},
           "tuned_best": {"thr": best[0][0], "min_area": best[0][1], "pq": best[1].pq, "sq": best[1].sq, "rq": best[1].rq},
           "grid": {f"{k[0]}_{k[1]}": v.pq for k, v in grid.items()}}
    out["tag"] = a.tag
    (OUT / a.name / f"report_{'_'.join(a.targets)}{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps(out, indent=1))
    logger.info("report {} fixed PQ {:.4f} (sq {:.3f} rq {:.3f}) tuned {:.4f}@{}", a.name, fixed.pq, fixed.sq, fixed.rq, best[1].pq, best[0])


def cmd_infer(a, cfg):
    """保存済み重みから、foldの評価画像の確率を再計算して保存（tile/overlap/重み/TTAの比較用）。学習はしない。

    例: infer --name exp019_full --targets fold0 fold1 --tag tile512o50cos -o infer='{mode: tile, overlap: 0.5, weight: cos}' -o tta_set=d4
    """
    import time

    sp = _split()
    dev = engine.device()
    for t in a.targets:
        man = json.loads((OUT / a.name / f"{t}.manifest.json").read_text())
        assert man["split"] == sp["hash"], f"{t}: split hash mismatch"
        for k in ("factor", "encoder"):
            assert man["config"][k] == getattr(cfg, k), f"{t}: config {k} differs"
        val = sp["folds"][int(t[4:])]["eval"]
        assert not set(man["fit"]) & set(val), "fit and eval overlap"
        models = engine.load_models([OUT / a.name / f"{t}.pt"], cfg, dev)
        t0 = time.time()
        probs = {s: engine.predict(models, cache_mod.load_cfg(s, cfg)[0], dev, cfg).astype(np.float32) for s in val}
        np.savez_compressed(_probs_path(a, t), **probs)
        logger.info("infer {} {} tag={} n={} {:.0f}s infer={} tta={}", a.name, t, a.tag, len(val), time.time() - t0, cfg.infer, cfg.tta_set)


def cmd_post(a, cfg):
    """後処理候補の比較。--post-file のYAML（名前 -> instancesの引数）を、保存済み確率へ一括適用してPQを出す。"""
    from . import analysis as an

    lowres = _load_lowres(a)
    cands = yaml.safe_load(Path(a.post_file).read_text())
    gts = engine.gt_rles(list(lowres))
    res = {}
    for name, post in cands.items():
        rows = an.analyze(_LazyUp(lowres), gts, post)
        rows.pop("per_image")
        res[name] = rows
        logger.info("post {:<22s} PQ {:.4f} SQ {:.3f} RQ {:.3f} tp={} fp={} fn={} split={} merge={}", name, rows["pq"], rows["sq"],
                    rows["rq"], rows["tp"], rows["fp"], rows["fn"], rows["split_gt"], rows["merge_pred"])
    (OUT / a.name / f"post_{'_'.join(a.targets)}{'_' + a.tag if a.tag else ''}.json").write_text(json.dumps(res, indent=1))


def cmd_analyze(a, cfg):
    """誤り分析: 面積別検出率・分断/誤結合をJSONへ、画像ごとの比較図（入力/正解/確率/一致・見逃し・余分）をPNGへ。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import ndimage as ndi

    from . import analysis as an
    from .metrics import encode
    from .postprocess import instances

    lowres = _load_lowres(a)
    post = json.loads(a.post)
    gts = engine.gt_rles(list(lowres))
    res = an.analyze(_LazyUp(lowres), gts, post)
    per = res["per_image"]
    tag = "_".join(a.targets) + (f"_{a.tag}" if a.tag else "")
    (OUT / a.name / f"analysis_{tag}.json").write_text(json.dumps(res, indent=1))
    order = sorted(per, key=lambda k: per[k]["pq"])
    pick = order[: a.cases // 2] + order[len(order) // 2 - 1 : len(order) // 2] + order[-(a.cases - a.cases // 2 - 1) :]
    fig, ax = plt.subplots(len(pick), 4, figsize=(16, 4 * len(pick)))
    from pycocotools import mask as mu

    for r, s in enumerate(pick):
        img, _ = cache_mod.load_cfg(s, cfg)
        p = engine.upsample(lowres[s])
        pred = instances(p, **post)
        g = np.zeros(p.shape, bool)
        for m in mu.decode(gts[s][0]).transpose(2, 0, 1) if gts[s][0] else []:
            g |= m.astype(bool)
        pm = pred.any(0) if len(pred) else np.zeros(p.shape, bool)
        ds = lambda x: x[::2, ::2]  # noqa: E731
        ax[r, 0].imshow(img, cmap="gray"); ax[r, 0].set_title(f"{s[-24:]} PQ {per[s]['pq']:.2f} tp{per[s]['tp']} fp{per[s]['fp']} fn{per[s]['fn']}", fontsize=8)  # noqa: E702
        ov = np.dstack([ds(np.kron(img, np.ones((2, 2)))) / 255] * 3)
        ov = ov.copy()
        ov[..., 1] = np.clip(ov[..., 1] + 0.6 * ds(g), 0, 1)
        ax[r, 1].imshow(ov); ax[r, 1].set_title("input + GT set0 (green)", fontsize=8)  # noqa: E702
        ax[r, 2].imshow(ds(p), cmap="magma", vmin=0, vmax=1); ax[r, 2].set_title("probability", fontsize=8)  # noqa: E702
        tp_, fn_, fp_ = ds(g & pm), ds(g & ~pm), ds(pm & ~g)
        vis = np.zeros((*tp_.shape, 3)); vis[tp_] = (0, 1, 0); vis[fn_] = (1, 0.2, 0.2); vis[fp_] = (0.2, 0.5, 1)  # noqa: E702
        ax[r, 3].imshow(vis); ax[r, 3].set_title("green=match red=missed blue=extra (pixel level)", fontsize=8)  # noqa: E702
        for q in ax[r]: q.set_xticks([]); q.set_yticks([])  # noqa: E702
    fig.tight_layout()
    fig.savefig(OUT / a.name / f"cases_{tag}.png", dpi=60)
    logger.info("analysis PQ {:.4f} by_area {} split={} merge={}", res["pq"],
                {k: (v["gt"], v["detected"], v["false_positive_pred"]) for k, v in res["by_area"].items()}, res["split_gt"], res["merge_pred"])


def _predict_masks(stems_paths, models, cfg, dev, post):
    for stem, path in stems_paths:
        img = read_gray(path)
        small = cache_mod.downscale(img, cfg.factor)
        p = engine.upsample(engine.predict(models, small, dev, cfg))
        from .postprocess import instances
        yield stem, instances(p, post["thr"], post["min_area"], post.get("min_peak", 0.0)), p


def cmd_predict(a, cfg):
    """--split test でsubmission.csv、--split holdout で最終holdoutのPQを一度だけ計算。"""
    post = json.loads((OUT / a.name / "post.json").read_text())
    assert post["split"] == _split()["hash"], "post.json was tuned on a different split"
    dev = engine.device()
    paths = [Path(p) for p in a.weights]
    models = engine.load_models(paths, cfg, dev)
    sp = _split()
    for p in paths:  # 重みの来歴を照合: holdout評価ではholdout/bufferを学習した重みを拒否
        man = json.loads(p.with_suffix(".manifest.json").read_text())
        assert man["split"] == sp["hash"], f"{p}: split hash mismatch"
        for k in ("factor", "encoder", "tta"):  # 学習時と推論時の前処理・TTAの不一致を拒否
            if man["config"][k] != getattr(cfg, k):
                raise SystemExit(f"{p}: config {k}={getattr(cfg, k)} differs from training ({man['config'][k]})")
        if a.split == "holdout" and set(man["fit"]) & (set(sp["holdout"]) | set(sp["holdout_buffer"])):
            raise SystemExit(f"{p} was trained on holdout/buffer images; refuse to score holdout")
    if a.split == "test":
        files = list_test_images()
        ids = [f.stem for f in files]
        dst = OUT / a.name / "submission.csv"
        n = submission.write_stream(
            ((stem, m) for stem, m, _ in _predict_masks([(f.stem, f) for f in files], models, cfg, dev, post)),
            dst, ids)
        logger.info("verify {}", submission.verify(dst, ids))
        logger.info("wrote {} rows -> {}", n, dst)
    else:
        stems = sp["holdout"]
        probs = {s: engine.upsample(engine.predict(models, cache_mod.load_cfg(s, cfg)[0], dev, cfg))
                 for s in stems}
        st = engine.score_probs(probs, post["thr"], post["min_area"], post.get("min_peak", 0.0))
        logger.info("holdout PQ {:.4f} SQ {:.4f} RQ {:.4f} tp={} fp={} fn={}", st.pq, st.sq, st.rq, st.tp, st.fp, st.fn)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["split", "cache", "train", "tune", "predict", "report", "infer", "post", "analyze"])
    ap.add_argument("--config")
    ap.add_argument("--name", default="exp001")
    ap.add_argument("--target", default="fold0")
    ap.add_argument("--targets", nargs="*", default=[f"fold{i}" for i in range(5)])
    ap.add_argument("--weights", nargs="*", default=[])
    ap.add_argument("--tag", default="", help="確率ファイルの識別子（推論設定ごと）")
    ap.add_argument("--post-file", default="configs/filament/post_candidates.yaml")
    ap.add_argument("--post", default='{"thr": 0.5, "min_area": 400}')
    ap.add_argument("--cases", type=int, default=6)
    ap.add_argument("--split", default="test", choices=["test", "holdout"])
    ap.add_argument("-o", "--override", action="append", default=[])
    a = ap.parse_args(argv)
    cfg = _cfg(a.config, a.override)
    globals()[f"cmd_{a.command}"](a, cfg)


if __name__ == "__main__":
    main()
