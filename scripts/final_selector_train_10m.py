"""10 モデル(seed 42 の EMA 5 本 + seed 43 の EMA 5 本)用の最終選別器を、seed 42・43 の内側調整画像の確率の平均(5 fold、496 行)で学習する。CPUのみ、テストの確率は使わない。
現行の final_selector.pkl は上書きしない。τ は物理画像(同じ stem)単位の4群 GroupKFold(audit_tau_regroup.py の V1)で選び、従来の元 fold 単位(V0)も併記する。
出力: outputs/filament/native_remote/candidate_select_10m/final_selector_10m.{pkl,json}"""
import hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import candidate_select as cs

OUT = cs.NR / "candidate_select_10m"; AUD = cs.NR / "candidate_select_audit"; F = [0, 1, 2, 3, 4]
tune = {k: pickle.load(open(AUD / f"seedexp_nodes_f{k}_tune_avg.pkl", "rb")) for k in F}
pool = {f"f{k}:{s}": v for k in F for s, v in tune[k].items()}; ps = sorted(pool); stem = lambda p: p.split(":", 1)[1]
def curve_of(splits):
    sc = {}
    for tr, te in splits: sc.update(cs.score_nodes(cs.fit_models(*cs.make_xy(pool, tr, True)), pool, te))
    assert set(sc) == set(pool)
    c = {t: round(cs.pq_of(cs.pq_rows(pool, cs.select_C(pool, sc, t))), 4) for t in cs.TAUS}; return c, max(cs.TAUS, key=lambda t: (c[t], -t))
v0 = [([p for p in ps if not p.startswith(f"f{h}:")], [p for p in ps if p.startswith(f"f{h}:")]) for h in F]
v1 = [([ps[i] for i in a], [ps[i] for i in b]) for a, b in GroupKFold(n_splits=4).split(ps, groups=[stem(p) for p in ps])]
for tr, te in v1: assert not ({stem(p) for p in tr} & {stem(p) for p in te})
c0, t0 = curve_of(v0); c1, t1 = curve_of(v1); tau = t1
mr, mq = cs.fit_models(*cs.make_xy(pool, ps, True)); f = OUT / "final_selector_10m.pkl"
pickle.dump({"mr": mr, "mq": mq, "tau": tau, "features": ["log1p_area", "thr", "peak", "mean", "std", "area_over_parent", "n_children", "max_child_area_ratio", "aspect", "fill", "rho"], "level_thr": 0.4}, open(f, "wb"))
info = {"tau": tau, "tau_selection": "physical-image GroupKFold(4) (V1)", "tau_V0_original_fold_unit": t0, "curve_V1": c1, "curve_V0": c0, "n_rows": len(pool), "n_images": len({stem(p) for p in pool}),
        "A2_pooled_tune_pq(avg tune)": round(cs.pq_of(cs.pq_rows(pool, cs.select_A(pool, {"min_area": 400, "min_peak": 0.995}))), 4), "sha256_pkl": hashlib.sha256(f.read_bytes()).hexdigest(), "sklearn": __import__("sklearn").__version__,
        "training_probs": "mean of seed42 inner model and seed43 inner model tune probabilities (early12000), 5 folds"}
(OUT / "final_selector_10m.json").write_text(json.dumps(info, indent=1, ensure_ascii=False)); print(json.dumps({k: v for k, v in info.items() if not k.startswith("curve")}, indent=1, ensure_ascii=False)); print("curve V1", c1)
