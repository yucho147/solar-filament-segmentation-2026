"""保存済みの確率から、個体候補の「形と採否」を一緒に選ぶ方法の検証（CPUのみ）。事前登録: native_remote/candidate_select_preregistration.json
段階1 extract: 画像ごとに複数しきい値の8連結成分(入れ子の木)を作り、ラベルを使わない特徴と、注釈集合との対応(IoU>0.5)を保存する。
段階2 run: 内側tune(内側モデルが未学習)で r(対応率)と q(対応時IoU)を学習し、s=r_hat x q_hat。tau は内側tuneの時間順5分割の交差適合で選ぶ。外側evalには固定適用。
使い方: uv run python scripts/candidate_select.py extract | run"""
import json, pickle, sys, time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, "src")
from pycocotools import mask as mu
from sklearn.ensemble import HistGradientBoostingRegressor

from filament import engine
from filament.evalrecord import paired_block_bootstrap
from filament.post_pos import Geometry, position_values
from filament.postprocess import EIGHT

NR = Path("outputs/filament/native_remote")
OUT = NR / "candidate_select"
THR = (0.3, 0.4, 0.5, 0.6, 0.7, 0.85); LV04 = 1; MIN_AREA = 100
import os
BASE = NR / "cos_base_f01/cos_base"
BASEDIR = {0: BASE, 1: BASE, 2: NR / "cos_f2/cos_f234", 3: NR / "cos_f3/cos_f234", 4: NR / "cos_f4/cos_f234"}
NEW = {0: "ema_s42f0", 1: "ema_s42f1_run2", 2: "ema_s42f2", 3: "ema_s42f3", 4: "ema_s42f4"}
FOLDS = tuple(int(x) for x in os.environ.get("CS_FOLDS", "0,1").split(","))
TAG = os.environ.get("CS_TAG", "")
WITH_D = os.environ.get("CS_SKIP_D", "0") != "1"
EVS = tuple(os.environ.get("CS_EVS", "eval_base,eval_ema").split(","))
GEOM = Geometry()
TAUS = [round(x, 2) for x in np.arange(0.0, 0.61, 0.05)]


def sources(k):
    B = BASEDIR[k]
    return {"tune": (B / f"fold{k}/inner0/early12000/tune_probs.npz", json.load(open(B / f"fold{k}/inner0/selection.json"))["tune_ids"]),
            "eval_base": (B / f"outer/fold{k}/eval_probs.npz", sorted(np.load(B / f"outer/fold{k}/eval_probs.npz").files)),
            "eval_ema": (NR / NEW[k] / f"outer/fold{k}/ema_0.999/eval_probs.npz", sorted(np.load(B / f"outer/fold{k}/eval_probs.npz").files)),
            "eval_avg": (str(B / f"outer/fold{k}/eval_probs.npz") + "|" + str(NR / NEW[k] / f"outer/fold{k}/ema_0.999/eval_probs.npz"), sorted(np.load(B / f"outer/fold{k}/eval_probs.npz").files))}


def extract_one(a):
    npz, stem, gts = a
    prob = np.mean([np.load(x)[stem].astype(np.float32) for x in npz.split("|")], 0) if "|" in npz else np.load(npz)[stem].astype(np.float32)
    return stem, nodes_from_prob(prob, gts)


def nodes_from_prob(prob, gts):
    """確率地図 -> 候補の木(ノードのリスト)。gts=[] ならラベル対応は作らない(テスト画像用)。"""
    flat_gt = [g for s in gts for g in s]; set_of = [i for i, s in enumerate(gts) for _ in s]; gi_of = [j for s in gts for j in range(len(s))]
    buf = np.zeros(prob.shape, np.uint8, order="F")
    nodes, prev_map, prev_lab = [], {}, None
    for li, thr in enumerate(THR):
        lab, n = ndi.label(prob > thr, structure=EIGHT)
        cur_map = {}
        if n:
            idx = np.arange(1, n + 1); area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
            peak = np.asarray(ndi.maximum(prob, lab, idx)); mean = np.asarray(ndi.mean(prob, lab, idx)); std = np.asarray(ndi.standard_deviation(prob, lab, idx))
            objs = ndi.find_objects(lab)
            for i in idx:
                if area[i - 1] < MIN_AREA: continue
                sl = objs[i - 1]; m = lab[sl] == i
                yy, xx = np.nonzero(m); cy, cx = yy.mean() + sl[0].start, xx.mean() + sl[1].start
                h, w = m.shape
                parent = -1
                if li > 0 and prev_lab is not None:
                    pl = int(prev_lab[sl][m][0]); parent = prev_map.get(pl, -1)
                buf[sl] = m; rle = mu.encode(buf); buf[sl] = 0
                hits = []
                if flat_gt:
                    iou = np.asarray(mu.iou([rle], flat_gt, [0] * len(flat_gt))).ravel()
                    hits = [(set_of[t], gi_of[t], float(iou[t])) for t in np.nonzero(iou > 0.5)[0]]
                cur_map[int(i)] = len(nodes)
                nodes.append({"lab": int(i), "level": li, "parent": parent, "area": int(area[i - 1]), "peak": float(peak[i - 1]), "mean": float(mean[i - 1]), "std": float(std[i - 1]),
                              "aspect": float(max(h, w) / max(min(h, w), 1)), "fill": float(area[i - 1] / (h * w)), "cy": float(cy), "cx": float(cx), "hits": hits})
        prev_map, prev_lab = cur_map, lab
    for j, nd in enumerate(nodes): nd["children"] = []
    for j, nd in enumerate(nodes):
        if nd["parent"] >= 0: nodes[nd["parent"]]["children"].append(j)
    rho = position_values(np.array([nd["cy"] for nd in nodes]), np.array([nd["cx"] for nd in nodes]), GEOM, "rho") if nodes else []
    for j, nd in enumerate(nodes):
        p = nodes[nd["parent"]]["area"] if nd["parent"] >= 0 else nd["area"]
        ch = [nodes[c]["area"] for c in nd["children"]]
        nd["f"] = [np.log1p(nd["area"]), THR[nd["level"]], nd["peak"], nd["mean"], nd["std"], nd["area"] / p, len(ch), (max(ch) / nd["area"]) if ch else 0.0, nd["aspect"], nd["fill"], float(rho[j])]
    return {"nodes": nodes, "n_gt": [len(s) for s in gts]}


def extract():
    OUT.mkdir(parents=True, exist_ok=True)
    for k in FOLDS:
        for name, (npz, stems) in sources(k).items():
            f = OUT / f"nodes_f{k}_{name}.pkl"
            if f.exists(): print("skip", f); continue
            gts = engine.gt_rles(list(stems)); t0 = time.time()
            with Pool(4) as pool: res = dict(pool.imap_unordered(extract_one, [(str(npz), s, gts[s]) for s in stems], chunksize=1))
            pickle.dump(res, open(f, "wb")); print(k, name, len(res), "images", sum(len(v["nodes"]) for v in res.values()), "nodes", round(time.time() - t0), "s", flush=True)


# ---------- 評価 ----------
def pq_rows(data, selection):
    """selection: stem -> 採用ノード番号のリスト。画像ごとの (iou_sum, tp, fp, fn) を返す。"""
    rows = {}
    for stem, d in data.items():
        iou_sum = tp = fp = fn = 0.0
        sel = selection.get(stem, [])
        for s, n_gt in enumerate(d["n_gt"]):
            hit_g = set(); 
            for j in sel:
                hs = [h for h in d["nodes"][j]["hits"] if h[0] == s]
                if hs:
                    tp += len(hs); iou_sum += sum(h[2] for h in hs); hit_g |= {h[1] for h in hs}
                else: fp += 1
            fn += n_gt - len(hit_g)
        rows[stem] = (iou_sum, tp, fp, fn)
    return rows


def pq_of(rows):
    a = np.array(list(rows.values()), float).sum(0)
    den = a[1] + .5 * a[2] + .5 * a[3]
    return a[0] / den if den > 0 else 0.0


def totals(rows):
    a = np.array(list(rows.values()), float).sum(0); return [int(a[1]), int(a[2]), int(a[3])]


def targets(d, j):
    best = {}
    for s, g, i in d["nodes"][j]["hits"]: best[s] = max(best.get(s, 0), i)
    ns = len(d["n_gt"]); r = len(best) / ns if ns else 0.0
    return r, (float(np.mean(list(best.values()))) if best else np.nan)


def make_xy(data, stems, only04=False):
    X, R, Q = [], [], []
    for s in stems:
        for j, nd in enumerate(data[s]["nodes"]):
            if only04 and nd["level"] != LV04: continue
            r, q = targets(data[s], j); X.append(nd["f"]); R.append(r); Q.append(q)
    return np.array(X), np.array(R), np.array(Q)


def fit_models(X, R, Q):
    kw = dict(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=20, random_state=0)
    mr = HistGradientBoostingRegressor(**kw).fit(X, R); ok = ~np.isnan(Q)
    mq = HistGradientBoostingRegressor(**kw).fit(X[ok], Q[ok])
    return mr, mq


def score_nodes(models, data, stems, only04=False):
    mr, mq = models; out = {}
    for s in stems:
        nodes = data[s]["nodes"]
        if not nodes: out[s] = np.zeros(0); continue
        X = np.array([nd["f"] for nd in nodes]); out[s] = np.clip(mr.predict(X), 0, 1) * np.clip(mq.predict(X), 0, 1)
    return out


def select_C(data, scores, tau):
    return {s: [j for j, nd in enumerate(d["nodes"]) if nd["level"] == LV04 and scores[s][j] >= tau] for s, d in data.items()}


def select_D(data, scores, tau):
    sel = {}
    for s, d in data.items():
        nodes = d["nodes"]; n = len(nodes); best = [0.0] * n; take = [False] * n
        for j in range(n - 1, -1, -1):   # 子は親より後ろの番号(高いしきい値が後)
            cs = sum(best[c] for c in nodes[j]["children"]); own = scores[s][j] - tau
            if own > cs and own > 0: best[j], take[j] = own, True
            else: best[j] = cs
        chosen = []
        def walk(j):
            if take[j]: chosen.append(j); return
            for c in nodes[j]["children"]: walk(c)
        for j in range(n):
            if nodes[j]["parent"] < 0: walk(j)
        sel[s] = chosen
    return sel


def select_A(data, post):
    return {s: [j for j, nd in enumerate(d["nodes"]) if nd["level"] == LV04 and nd["area"] >= post["min_area"] and nd["peak"] >= post["min_peak"]] for s, d in data.items()}


def select_B(data, a, p, m):
    return {s: [j for j, nd in enumerate(d["nodes"]) if nd["level"] == LV04 and nd["area"] >= a and nd["peak"] >= p and nd["mean"] >= m] for s, d in data.items()}


def crossfit_scores(data, stems, only04):
    stems = sorted(stems); chunks = np.array_split(np.arange(len(stems)), 5); sc = {}
    for c in chunks:
        te = [stems[i] for i in c]; tr = [stems[i] for i in range(len(stems)) if i not in set(c)]
        sc.update(score_nodes(fit_models(*make_xy(data, tr, only04)), data, te))
    return sc


def run():
    import csv
    res = {"folds": list(FOLDS), "conditions": {}}; per_image = {}; names = ["A", "A2", "B", "C"] + (["D"] if WITH_D else [])
    for k in FOLDS:
        tune = pickle.load(open(OUT / f"nodes_f{k}_tune.pkl", "rb")); ts = sorted(tune)
        post = json.load(open(BASEDIR[k] / f"fold{k}/inner0/selection.json"))["selected"]["post"]
        t0 = time.time()
        gridB = [(a, p, m) for a in (100, 200, 400, 800) for p in (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.995) for m in (0, 0.5, 0.6, 0.7, 0.8, 0.9)]
        scB = {g: pq_of(pq_rows(tune, select_B(tune, *g))) for g in gridB}; bB = max(scB, key=scB.get)
        cfC = crossfit_scores(tune, ts, True)
        tauC = max(TAUS, key=lambda t: (round(pq_of(pq_rows(tune, select_C(tune, cfC, t))), 4), -t))
        tune_pq = {"A": pq_of(pq_rows(tune, select_A(tune, post))), "A2": pq_of(pq_rows(tune, select_A(tune, {"min_area": 400, "min_peak": 0.995}))), "B": scB[bB], "C_crossfit": pq_of(pq_rows(tune, select_C(tune, cfC, tauC)))}
        mC = fit_models(*make_xy(tune, ts, True)); info = {"post_A": post, "B": list(bB), "tau_C": tauC}
        if WITH_D:
            cfD = crossfit_scores(tune, ts, False)
            tauD = max(TAUS, key=lambda t: (round(pq_of(pq_rows(tune, select_D(tune, cfD, t))), 4), -t)); mD = fit_models(*make_xy(tune, ts, False))
            tune_pq["D_crossfit"] = pq_of(pq_rows(tune, select_D(tune, cfD, tauD))); info["tau_D"] = tauD
        info.update({"tune_pq": {x: round(v, 4) for x, v in tune_pq.items()}, "n_tune": len(ts), "n_nodes_tune": sum(len(v["nodes"]) for v in tune.values()), "seconds_fit": round(time.time() - t0)})
        res[f"fold{k}"] = {"info": info}
        for ev in EVS:
            data = pickle.load(open(OUT / f"nodes_f{k}_{ev}.pkl", "rb")); es = sorted(data)
            sels = {"A": select_A(data, post), "A2": select_A(data, {"min_area": 400, "min_peak": 0.995}), "B": select_B(data, *bB), "C": select_C(data, score_nodes(mC, data, es), tauC)}
            if WITH_D: sels["D"] = select_D(data, score_nodes(mD, data, es), tauD)
            r = {}
            for name, sel in sels.items():
                rows = pq_rows(data, sel); r[name] = {"pq": round(pq_of(rows), 4), "tp_fp_fn": totals(rows), "n_pred": int(sum(len(v) for v in sel.values()))}
                per_image[(k, ev, name)] = rows
            res[f"fold{k}"][ev] = r
            print(k, ev, {n: (v["pq"], v["tp_fp_fn"]) for n, v in r.items()}, flush=True)
    pairs = [("C", "A"), ("A2", "A"), ("B", "A2"), ("C", "A2"), ("C", "B")] + ([("D", "A")] if WITH_D else [])
    for ev in EVS:
        blocks = {}
        for k in FOLDS:
            rec = NR / (("cos_base_f01/cos_base" if k < 2 else f"cos_f{k}/cos_f234") if ev in ("eval_base", "eval_avg") else NEW[k]) / f"outer/fold{k}/per_image.csv"
            for row in csv.DictReader(open(rec)): blocks[row["stem"]] = int(row["block"])
        def rows_of(name): return [dict(stem=s, block=blocks[s], iou_sum=v[0], tp=v[1], fp=v[2], fn=v[3]) for k in FOLDS for s, v in per_image[(k, ev, name)].items()]
        pooled = {}
        for name in names:
            a = np.array([[r["iou_sum"], r["tp"], r["fp"], r["fn"]] for r in rows_of(name)]).sum(0); pooled[name] = {"pq": round(a[0] / (a[1] + .5 * a[2] + .5 * a[3]), 4), "tp_fp_fn": [int(x) for x in a[1:]]}
        diffs = {}
        for x, y in pairs:   # x - y
            b = paired_block_bootstrap(rows_of(y), rows_of(x), n_boot=5000, seed=0)
            diffs[f"{x} - {y}"] = {"diff": round(b["diff_b_minus_a"], 4), "ci95": [round(v, 4) for v in b["diff_ci"]], "per_fold": {str(k): round(res[f"fold{k}"][ev][x]["pq"] - res[f"fold{k}"][ev][y]["pq"], 4) for k in FOLDS}}
        res["conditions"][ev] = {"pooled": pooled, "diff": diffs}
    (OUT / f"candidate_select_results{TAG}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    for ev, v in res["conditions"].items():
        print(ev, "合算", {n: x["pq"] for n, x in v["pooled"].items()}); [print("   ", n, d) for n, d in v["diff"].items()]


if __name__ == "__main__":
    {"extract": extract, "run": run}[sys.argv[1]]()
