"""担当a: 外側再学習の崩れ（過剰予測）を、評価画像のラベルを使わずに点検する関数群。
入力は確率マップ（2048x2048, float32）と、run ごとに集計した統計量だけ。外側のPQは検出の閾値選びに使わない。"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi

THR, MIN_AREA, MIN_PEAK = 0.4, 200, 0.9  # 内側tuneで多くのrunが選んだ後処理（採用候補の定義に使う）


def prob_features(p: np.ndarray, thr: float = THR, min_area: int = MIN_AREA, min_peak: float = MIN_PEAK):
    """確率マップ1枚の統計量と採用マスク（thr以上の連結成分のうち面積・最大確率が条件を満たすもの）を返す。"""
    p = np.asarray(p, np.float32)
    n_pix = p.size
    lab, n = ndi.label(p > thr, structure=np.ones((3, 3)))
    if n:
        idx = np.arange(1, n + 1)
        area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
        peak = np.asarray(ndi.maximum(p, lab, idx))
    else:
        area = np.zeros(0, np.int64); peak = np.zeros(0)
    ok = (area >= min_area) & (peak >= min_peak)
    keep = np.zeros(p.shape, bool)
    if ok.any():
        keep = np.isin(lab, np.nonzero(ok)[0] + 1)
    q = np.clip(p, 1e-6, 1 - 1e-6)
    ent = -(q * np.log(q) + (1 - q) * np.log(1 - q))
    big = area[area >= 400]
    f = {
        "mean_p": float(p.mean()), "frac_gt04": float((p > thr).mean()), "frac_gt09": float((p > 0.9).mean()),
        "frac_mid": float(((p > 0.1) & (p < 0.9)).mean()), "entropy": float(ent.mean()), "pmax": float(p.max()),
        "n_comp200": int((area >= 200).sum()), "n_comp400": int((area >= 400).sum()), "n_keep": int(ok.sum()),
        "keep_area": int(area[ok].sum()), "area_gt04": int((p > thr).sum()),
        "median_keep_area": float(np.median(area[ok])) if ok.any() else 0.0,
        "n_keep_small": int((ok & (area < 800)).sum()), "n_keep_large": int((ok & (area >= 3000)).sum()),
        "hi_core_ratio": float((p > 0.9).sum() / max((p > thr).sum(), 1)),  # 採用領域のうち高信頼(>0.9)の割合
        "n_pix": n_pix,
    }
    return f, keep


def log_ratio(outer: float, inner: float, eps: float = 1e-9) -> float:
    return float(np.log((outer + eps) / (inner + eps)))


def leave_one_out_z(values: dict[str, float], key: str) -> float:
    """key 以外の run の値の平均・標準偏差に対する key の z。基準が2本未満なら nan。他のrunの情報のみ使う。"""
    others = np.array([v for k, v in values.items() if k != key and np.isfinite(v)], float)
    if len(others) < 3 or others.std(ddof=1) == 0:
        return float("nan")
    return float((values[key] - others.mean()) / others.std(ddof=1))


def robust_z(values: dict[str, float], key: str) -> float:
    """key 以外の中央値・MAD(×1.4826)に対する key の z。外れ値に強い。"""
    others = np.array([v for k, v in values.items() if k != key and np.isfinite(v)], float)
    if len(others) < 3:
        return float("nan")
    med = np.median(others); mad = np.median(np.abs(others - med)) * 1.4826
    return float((values[key] - med) / mad) if mad > 0 else float("nan")


def peer_unsupported_fraction(keep_ds: np.ndarray, peers_ds: list[np.ndarray], cover: float = 0.5) -> float:
    """あるモデルの採用マスク(縮小, 被覆率0..1)のうち、他の全モデルがどれも採用していない面積の割合。
    peers が空なら nan。『他の独立に学習したモデルの誰も予測しない領域を、このモデルだけが予測している割合』。"""
    if not peers_ds:
        return float("nan")
    mine = keep_ds >= cover
    sup = np.zeros_like(mine)
    for q in peers_ds:
        sup |= q >= cover
    a = mine.sum()
    return float((mine & ~sup).sum() / a) if a else 0.0


def mean_pairwise_dice(masks: list[np.ndarray]) -> np.ndarray:
    n = len(masks); M = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(n):
            a, b = masks[i], masks[j]
            M[i, j] = 2 * (a & b).sum() / max(a.sum() + b.sum(), 1)
    return M


def dip_depth(steps: np.ndarray, val: np.ndarray, at: int, k: int = 2) -> dict:
    """検証loss系列で『選ばれた更新数 at』が近傍よりどれだけ孤立して低いか。
    dip = 近傍(前後k点)の平均 − at の値。global_rank = at の値が全点で何番目に低いか(1=最低)。"""
    steps = np.asarray(steps); val = np.asarray(val, float)
    i = int(np.argmin(np.abs(steps - at)))
    nb = [j for j in range(max(0, i - k), min(len(val), i + k + 1)) if j != i]
    return {"val_at": float(val[i]), "dip": float(val[nb].mean() - val[i]), "rank": int((val < val[i]).sum() + 1),
            "val_min": float(val.min()), "argmin_step": int(steps[int(np.argmin(val))])}


def peer_count_ratio(counts: dict[str, np.ndarray], offset: float = 0.0) -> dict[str, float]:
    """同じ評価画像での採用成分数（画像順は共通）を、他の全 run の画像別中央値の合計で割った比。
    1.0=他と同程度、>1=このrunだけ多く予測。他 run が2本未満のときは nan。ラベルを使わない。"""
    ks = list(counts)
    out = {}
    for k in ks:
        peers = [np.asarray(counts[j], float) for j in ks if j != k]
        if len(peers) < 2:
            out[k] = float("nan"); continue
        ref = np.median(np.stack(peers), 0).sum()
        out[k] = float((np.asarray(counts[k], float).sum() + offset) / (ref + offset)) if ref + offset > 0 else float("nan")
    return out


def screen_runs(counts: dict[str, np.ndarray], thr: float) -> dict[str, dict]:
    """peer_count_ratio が thr を超える run を『過剰予測の疑い』として返す。thr は外側PQではなく、
    崩れていない別foldの run の比の分布（平均+3標準偏差など）から決めること。"""
    r = peer_count_ratio(counts)
    return {k: {"ratio": v, "flagged": bool(np.isfinite(v) and v > thr)} for k, v in r.items()}


def threshold_from_reference(ratios: list[float], n_sd: float = 3.0) -> float:
    """参照（崩れの有無を使わず、別fold・別条件で観測した比）の 平均 + n_sd × 標準偏差(ddof=1)。"""
    a = np.asarray([x for x in ratios if np.isfinite(x)], float)
    return float(a.mean() + n_sd * a.std(ddof=1)) if len(a) >= 3 else float("nan")
