"""物理正規化（exp029）: 円盤の限界減光と、ゆっくり変わる背景を画像1枚の画素だけから除き、フィラメントの対比を取り出す。

モデル: I(x) = I_clv(ρ) · B(x) · (1 + c(x))
  - ρ: 円盤中心からの距離（画素、実数）
  - I_clv(ρ): 限界減光（円盤の中心から縁へ向かう明るさの系統的変化）。ρごとの方位角中央値で推定
  - B(x): 雲・勾配・装置段差などゆっくり変わる背景。I/I_clv を、円盤内の外れ値を除いた正規化畳み込み
    （重み付きガウス平滑化）で推定
  - c(x): 対比。出力 u = clip(round(OUT_BASE · (1 + c)), 1, 255)、円盤外は 0

守っていること:
  - 画像の画素だけを使う。ファイル名・観測所・他画像・ラベルを一切使わない（推論時に1枚で再現できる）。
  - 画素の位置を動かさない。画像の再標本化（補間・縮小拡大）をしない。円盤中心が整数でなくても、ρを実数で
    計算して各画素にそのまま対応づける（光線追跡の円盤端の推定だけは線形補間で画素値を読むが、これは幾何の推定用で出力には現れない）。
  - 決定的（乱数なし）。冪等ではない: 2回かけると、1回目の出力を入力として限界減光・背景を再推定するため、
    出力はほぼ128に平坦な画像のまま概ね変わらないが、画素値は完全一致しない（正規化済み画像への再適用は想定しない）。
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, map_coordinates

SIZE = 2048

# 実験前に固定した値（outputs/filament/exp029/preregistration.json に同じ内容を記録）
PARAMS = {
    "geometry": "auto",           # auto=画像ごと推定（失敗時は固定値）| fixed=公開著者報告の固定値
    "fixed_center_xy": (1023.1, 1024.1),
    "fixed_radius": 899.7,
    "n_rays": 360,
    "ray_parity": None,           # 通常はNone（全光線）。0/1は幾何の安定性検証専用
    "ray_halfwidth": 60.0,        # 1回目: 画像中心・固定半径の周り±この範囲で円盤端を探す
    "ray_halfwidth2": 12.0,       # 2回目: 1回目のフィット円の周り
    "ray_step": 0.5,
    "edge_sigma": 1.0,            # 光線方向の平滑化（サンプル数。0.5px刻みなので約0.5px）
    "min_edge_drop": 20.0,        # 円盤端の内外（各約5画素の平均）の明るさ差の下限（画素値）
    "fit_min_inliers": 180,
    "fit_radius_range": (850.0, 950.0),
    "fit_max_center_offset": 40.0,
    "fit_max_rms": 2.0,
    "edge_pad": 0.5,              # ρ <= R + edge_pad を円盤内とする
    "clv_smooth_bins": 2.0,       # 限界減光（中央値の曲線）の平滑化。円盤端から8画素内は平滑化しない
    "clv_raw_edge_bins": 8,
    "bg_sigma": 64.0,             # 背景のガウス幅（画素）
    "bg_iters": 2,                # 外れ値除外の反復回数
    "bg_lo": 0.8,                 # 背景に対する比がこの範囲外の画素は背景推定から除く
    "bg_hi": 1.25,
    "edge_neutral": 2.0,          # ρ > R - この値 の円盤内画素は出力を out_base（中立値）に固定。縁の急な段差（暗い画素列→明るい輪）は
                                  # 方位角ごとの位置ずれで限界減光の補正が合わず、この帯の標準偏差が内部の約8倍になるため
    "out_base": 128.0,            # 出力 = out_base * I / (I_clv * B)  （円盤内の中央値が約128）
}


def _fit_circle(x: np.ndarray, y: np.ndarray, p: dict) -> tuple[float, float, float, int, float]:
    """代数的最小二乗円フィットを、残差の外れ値（3×MAD、下限2画素）を除きながら繰り返す。"""
    keep = np.ones(len(x), bool)
    for _ in range(8):
        a = np.c_[2 * x[keep], 2 * y[keep], np.ones(keep.sum())]
        sol, *_ = np.linalg.lstsq(a, x[keep] ** 2 + y[keep] ** 2, rcond=None)
        cx, cy = sol[0], sol[1]
        r = np.sqrt(max(sol[2] + cx**2 + cy**2, 0.0))
        res = np.hypot(x - cx, y - cy) - r
        s = max(2.0, 3 * 1.4826 * float(np.median(np.abs(res[keep] - np.median(res[keep])))))
        new = np.abs(res) <= s
        if new.sum() < 10 or (new == keep).all():
            break
        keep = new
    rms = float(np.sqrt(np.mean(res[keep] ** 2)))
    return float(cx), float(cy), float(r), int(keep.sum()), rms


def _trace(f: np.ndarray, cx: float, cy: float, r0: float, halfwidth: float, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """中心(cx,cy)から360方向の光線上で、円盤端（外向きに最も急に明るくなる位置）を探す。

    この画像の円盤端は、内側の暗い画素列のすぐ外側に明るい輪（縁の明るい線、画素値200前後）が始まる段差。
    その外向きの最大勾配位置を端とする。端の内外の明るさ差が小さい方向は捨てる。画素値は線形補間で読む（幾何の推定専用）。"""
    th = np.linspace(0, 2 * np.pi, p["n_rays"], endpoint=False)
    rr = np.arange(r0 - halfwidth, r0 + halfwidth + 1e-6, p["ray_step"])
    xs = cx + np.cos(th)[:, None] * rr[None]
    ys = cy + np.sin(th)[:, None] * rr[None]
    inside = (xs >= 0) & (xs <= SIZE - 1) & (ys >= 0) & (ys <= SIZE - 1)
    prof = map_coordinates(f, [ys.ravel(), xs.ravel()], order=1, mode="nearest").reshape(xs.shape)
    sm = gaussian_filter1d(prof, p["edge_sigma"], axis=1)
    d = np.gradient(sm, p["ray_step"], axis=1)
    d[~inside] = 0
    k = d.argmax(1)
    pad = int(round(6 / p["ray_step"]))
    gap = int(round(1 / p["ray_step"]))
    ex, ey = [], []
    for i in np.nonzero(inside.all(1))[0]:
        if p.get("ray_parity") is not None and i % 2 != p["ray_parity"]:
            continue  # 検証用: 偶数/奇数番の光線だけで推定し、2つの推定の一致を見る
        ki = k[i]
        if ki - pad < 1 or ki + pad >= len(rr) - 1:
            continue
        if sm[i, ki + gap:ki + pad].mean() - sm[i, ki - pad:ki - gap].mean() < p["min_edge_drop"]:
            continue
        y0, y1, y2 = d[i, ki - 1], d[i, ki], d[i, ki + 1]
        den = y0 - 2 * y1 + y2
        off = 0.0 if abs(den) < 1e-9 else 0.5 * (y0 - y2) / den
        r_edge = rr[ki] + np.clip(off, -1, 1) * p["ray_step"]
        ex.append(cx + np.cos(th[i]) * r_edge)
        ey.append(cy + np.sin(th[i]) * r_edge)
    return np.array(ex), np.array(ey)


def estimate_disc(img: np.ndarray, params: dict | None = None) -> dict:
    """光線追跡で円盤端を探し、円をロバストにフィットする。1回目は画像中心・固定半径を事前値に±ray_halfwidth、
    2回目は1回目のフィット円の周り±ray_halfwidth2で探し直す。
    戻り値: cx, cy（画素index座標）, r, n_valid（端を検出できた方向数）, n_inliers, rms, ok, reason。
    ok=False でも推定値は返す（失敗の診断用）。"""
    p = {**PARAMS, **(params or {})}
    f = img.astype(np.float32)
    cx, cy, r = SIZE / 2, SIZE / 2, p["fixed_radius"]
    n_valid = ni = 0
    rms = float("nan")
    for hw in (p["ray_halfwidth"], p["ray_halfwidth2"]):
        ex, ey = _trace(f, cx, cy, r, hw, p)
        n_valid = len(ex)
        if n_valid < (30 if p.get('ray_parity') is None else 15):
            return {"cx": float("nan"), "cy": float("nan"), "r": float("nan"), "n_valid": n_valid, "n_inliers": 0,
                    "rms": float("nan"), "ok": False, "reason": "few_edges"}
        cx, cy, r, ni, rms = _fit_circle(ex, ey, p)
    reason = ""
    lo, hi = p["fit_radius_range"]
    if ni < p["fit_min_inliers"]:
        reason = "few_inliers"
    elif not lo <= r <= hi:
        reason = "radius_out_of_range"
    elif np.hypot(cx - SIZE / 2, cy - SIZE / 2) > p["fit_max_center_offset"]:
        reason = "center_out_of_range"
    elif rms > p["fit_max_rms"]:
        reason = "large_residual"
    return {"cx": cx, "cy": cy, "r": r, "n_valid": n_valid, "n_inliers": ni, "rms": rms,
            "ok": reason == "", "reason": reason}


def _geometry(img: np.ndarray, p: dict) -> dict:
    fx, fy = p["fixed_center_xy"]
    fixed = {"cx": fx, "cy": fy, "r": p["fixed_radius"], "ok": True, "reason": "fixed", "used": "fixed"}
    if p["geometry"] == "fixed":
        return fixed
    g = estimate_disc(img, p)
    if g["ok"]:
        return {**g, "used": "estimated"}
    return {**fixed, "used": "fixed_fallback", "reason": g["reason"], "estimate": g}


def radial_profile(img: np.ndarray, rho: np.ndarray, r_edge: float, p: dict) -> np.ndarray:
    """ρを1画素幅のビンに分け、各ビンの方位角中央値を返す（円盤内のみ。ビン数 = floor(R+pad)+1）。"""
    nb = int(np.floor(r_edge + p["edge_pad"])) + 1
    sel = rho <= r_edge + p["edge_pad"]
    b = np.floor(rho[sel]).astype(np.int64)
    v = img[sel].astype(np.float32)
    order = np.argsort(b, kind="stable")
    b, v = b[order], v[order]
    bounds = np.searchsorted(b, np.arange(nb + 1))
    med = np.empty(nb, np.float32)
    for k in range(nb):
        lo, hi = bounds[k], bounds[k + 1]
        med[k] = np.median(v[lo:hi]) if hi > lo else (med[k - 1] if k else 1.0)
    sm = gaussian_filter1d(med, p["clv_smooth_bins"], mode="nearest")
    cut = max(nb - p["clv_raw_edge_bins"], 1)
    out = med.copy()
    out[:cut] = sm[:cut]
    return np.maximum(out, 1.0)


def _norm_conv(q: np.ndarray, w: np.ndarray, sigma: float) -> np.ndarray:
    num = gaussian_filter(q * w, sigma, mode="constant", truncate=3.0)
    den = gaussian_filter(w, sigma, mode="constant", truncate=3.0)
    return np.where(den > 1e-3, num / np.maximum(den, 1e-3), 1.0).astype(np.float32)


def normalize_with_info(img: np.ndarray, params: dict | None = None) -> tuple[np.ndarray, dict]:
    """正規化画像uint8[2048,2048]と、診断情報（幾何・限界減光・背景の範囲）を返す。"""
    p = {**PARAMS, **(params or {})}
    if img.shape != (SIZE, SIZE) or img.dtype != np.uint8:
        raise ValueError(f"expected uint8[{SIZE},{SIZE}], got {img.dtype}{img.shape}")
    geo = _geometry(img, p)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    rho = np.hypot(xx - np.float32(geo["cx"]), yy - np.float32(geo["cy"]))
    disc = rho <= geo["r"] + p["edge_pad"]
    prof = radial_profile(img, rho, geo["r"], p)
    clv = np.ones((SIZE, SIZE), np.float32)
    clv[disc] = prof[np.minimum(np.floor(rho[disc]).astype(np.int64), len(prof) - 1)]
    f = img.astype(np.float32)
    q = np.where(disc, f / clv, 0.0).astype(np.float32)
    w = disc.astype(np.float32)
    bg = _norm_conv(q, w, p["bg_sigma"])
    for _ in range(p["bg_iters"]):
        ratio = q / np.maximum(bg, 1e-3)
        w = (disc & (ratio >= p["bg_lo"]) & (ratio <= p["bg_hi"])).astype(np.float32)
        bg = _norm_conv(q, w, p["bg_sigma"])
    u = p["out_base"] * q / np.maximum(bg, 1e-3)
    out = np.zeros((SIZE, SIZE), np.uint8)
    out[disc] = np.clip(np.rint(u[disc]), 1, 255).astype(np.uint8)
    if p["edge_neutral"] > 0:
        out[disc & (rho > geo["r"] - p["edge_neutral"])] = int(round(p["out_base"]))
    info = {"geometry": {k: v for k, v in geo.items() if k != "estimate"}, "bg_min": float(bg[disc].min()),
            "bg_max": float(bg[disc].max()), "clv_center": float(prof[0]), "clv_edge": float(prof[-1]),
            "disc_pixels": int(disc.sum())}
    return out, info


def normalize_image(img: np.ndarray, params: dict | None = None) -> np.ndarray:
    """画像1枚（uint8[2048,2048]）→ 正規化画像（uint8[2048,2048]）。将来のテスト画像にも同じ処理を適用できる単独の口。"""
    return normalize_with_info(img, params)[0]


def disc_mask(img: np.ndarray, params: dict | None = None) -> tuple[np.ndarray, dict]:
    """推定した円盤内のマスク(bool)と幾何。解析用。"""
    p = {**PARAMS, **(params or {})}
    geo = _geometry(img, p)
    yy, xx = np.mgrid[0:SIZE, 0:SIZE].astype(np.float32)
    rho = np.hypot(xx - np.float32(geo["cx"]), yy - np.float32(geo["cy"]))
    return rho <= geo["r"] + p["edge_pad"], geo


# ------------------------------------------------------------------ 正規化cache（x1n）の生成・検査
def _target_native(pi) -> np.ndarray:
    """cache.build と同じ式の原寸の平均ソフト教師 uint8[2048,2048]（x1に無い画像用。x1は変更しない）。"""
    per_set = []
    for s in pi.sets:
        m = s.instance_masks()
        per_set.append(m.max(0) if len(m) else np.zeros((SIZE, SIZE), np.uint8))
    return (np.mean(per_set, 0) * 255).astype(np.uint8)


def _build_one(args) -> dict:
    stem, file_name, pi, root, cache_root, params = args
    from pathlib import Path

    from .data import read_gray

    dst = Path(cache_root) / "x1n" / f"{stem}.npz"
    if dst.exists():
        return {"stem": stem, "skipped": True}
    img = read_gray(Path(root) / "train/train_images" / file_name)
    x1 = Path(cache_root) / "x1" / f"{stem}.npz"
    target = np.load(x1)["target"] if x1.exists() else _target_native(pi)
    out, info = normalize_with_info(img, params)
    tmp = dst.with_name(dst.name + ".tmp.npz")
    np.savez(tmp, image=out, target=target)
    tmp.replace(dst)
    g = info["geometry"]
    return {"stem": stem, "skipped": False, "cx": g["cx"], "cy": g["cy"], "r": g["r"], "used": g["used"],
            "reason": g.get("reason", ""), "bg_min": info["bg_min"], "bg_max": info["bg_max"]}


def build_cache(root=None, cache_root=None, workers: int = 6, limit: int | None = None,
                stems: set[str] | None = None, params: dict | None = None) -> list[dict]:
    """学習画像を正規化した原寸cache `<cache_root>/x1n/<stem>.npz`（image=正規化uint8, target=x1と同一）を作る。
    既存ファイルは作り直さない。x1・x2は読むだけで変更しない。診断（幾何など）のリストを返す。"""
    import json
    from multiprocessing import Pool

    from . import cache as cache_mod
    from .data import ROOT, load_train

    root = root or ROOT
    cache_root = cache_root or cache_mod.CACHE
    (cache_root / "x1n").mkdir(parents=True, exist_ok=True)
    jobs = []
    for n, (name, pi) in enumerate(sorted(load_train(root).items())):
        if limit and n >= limit:
            break
        if stems is not None and pi.stem not in stems:
            continue
        jobs.append((pi.stem, name, pi, str(root), str(cache_root), params))
    if workers <= 1:
        res = [_build_one(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            res = list(pool.imap(_build_one, jobs, chunksize=2))
    import hashlib
    from pathlib import Path

    man = {"params": {**PARAMS, **(params or {})}, "physnorm_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "n_images": len(jobs), "n_new": sum(not r["skipped"] for r in res)}
    (cache_root / "x1n" / "_manifest.json").write_text(json.dumps(man, ensure_ascii=False, indent=1, default=list))
    return res


def verify_x1n(stem: str, cache_root=None, params: dict | None = None) -> dict:
    """x1n の画像が「x1 の画像（JPEGデコード値）に normalize_image を適用した結果」と一致し、教師が x1 と同一で、
    形状・dtype・円盤外=0 を満たすことを検査する（不一致は例外）。"""
    from . import cache as cache_mod

    cache_root = cache_root or cache_mod.CACHE
    z = np.load(cache_root / "x1n" / f"{stem}.npz")
    img, tgt = np.load(cache_root / "x1" / f"{stem}.npz")["image"], np.load(cache_root / "x1" / f"{stem}.npz")["target"]
    if set(z.files) != {"image", "target"}:
        raise AssertionError(f"{stem}: unexpected arrays {z.files}")
    if z["image"].shape != (SIZE, SIZE) or z["image"].dtype != np.uint8:
        raise AssertionError(f"{stem}: bad image shape/dtype")
    if not np.array_equal(z["target"], tgt):
        raise AssertionError(f"{stem}: target differs from x1")
    if not np.array_equal(z["image"], normalize_image(img, params)):
        raise AssertionError(f"{stem}: image differs from normalize_image(x1 image)")
    return {"stem": stem, "outside_zero_pixels": int((z["image"] == 0).sum())}
