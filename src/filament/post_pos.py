"""円盤上の位置（円盤中心からの距離、縦位置）の帯ごとに後処理を変える（exp026-diag / exp027）。

位置の量（画像のみから決まる。ファイル名の時刻・観測所は使わない）:
- rho = 円盤中心からの距離 / 円盤半径（0〜1。1 が縁）。
- lat = |y - cy| / R。中心を通る水平線からの縦位置。画像の上が太陽の北であることは未検証なので「緯度の代理」。

後処理の方式（exp027 で採用した方）:
  成分化は全体で共通のしきい値 thr で1回だけ行い、成分の重心が入る帯ごとに min_area / min_peak を変える。
  理由: 画素の位置でしきい値を変えると、帯の境界をまたぐ1本の暗い筋が人工的に切れる・つながるため、
  帯の境界で個体の形が変わる。重心の帯で絞る方式は個体の形を変えず、帯ごとの採否だけが変わる。
  instances_banded は参考に、画素の位置でしきい値を変える方式（params の thr が帯で異なる場合）にも対応する。
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .postprocess import EIGHT

# 実測（学習707枚+テスト180枚の中央値、scripts/disc_check.py、exp026_diag/disc_geometry.json）。
# 円盤の位置・大きさは全画像でほぼ固定（中心の標準偏差0.4画素、半径1画素、最大ずれ3.2画素）。半径は縁の最急下降点で、
# 公開Notebook著者の報告値（中心 (1023.1, 1024.1)、半径約 899.7）より4画素大きい（縁の定義の差）。
CENTER_X, CENTER_Y, RADIUS = 1023.0, 1024.0, 904.0


@dataclass(frozen=True)
class Geometry:
    cx: float = CENTER_X
    cy: float = CENTER_Y
    radius: float = RADIUS


@dataclass(frozen=True)
class Bands:
    """位置の量 quantity ('rho' | 'lat') を edges（昇順の内側境界）で区切った帯。帯の数は len(edges)+1。
    帯 i は edges[i-1] <= 値 < edges[i]（最初は下限なし、最後は上限なし）。"""

    quantity: str
    edges: tuple[float, ...]

    def __post_init__(self):
        if self.quantity not in ("rho", "lat"):
            raise ValueError(f"unknown quantity {self.quantity}")
        if list(self.edges) != sorted(set(self.edges)):
            raise ValueError("edges must be strictly increasing")

    @property
    def n_bands(self) -> int:
        return len(self.edges) + 1

    def index(self, values: np.ndarray) -> np.ndarray:
        return np.digitize(values, np.asarray(self.edges, float))  # 境界ちょうどは上の帯


def position_values(y, x, geom: Geometry = Geometry(), quantity: str = "rho"):
    """画素/重心座標 (行 y, 列 x) の位置の量。rho = 距離/R、lat = |y-cy|/R。"""
    y, x = np.asarray(y, float), np.asarray(x, float)
    if quantity == "rho":
        return np.hypot(x - geom.cx, y - geom.cy) / geom.radius
    if quantity == "lat":
        return np.abs(y - geom.cy) / geom.radius
    raise ValueError(quantity)


def band_map(shape: tuple[int, int], bands: Bands, geom: Geometry = Geometry()) -> np.ndarray:
    """各画素の帯番号 int8[H,W]。"""
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    return bands.index(position_values(yy, xx, geom, bands.quantity)).astype(np.int8)


def instances_banded(prob: np.ndarray, bands: Bands, params: list[dict], geom: Geometry = Geometry()) -> np.ndarray:
    """prob: float[H,W]。params: 帯ごとの {'thr','min_area','min_peak'}（長さ bands.n_bands）。
    戻り値 uint8[N,H,W]（重なりなし）。postprocess.instances と同じ形式。

    手順: 画素のしきい値（帯の thr、全帯同じなら全体で共通）→ 8連結成分 → 重心の帯の min_area/min_peak で絞る。
    帯が1つなら postprocess.instances(prob, thr, min_area, min_peak) と一致する。
    """
    if len(params) != bands.n_bands:
        raise ValueError(f"params must have {bands.n_bands} entries")
    if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1 + 1e-4:
        raise ValueError("probability map has NaN/inf or is outside [0,1]")
    thrs = [p["thr"] for p in params]
    if len(set(thrs)) == 1:
        m = prob > thrs[0]
    else:  # 画素の位置でしきい値を変える方式（参考）
        thr_map = np.asarray(thrs, float)[band_map(prob.shape, bands, geom)]
        m = prob > thr_map
    lab, n = ndi.label(m, structure=EIGHT)
    if n == 0:
        return np.zeros((0, *prob.shape), np.uint8)
    idx = np.arange(1, n + 1)
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    peak = np.asarray(ndi.maximum(prob, lab, idx))
    cy, cx = (np.asarray(c) for c in zip(*ndi.center_of_mass(np.ones_like(lab), lab, idx))) if n else ([], [])
    band = bands.index(position_values(cy, cx, geom, bands.quantity))
    ma = np.array([p["min_area"] for p in params])[band]
    mp = np.array([p["min_peak"] for p in params])[band]
    keep = idx[(area >= ma) & (peak >= mp)]
    if len(keep) == 0:
        return np.zeros((0, *prob.shape), np.uint8)
    return np.stack([(lab == i).astype(np.uint8) for i in keep])


# ---- 円盤の推定 ----
def fit_circle(img: np.ndarray, trim: float = 3.0, half: int = 40, n_ang: int = 360) -> tuple[float, float, float, int]:
    """グレー画像から円盤の (cx, cy, R, 採用した縁の点数)。

    手順: 1/4 に縮小した画像で Hough 変換により半径 200〜250（原寸 800〜1000）の円を粗く探す →
    原寸で、その円の周り ±half 画素の半径方向の明るさの最も急な下降（円盤の縁）を各角度で拾う →
    円を最小二乗であて、残差が trim 画素を超える点（プロミネンス・文字など）を除いて再あてはめ。
    空の輝き（観測所で明るさが違う）に影響されない。R は縁の外側寄り（最急下降点）なので、公開 Notebook の値より数画素大きい。
    """
    import cv2
    from scipy import ndimage as ndi

    sm4 = cv2.resize(cv2.GaussianBlur(img, (0, 0), 2), (img.shape[1] // 4, img.shape[0] // 4), interpolation=cv2.INTER_AREA)
    h = cv2.HoughCircles(sm4, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=60, param2=40, minRadius=200, maxRadius=250)
    if h is None:
        raise ValueError("Hough: no disc found")
    cx, cy, r = (float(v) * 4 for v in h[0, 0])
    sm = cv2.GaussianBlur(img, (0, 0), 2).astype(np.float32)
    rs = np.arange(r - half, r + half, 0.5)
    pts = []
    for a in np.linspace(0, 2 * np.pi, n_ang, endpoint=False):
        x, y = cx + rs * np.cos(a), cy + rs * np.sin(a)
        if x.min() < 0 or y.min() < 0 or x.max() > img.shape[1] - 1 or y.max() > img.shape[0] - 1:
            continue
        g = np.gradient(ndi.map_coordinates(sm, [y, x], order=1))
        i = int(np.argmin(g))
        if g[i] < -3:
            pts.append((cx + rs[i] * np.cos(a), cy + rs[i] * np.sin(a)))
    pts = np.asarray(pts)
    if len(pts) < 50:
        raise ValueError("too few limb points")
    xs, ys = pts[:, 0], pts[:, 1]
    sel = np.ones(len(xs), bool)
    for _ in range(3):
        A = np.c_[2 * xs[sel], 2 * ys[sel], np.ones(sel.sum())]
        (fx, fy, c), *_ = np.linalg.lstsq(A, xs[sel] ** 2 + ys[sel] ** 2, rcond=None)
        fr = np.sqrt(c + fx**2 + fy**2)
        sel = np.abs(np.hypot(xs - fx, ys - fy) - fr) <= trim
    return float(fx), float(fy), float(fr), int(sel.sum())


# ---- 成分ごとの採点材料 ----
def components(prob: np.ndarray, thr: float, min_area: int, gt_sets: list[list[dict]], geom: Geometry = Geometry()) -> dict:
    """prob を thr で成分化し、面積 >= min_area の成分について、面積・ピーク・重心の位置・各注釈集合とのIoUを返す。

    絞り込み（min_area/min_peak を後から変える）に必要な量だけを持つ。IoU は pycocotools（採点と同じ）。
    """
    from pycocotools import mask as mask_utils

    from .metrics import encode

    lab, n = ndi.label(prob > thr, structure=EIGHT)
    H, W = prob.shape
    out = {"thr": thr, "area": np.zeros(0, int), "peak": np.zeros(0), "cy": np.zeros(0), "cx": np.zeros(0),
           "iou": [np.zeros((0, len(g))) for g in gt_sets]}
    if n == 0:
        return out
    area = np.bincount(lab.ravel(), minlength=n + 1)[1:]
    peak = np.asarray(ndi.maximum(prob, lab, np.arange(1, n + 1)))
    objs = ndi.find_objects(lab)
    ids = [i for i in range(1, n + 1) if area[i - 1] >= min_area]
    if not ids:
        return out
    cy, cx, rles = [], [], []
    buf = np.zeros((H, W), np.uint8, order="F")
    for i in ids:
        sl = objs[i - 1]
        m = lab[sl] == i
        yy, xx = np.nonzero(m)
        cy.append(yy.mean() + sl[0].start)
        cx.append(xx.mean() + sl[1].start)
        buf[sl] = m
        rles.append(mask_utils.encode(buf))
        buf[sl] = 0
    out.update(area=area[np.array(ids) - 1], peak=peak[np.array(ids) - 1], cy=np.array(cy), cx=np.array(cx))
    out["iou"] = [np.asarray(mask_utils.iou(rles, g, [0] * len(g))).reshape(len(rles), len(g)) if g else
                  np.zeros((len(rles), 0)) for g in gt_sets]
    return out


def score_keep(comp: dict, keep: np.ndarray) -> dict:
    """成分の採否 keep（bool[P]）で、注釈集合ごとの統計量を合算した iou_sum, tp, fp, fn と面積別内訳を返す。"""
    iou_sum = 0.0
    tp = fp = fn = 0
    for I in comp["iou"]:
        sub = I[keep]
        hit = sub > 0.5
        iou_sum += float(sub[hit].sum())
        tp += int(hit.sum())
        fp += int((~hit.any(1)).sum())
        fn += int((~hit.any(0)).sum()) if I.shape[1] else 0
    return {"iou_sum": iou_sum, "tp": tp, "fp": fp, "fn": fn}


def search_combos(comp_list: list[dict], band_idx_list: list[np.ndarray], n_bands: int, areas: list[int],
                  peaks: list[float]) -> tuple[list[tuple], np.ndarray]:
    """帯ごとの (min_area, min_peak) の全組合せについて、成分リスト（画像ごと、同一 thr）の合計 [iou_sum,tp,fp,fn] を返す。

    戻り値: (組合せ [((a_0,q_0),(a_1,q_1),...)], 合計 float[C,4])。全組合せを行列積でまとめて採点する。
    """
    import itertools

    per_band = list(itertools.product(areas, peaks))
    combos = list(itertools.product(per_band, repeat=n_bands))
    A = np.array([[c[b][0] for b in range(n_bands)] for c in combos])  # [C,B]
    Q = np.array([[c[b][1] for b in range(n_bands)] for c in combos])
    tot = np.zeros((len(combos), 4))
    for comp, band in zip(comp_list, band_idx_list):
        if len(comp["area"]) == 0:
            for I in comp["iou"]:
                tot[:, 3] += I.shape[1]
            continue
        keep = (comp["area"][None, :] >= A[:, band]) & (comp["peak"][None, :] >= Q[:, band])  # [C,P]
        kf = keep.astype(float)
        for I in comp["iou"]:
            hit = I > 0.5
            hf = hit.astype(float)
            tot[:, 0] += kf @ np.where(hit, I, 0.0).sum(1)
            tot[:, 1] += kf @ hf.sum(1)
            tot[:, 2] += kf @ (~hit.any(1)).astype(float)
            tot[:, 3] += I.shape[1] - ((kf @ hf) > 0).sum(1)
    return combos, tot


# ---- 保存済みの確率の場所（native_remote の n1 = DINOv3-L, n0 = 原寸EfficientNetV2-S） ----
REMOTE = "outputs/filament/native_remote"


def run_paths(model: str, k: int) -> dict:
    """model ('n1'|'n0') と外側fold k の、外側評価・内側tune・選択記録・評価記録の場所。"""
    from pathlib import Path

    if model == "n1":
        base = Path(REMOTE) / f"n1_f{k}" / "native_l"
    elif model == "n0":
        base = Path(REMOTE) / ("n0" if k in (0, 1) else "n0_f234") / "native_n0"
    else:
        raise ValueError(model)
    return {"eval_probs": base / "outer" / f"fold{k}" / "eval_probs.npz",
            "record": base / "outer" / f"fold{k}" / "evaluation_record.json",
            "per_image": base / "outer" / f"fold{k}" / "per_image.csv",
            "tune_probs": base / f"fold{k}" / "inner0" / "early6000" / "tune_probs.npz",
            "selection": base / f"fold{k}" / "inner0" / "selection.json"}


def _comp_task(args):
    path, stem, thrs, min_area, gt_sets = args
    prob = np.load(path)[stem]
    return stem, {t: components(prob, t, min_area, gt_sets) for t in thrs}


def components_for_images(path, stems: list[str], thrs: list[float], min_area: int, gts: dict, procs: int = 4) -> dict:
    """npz の画像 stems について、しきい値ごとの成分情報（components）を画像単位で並列に作る。stem -> {thr: comp}。"""
    from multiprocessing import Pool

    tasks = [(str(path), s, list(thrs), min_area, gts[s]) for s in stems]
    with Pool(procs) as pool:
        return dict(pool.imap_unordered(_comp_task, tasks, chunksize=2))
