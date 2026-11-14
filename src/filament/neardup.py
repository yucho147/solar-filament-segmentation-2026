"""近似画像の候補抽出。自動判定ではなく、人が目視で確認するための候補一覧と比較画像を作る。

単純な縮小画像の知覚ハッシュは、太陽円盤の共通形状（円盤の輪郭・周縁減光）が支配して判別できなかった。
そこで次の手順で円盤の共通形状を取り除いた残差の相関を使う:
  1. 円盤を検出して中心と半径を求め、円盤が同じ位置・大きさになるよう切り出して 128x128 に揃える。
  2. 半径方向の中央値プロファイル（周縁減光）を引く。
  3. 観測所(site)ごとの平均残差（固定パターン）を引く。
  4. 円盤の内側(r<0.85)だけを使い、帯域通過（細かいノイズと大域の勾配を除く）した残差を 64x64 に縮小し、画像ごとに正規化する。
相関が高い = 円盤内部の濃淡パターン（暗条・活動領域・黒点・画像ノイズ）が似ている、という意味で、
「同一画像」や「同一構造」を断定する値ではない。比較対象として、時間が近い組（正の参照）と離れた組（負の参照）の分布を併記する。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

G = 128  # 円盤を揃えたあとの一辺
R_IN = 0.95  # 切り出し時に円盤外として落とす半径
R_USE = 0.85  # 比較に使う半径（周縁は中心ずれ・周縁減光の残りが支配するので除く）


def _disc(img: np.ndarray) -> tuple[float, float, float]:
    """円盤の中心(y,x)と半径(縮小画像の画素)。"""
    sm = ndi.gaussian_filter(img.astype(np.float32), 2)
    lab, n = ndi.label(sm > 12)
    if n == 0:
        raise ValueError("no disc found")
    size = ndi.sum(np.ones_like(lab), lab, np.arange(1, n + 1))
    m = ndi.binary_fill_holes(lab == 1 + int(np.argmax(size)))
    cy, cx = ndi.center_of_mass(m)
    return float(cy), float(cx), float(np.sqrt(m.sum() / np.pi))


def disc_normalized(path: Path, work: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """円盤を中心・半径で揃えた float32[G,G]（円盤外は NaN）と、各画素の正規化半径。"""
    with Image.open(path) as im:
        small = np.asarray(im.convert("L").resize((work, work), Image.Resampling.BOX))
    cy, cx, r = _disc(small)
    half = 1.05 * r
    ys = np.linspace(cy - half, cy + half, G)
    xs = np.linspace(cx - half, cx + half, G)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    out = ndi.map_coordinates(small.astype(np.float32), [yy, xx], order=1, mode="constant", cval=0)
    rho = np.hypot(*(np.mgrid[0:G, 0:G] - (G - 1) / 2)) / ((G - 1) / 2 / 1.05)
    out[rho > R_IN] = np.nan
    return out, rho


def remove_limb(a: np.ndarray, rho: np.ndarray, bins: int = 40) -> np.ndarray:
    """半径ごとの中央値(周縁減光)を引く。"""
    out = a.copy()
    edges = np.linspace(0, R_IN, bins + 1)
    idx = np.digitize(rho, edges) - 1
    for b in range(bins):
        m = (idx == b) & np.isfinite(a)
        if m.any():
            out[m] = a[m] - np.median(a[m])
    return out


def embeddings(paths: dict[str, Path], sites: dict[str, str], sigma: float = 1.5) -> tuple[list[str], np.ndarray]:
    """stem -> 残差ベクトル[N, 64*64]。円盤外は0、画像ごとに平均0・ノルム1。"""
    stems = sorted(paths)
    res = {}
    rho = None
    for s in stems:
        a, rho = disc_normalized(paths[s])
        res[s] = remove_limb(a, rho)
    site_mean = {}
    for site in set(sites.values()):
        g = [np.nan_to_num(res[s]) for s in stems if sites[s] == site]
        site_mean[site] = np.mean(g, 0)
    inside = rho <= R_USE
    vecs = []
    for s in stems:
        r = np.nan_to_num(res[s]) - site_mean[sites[s]]
        r[~inside] = 0
        r = ndi.gaussian_filter(r, sigma) - ndi.gaussian_filter(r, 5 * sigma)  # 帯域通過
        r[~inside] = 0
        r = np.asarray(Image.fromarray(r).resize((64, 64), Image.Resampling.BOX))
        v = r.ravel() - r.ravel().mean()
        vecs.append(v / (np.linalg.norm(v) + 1e-8))
    return stems, np.stack(vecs).astype(np.float32)


def reference_distributions(corr: np.ndarray, days: np.ndarray, same_site: np.ndarray) -> dict:
    """相関の目安: 正の参照(1日以内)・離れた組(200日超)の分位点。"""
    iu = np.triu_indices_from(corr, 1)
    c, d, ss = corr[iu], days[iu], same_site[iu]
    q = lambda x: {"n": int(len(x)), **{f"p{p}": float(np.quantile(x, p / 100)) for p in (50, 90, 99, 99.9)}} if len(x) else {"n": 0}  # noqa: E731
    return {"within_1day": q(c[d <= 1]), "gap_over_200days": q(c[d > 200]),
            "gap_over_200days_same_site": q(c[(d > 200) & ss]), "gap_over_200days_diff_site": q(c[(d > 200) & ~ss])}


def candidates(stems, corr, days, same_site, relation, top: int = 60, min_gap: float = 14.0) -> list[dict]:
    """境界をまたぐ組(relationが空でない)かつ撮影間隔 min_gap 日超の組を相関の高い順に top 件。"""
    n = len(stems)
    rows = []
    iu = np.triu_indices(n, 1)
    order = np.argsort(-corr[iu])
    for o in order:
        i, j = iu[0][o], iu[1][o]
        if days[i, j] <= min_gap:
            continue
        rel = relation(stems[i], stems[j])
        if not rel:
            continue
        rows.append({"a": stems[i], "b": stems[j], "corr": float(corr[i, j]), "gap_days": float(days[i, j]),
                     "same_site": bool(same_site[i, j]), "boundaries": ";".join(rel), "review": ""})
        if len(rows) >= top:
            break
    return rows


def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def contact_sheet(rows: list[dict], paths: dict[str, Path], times: dict, out: Path, n: int = 12) -> None:
    """候補の上位 n 組を、左右に 元画像（縮小）と残差を並べた画像にする。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = rows[:n]
    fig, ax = plt.subplots(len(rows), 4, figsize=(13, 3.3 * len(rows)))
    ax = np.atleast_2d(ax)
    for r, row in enumerate(rows):
        for c, key in enumerate(("a", "b")):
            s = row[key]
            a, rho = disc_normalized(paths[s])
            ax[r, 2 * c].imshow(np.nan_to_num(a), cmap="gray")
            ax[r, 2 * c].set_title(f"{s}\n{times[s]:%Y-%m-%d %H:%M}", fontsize=8)
            res = np.nan_to_num(remove_limb(a, rho))
            res[rho > R_USE] = 0
            bp = ndi.gaussian_filter(res, 1.5) - ndi.gaussian_filter(res, 7.5)  # 順位付けと同じ帯域通過（サイト平均は未除去）
            bp[rho > R_USE] = 0
            ax[r, 2 * c + 1].imshow(bp, cmap="RdBu_r", vmin=-8, vmax=8)
            ax[r, 2 * c + 1].set_title("residual" + (f"   corr={row['corr']:.3f} gap={row['gap_days']:.0f}d" if c else ""), fontsize=8)
        for q in ax[r]:
            q.set_xticks([])
            q.set_yticks([])
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=70)
    plt.close(fig)
