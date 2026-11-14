"""評価画像ごとの IoU 合計・TP・FP・FN の保存と、時間ブロック単位の対応あり再標本化（paired block bootstrap）。

PQ = IoU合計 / (TP + FP/2 + FN/2) は画像の和で書けるので、画像ごとの4つの合計だけ保存しておけば、
別のモデルの結果と画像を揃えて、時間ブロック（27日）を単位に再標本化できる。
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .metrics import PQStats, encode, pq_stats
from .postprocess import instances
from .split import BLOCK_DAYS  # noqa: F401  (ブロック幅の出典)

FIELDS = ["stem", "block", "year", "site", "n_sets", "n_gt", "n_pred", "iou_sum", "tp", "fp", "fn"]


def block_table(times: dict) -> dict[str, int]:
    """stem -> 27日ブロック番号。全学習画像の最初の撮影時刻を起点にする（split.py と同じ定義）。"""
    from .split import _blocks

    return _blocks(times)


def per_image_stats(probs, gts, post: dict, blocks: dict[str, int], times: dict) -> list[dict]:
    """probs: stem -> 原寸確率（.items()）。gts: stem -> 注釈集合ごとの個体RLE。注釈集合の統計量を画像内で合算する。"""
    rows = []
    for s, p in probs.items():
        pred = encode(instances(p, **post))
        st = PQStats()
        for g in gts[s]:
            st.add(pq_stats(g, pred))
        rows.append({"stem": s, "block": int(blocks[s]), "year": times[s].year, "site": s[-2:],
                     "n_sets": len(gts[s]), "n_gt": sum(len(g) for g in gts[s]), "n_pred": len(pred),
                     "iou_sum": st.iou_sum, "tp": st.tp, "fp": st.fp, "fn": st.fn})
    return rows


def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows({k: r[k] for k in FIELDS} for r in rows)


def read_rows(path: Path) -> list[dict]:
    with Path(path).open() as f:
        out = []
        for r in csv.DictReader(f):
            out.append({**r, **{k: int(r[k]) for k in ("block", "year", "n_sets", "n_gt", "n_pred", "tp", "fp", "fn")},
                        "iou_sum": float(r["iou_sum"])})
    return out


def pq_of(rows: list[dict]) -> float:
    iou, tp, fp, fn = (sum(r[k] for r in rows) for k in ("iou_sum", "tp", "fp", "fn"))
    den = tp + 0.5 * fp + 0.5 * fn
    return iou / den if den > 0 else 0.0


def _block_sums(rows: list[dict], order: list[int]) -> np.ndarray:
    idx = {b: i for i, b in enumerate(order)}
    m = np.zeros((len(order), 4))
    for r in rows:
        m[idx[r["block"]]] += (r["iou_sum"], r["tp"], r["fp"], r["fn"])
    return m


def _pq_from_sums(m: np.ndarray) -> np.ndarray:
    den = m[..., 1] + 0.5 * m[..., 2] + 0.5 * m[..., 3]
    return np.divide(m[..., 0], den, out=np.zeros_like(den), where=den > 0)


def paired_block_bootstrap(rows_a: list[dict], rows_b: list[dict], n_boot: int = 10000, seed: int = 0,
                           ci: float = 0.95) -> dict:
    """同じ評価画像に対する2モデルの PQ 差を、時間ブロックを単位に復元抽出で再標本化する。

    画像IDの集合とブロックの対応が両モデルで一致していなければ例外。信頼区間は「評価画像の標本ゆらぎ」だけを表し、
    学習seedの違い・条件選択の偏り・splitの作り方による不確かさは含まない。
    """
    ka = {r["stem"]: r["block"] for r in rows_a}
    kb = {r["stem"]: r["block"] for r in rows_b}
    if len(ka) != len(rows_a) or len(kb) != len(rows_b):
        raise ValueError("duplicate stems in per-image rows")
    if ka != kb:
        raise ValueError(f"image sets/blocks differ: only_a={len(set(ka) - set(kb))} only_b={len(set(kb) - set(ka))}")
    order = sorted(set(ka.values()))
    sa, sb = _block_sums(rows_a, order), _block_sums(rows_b, order)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(order), np.full(len(order), 1 / len(order)), size=n_boot)  # [n_boot, n_blocks]
    pa, pb = _pq_from_sums(counts @ sa), _pq_from_sums(counts @ sb)
    d = pb - pa
    lo, hi = np.quantile(d, [(1 - ci) / 2, 1 - (1 - ci) / 2])
    return {"n_images": len(ka), "n_blocks": len(order), "n_boot": n_boot, "seed": seed,
            "pq_a": float(_pq_from_sums(sa.sum(0))), "pq_b": float(_pq_from_sums(sb.sum(0))),
            "diff_b_minus_a": float(_pq_from_sums(sb.sum(0)) - _pq_from_sums(sa.sum(0))),
            "diff_ci": [float(lo), float(hi)], "ci_level": ci, "prob_b_better": float((d > 0).mean()),
            "scope": "評価画像の時間ブロック再標本化のみ。学習seed・条件選択・split作成の不確かさは含まない"}


def subgroup_pq(rows: list[dict], key: str) -> dict:
    """year/site 別の PQ と枚数。群間の差は構成の違い（年・観測所）を含む。"""
    out = {}
    for v in sorted({r[key] for r in rows}):
        sub = [r for r in rows if r[key] == v]
        out[str(v)] = {"n_images": len(sub), "pq": pq_of(sub)}
    return out
