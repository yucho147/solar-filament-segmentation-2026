"""主催者の公開 Self Evaluation Notebook の PQ と、ローカルの PQ（metrics.py）を同じ正解・予測で比較する。

公開 Notebook (azimahmadzadeh/self-evaluation-notebook) の関数定義だけを AST で取り出して実行する（トップレベルの
コードは実行しない）。コードを複製せず、取得したファイルの SHA-256 を結果に残す。これは「公開 Notebook との一致」の確認で、
Kaggle の非公開採点処理との一致は確認できない。
"""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pycocotools import mask as mask_utils

from .metrics import PQStats, pq_stats

NOTEBOOK = Path("outputs/research/filament-segmentation-2026/20261031-winning-research/notebooks/azimahmadzadeh/"
                "self-evaluation-notebook/self-evaluation-notebook.ipynb")
NOTEBOOK_SHA256 = "ae7c9445b4eee2b77b216df5c9772fdf92fda15611e98abd7ef5f02a4e380801"
USED = ("get_overlap_matrices", "fp_count_hit", "fn_count_hit", "rles_to_layers", "process_entry", "get_overlap_df",
        "get_pq_score")


def load_reference(path: Path = NOTEBOOK) -> dict:
    """ノートブックから PQ に使う関数だけを取り出す。ハッシュが想定と違えば例外。"""
    from .provenance import sha256_file

    if sha256_file(path) != NOTEBOOK_SHA256:
        raise ValueError("Self Evaluation Notebook differs from the version this check was written for")
    nb = json.loads(Path(path).read_text())
    ns: dict = {"pd": pd, "np": np, "mask_util": mask_utils}
    import torch

    ns["torch"] = torch
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in USED]
        if keep:
            exec(compile(ast.Module(keep, []), str(path), "exec"), ns)  # noqa: S102 - 関数定義のみ
    missing = [n for n in USED if n not in ns]
    if missing:
        raise ValueError(f"functions not found in notebook: {missing}")
    return ns


# ---- DataFrame（提出と同じ filament_id, segmentation_rle）------------------------------------
def rle_str(mask: np.ndarray) -> str:
    return mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))["counts"].decode("ascii")


def make_gt_df(gt: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    """gt: 注釈集合ID('<annotator>-<image>') -> 個体マスク bool[2048,2048] のリスト。"""
    rows = [(f"{sid}_{k}", rle_str(m)) for sid, ms in gt.items() for k, m in enumerate(ms, 1)]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def make_pred_df(pred: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    """pred: 画像ID -> 個体マスクのリスト。"""
    rows = [(f"{iid}_{k}", rle_str(m)) for iid, ms in pred.items() for k, m in enumerate(ms, 1)]
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


def local_stats(gt_df: pd.DataFrame, pred_df: pd.DataFrame) -> dict[str, PQStats]:
    """ローカル実装で、注釈集合ごとの統計量を計算する（DataFrame から RLE へ戻して metrics.pq_stats を使う）。"""
    to_rle = lambda s: {"size": [2048, 2048], "counts": s.encode()}  # noqa: E731
    gid = gt_df["filament_id"].str.split("_", n=1).str[0]
    pid = pred_df["filament_id"].str.split("_", n=1).str[0]
    out = {}
    for sid in gid.unique():
        image = sid.split("-", 1)[1]
        g = [to_rle(s) for s in gt_df.loc[gid == sid, "segmentation_rle"]]
        p = [to_rle(s) for s in pred_df.loc[pid == image, "segmentation_rle"]]
        out[sid] = pq_stats(g, p)
    return out


def compare(gt_df: pd.DataFrame, pred_df: pd.DataFrame, ref: dict) -> dict:
    """公開 Notebook の PQ とローカル PQ、注釈集合ごとの TP/FP/FN/IoU 合計の差を返す。"""
    ov = ref["get_overlap_df"](gt_df, pred_df)
    ref_pq = float(ref["get_pq_score"](ov))
    loc = local_stats(gt_df, pred_df)
    tot = PQStats()
    for st in loc.values():
        tot.add(st)
    diffs = []
    rsum = np.zeros(4)
    for row in ov.itertuples(index=False):
        sid = row.annotator_image
        st = loc[sid]
        if row.n_gt == 0:
            r = (0.0, 0, row.n_pred, 0)
        elif row.n_pred == 0:
            r = (0.0, 0, 0, row.n_gt)
        else:
            hit = row.iou_matrix > 0.5
            r = (float(row.iou_matrix[hit].sum()), int(hit.sum()), ref["fp_count_hit"](hit), ref["fn_count_hit"](hit))
        rsum += r
        if (r[1], r[2], r[3]) != (st.tp, st.fp, st.fn) or abs(r[0] - st.iou_sum) > 1e-5:
            diffs.append({"set": sid, "ref": r, "local": (st.iou_sum, st.tp, st.fp, st.fn)})
    return {"ref_pq": ref_pq, "local_pq": tot.pq, "abs_diff": abs(ref_pq - tot.pq), "n_sets": len(loc),
            "ref": {"iou_sum": float(rsum[0]), "tp": int(rsum[1]), "fp": int(rsum[2]), "fn": int(rsum[3])}, "local": {"tp": tot.tp, "fp": tot.fp, "fn": tot.fn, "iou_sum": tot.iou_sum},
            "set_level_mismatches": diffs}
