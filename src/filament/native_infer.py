"""native cos_base 重みでテスト画像を推論し、提出形式CSVを作る経路（担当p、2026-11-06）。

既存の engine.predict / post_pos.instances_banded / postprocess.check_disjoint / submission を再利用する。
後処理設定は「内側tuneで選ばれた値を受け取るだけ」で、この経路は設定を探索・調整しない。
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

from . import engine, submission
from .post_pos import Bands, Geometry, instances_banded
from .postprocess import check_disjoint

ROOT = Path(__file__).resolve().parents[2]


# ---------- 後処理設定 ----------
def post_from_selection(selection: dict, fold: int, arm: str = "band") -> dict:
    """selection_cos_base*.json（{fold: {arm: {thr, combo:[[min_area,min_peak]..]}}}）から1つの設定を取り出す。
    arm='band' は円盤の縁ρ=0.85で2帯、'ref'/'ext' は1帯。取り出すだけで再選択しない。"""
    e = selection[str(fold)][arm]
    edges = [0.85] if arm == "band" else []
    g = Geometry()
    return make_post(e["thr"], [{"min_area": int(a), "min_peak": float(p)} for a, p in e["combo"]], edges=edges,
                     geometry=(g.cx, g.cy, g.radius), source={"fold": fold, "arm": arm, "pq_inner_tune": e.get("pq")})


def make_post(thr: float, params: list[dict], edges: list[float] = (), quantity: str = "rho",
              geometry: tuple = (Geometry().cx, Geometry().cy, Geometry().radius), source: dict | None = None) -> dict:
    post = {"thr": float(thr), "bands": {"quantity": quantity, "edges": [float(x) for x in edges]},
            "params": [{"min_area": int(p["min_area"]), "min_peak": float(p["min_peak"])} for p in params],
            "geometry": [float(x) for x in geometry], "source": source or {}}
    validate_post(post)
    return post


def validate_post(post: dict) -> None:
    b = Bands(post["bands"]["quantity"], tuple(post["bands"]["edges"]))
    if len(post["params"]) != b.n_bands:
        raise ValueError(f"params must have {b.n_bands} entries, got {len(post['params'])}")
    if not 0 < post["thr"] < 1 or len(post["geometry"]) != 3:
        raise ValueError("bad thr/geometry")


def save_post(post: dict, path: Path) -> None:
    validate_post(post)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(post, indent=1, ensure_ascii=False))


def load_post(path: Path) -> dict:
    post = json.loads(Path(path).read_text())
    validate_post(post)
    return post


def masks_from_prob(prob: np.ndarray, post: dict) -> np.ndarray:
    """確率 float[2048,2048] -> bool[N,2048,2048]（画素非重複）。"""
    b = Bands(post["bands"]["quantity"], tuple(post["bands"]["edges"]))
    params = [{"thr": post["thr"], **p} for p in post["params"]]
    m = instances_banded(prob, b, params, Geometry(*post["geometry"])).astype(bool)
    check_disjoint(m)
    return m


# ---------- 確率 ----------
def average_probs(probs: list[np.ndarray]) -> np.ndarray:
    if not probs:
        raise ValueError("no probabilities")
    if any(p.shape != probs[0].shape for p in probs):
        raise ValueError("shape mismatch")
    out = np.mean([p.astype(np.float32) for p in probs], axis=0, dtype=np.float32)
    if not np.isfinite(out).all() or out.min() < 0 or out.max() > 1 + 1e-4:
        raise ValueError("bad probabilities")
    return out


def manifest_path(weights: Path) -> Path:
    """重みに対応する manifest.json。EMA 重み `weights_ema_<decay>.pt` は同じ階層の `ema_<decay>/manifest.json`
    （重みの SHA-256 と decay、通常重みへの対応を記録）を読む。それ以外は重みの隣の manifest.json。"""
    m = re.fullmatch(r"weights_ema_([0-9.]+)\.pt", weights.name)
    if m:
        cand = weights.parent / f"ema_{m.group(1)}" / "manifest.json"
        if not cand.exists():
            raise FileNotFoundError(f"{weights.name}: EMA manifest not found at {cand}")
        return cand
    return weights.parent / "manifest.json"


def load_weights(weight_paths: list[Path], device, check_sha: bool = True):
    """各 weights.pt の隣の manifest.json から config を読み、モデルを構築・読込する（事前学習重みは取得しない）。
    戻り値: (models, cfg, manifests)。全 weights の config は一致が必要。"""
    mans = []
    for w in weight_paths:
        mp = manifest_path(Path(w))
        m = json.loads(mp.read_text())
        if check_sha and m.get("weights_sha256"):
            h = hashlib.sha256()
            with open(w, "rb") as f:
                for blk in iter(lambda: f.read(1 << 24), b""):
                    h.update(blk)
            if h.hexdigest() != m["weights_sha256"]:
                raise ValueError(f"{w}: sha256 differs from manifest")
        mans.append(m)
    if any(m["config"] != mans[0]["config"] for m in mans):
        raise ValueError("weights have different configs")
    cfg = engine.Config(**mans[0]["config"])
    return engine.load_models([Path(w) for w in weight_paths], cfg, device), cfg, mans


def infer_cfg(cfg: engine.Config, overlap: float | None = None, tta: bool | None = None) -> engine.Config:
    """推論設定の上書き（保存済み記録と一致させるなら None）。tile は変えない。"""
    inf = dict(cfg.infer)
    if overlap is not None:
        inf["overlap"] = overlap
    return dataclasses.replace(cfg, infer=inf, tta=cfg.tta if tta is None else tta)


# ---------- リーク検査（explore_d_leakcheck の再利用） ----------
def _lc():
    s = str(ROOT / "scripts")
    if s not in sys.path:
        sys.path.insert(0, s)
    import explore_d_leakcheck as lc

    return lc


def check_untrained(manifests: list[dict], eval_ids: list[str]) -> dict:
    """推論に使う全重みの学習画像(fit)が評価対象画像と重ならないこと。違反は Violation(AssertionError)。"""
    lc = _lc()
    for m in manifests:
        if m.get("init_checkpoint") is not None:
            raise lc.Violation("initialised from a previous checkpoint")
    return lc.check_ensemble(manifests, set(eval_ids), "native_infer")


# ---------- 提出CSV ----------
def run(weight_paths: list[Path], post: dict, images: list[Path], out_csv: Path, device, bs: int = 1,
        overlap: float | None = None, tta: bool | None = None, all_test_ids: list[str] | None = None,
        require_all: bool = True, check_sha: bool = True, keep_probs: dict | None = None, leak_check: bool = True) -> dict:
    """images のテスト画像を推論して CSV を書く。all_test_ids を省略すると images のIDだけを対象にした部分CSV。
    戻り値は時間・メモリ・件数の記録。"""
    import torch
    from .data import read_gray

    ids = [p.stem for p in images]
    t0 = time.perf_counter()
    models, cfg0, mans = load_weights(weight_paths, device, check_sha)
    cfg = infer_cfg(cfg0, overlap, tta)
    # leak_check=False は、開発用の分割ファイルと scripts/explore_d_leakcheck.py を持たない単独の提出環境用（学習画像とテスト画像の非重複は開発側で確認済みの記録に頼る）
    leak = check_untrained(mans, ids) if leak_check else {"label": "native_infer", "skipped": True}
    t_load = time.perf_counter() - t0
    per, items = [], []
    for p in images:
        img = read_gray(p)
        if img.shape != (2048, 2048) or img.dtype != np.uint8:
            raise ValueError(f"{p.name}: unexpected image {img.shape} {img.dtype}")
        t1 = time.perf_counter()
        prob = engine.predict(models, img, device, cfg)
        if device.type == "mps":
            torch.mps.synchronize()
        dt = time.perf_counter() - t1
        m = masks_from_prob(prob, post)
        if keep_probs is not None:
            keep_probs[p.stem] = prob
        per.append({"id": p.stem, "seconds": round(dt, 2), "n_instances": int(len(m)), "prob_max": float(prob.max())})
        items.append((p.stem, m))
    n = submission.write_stream(iter(items), out_csv, all_test_ids or ids)
    return {"rows": n, "load_seconds": round(t_load, 2), "per_image": per, "leak_check": leak,
            "infer": cfg.infer, "tta": cfg.tta, "tta_set": cfg.tta_set, "n_models": len(weight_paths),
            "empty_images": [d["id"] for d in per if d["n_instances"] == 0]}
