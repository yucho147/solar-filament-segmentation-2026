"""27日block + 前後14日purge の時間分割（blocked27d_purged14d_v1）。"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime

import numpy as np

BLOCK_DAYS = 27
PURGE_DAYS = 14


def _blocks(times: dict[str, datetime]) -> dict[str, int]:
    t0 = min(times.values())
    return {k: int((t - t0).total_seconds() // (BLOCK_DAYS * 86400)) for k, t in times.items()}


def _near(times, ids_a, ids_b, days):
    """ids_a のうち ids_b のどれかと days 日以内のもの。"""
    tb = np.array([times[i].timestamp() for i in ids_b]) if ids_b else np.array([])
    out = []
    for i in ids_a:
        if tb.size and np.abs(tb - times[i].timestamp()).min() <= days * 86400:
            out.append(i)
    return out


def make_split(times: dict[str, datetime], seed: int = 42, holdout_frac: float = 0.15, n_folds: int = 5) -> dict:
    """times: 物理画像 file_name -> 撮影時刻。"""
    rng = np.random.default_rng(seed)
    blk = _blocks(times)
    ids = sorted(times)
    by_block: dict[int, list[str]] = {}
    for i in ids:
        by_block.setdefault(blk[i], []).append(i)
    order = list(rng.permutation(sorted(by_block)))
    holdout: list[str] = []
    for b in order:  # holdoutは約15%になるまでblockを追加
        if len(holdout) >= holdout_frac * len(ids):
            break
        holdout += by_block[b]
    rest = [i for i in ids if i not in set(holdout)]
    buffer = _near(times, rest, holdout, PURGE_DAYS)
    dev = [i for i in rest if i not in set(buffer)]
    dev_blocks = sorted({blk[i] for i in dev})
    perm = list(rng.permutation(dev_blocks))
    folds = []
    for k in range(n_folds):
        vb = set(perm[k::n_folds])
        val = [i for i in dev if blk[i] in vb]
        trn_all = [i for i in dev if blk[i] not in vb]
        purge = _near(times, trn_all, val, PURGE_DAYS)
        train = [i for i in trn_all if i not in set(purge)]
        folds.append({"fit": train, "eval": val, "purge": purge})
    out = {
        "name": "blocked27d_purged14d_v1",
        "seed": seed,
        "holdout": holdout,
        "holdout_buffer": buffer,
        "development": dev,
        "folds": folds,
    }
    out["hash"] = split_hash(out)
    return out


def split_hash(s: dict) -> str:
    body = {k: v for k, v in s.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]


def validate_split(s: dict, times: dict[str, datetime] | None = None) -> None:
    """集合関係の検査。違反は例外。timesがあれば全被覆と14日距離も検査する。"""
    assert s["hash"] == split_hash(s), "split hash does not match contents"
    assert s["holdout"] and s["development"], "empty holdout/development"
    assert all(f["fit"] and f["eval"] for f in s["folds"]), "empty fold"
    if times is not None:
        allids = set(s["holdout"]) | set(s["holdout_buffer"]) | set(s["development"])
        assert allids == set(times), "split does not cover all images exactly"
        ho = [times[i] for i in s["holdout"]]
        for i in s["development"]:
            assert min(abs((times[i] - t).total_seconds()) for t in ho) > PURGE_DAYS * 86400, "development within purge of holdout"
        for f in s["folds"]:
            ev = [times[i] for i in f["eval"]]
            for i in f["fit"]:
                assert min(abs((times[i] - t).total_seconds()) for t in ev) > PURGE_DAYS * 86400, "fit within purge of eval"
    for f in s["folds"]:
        fit, ev, pu = set(f["fit"]), set(f["eval"]), set(f["purge"])
        assert not (fit & ev) and not (fit & pu) and not (ev & pu), "fold sets overlap"
        assert not ((fit | ev | pu) & (set(s["holdout"]) | set(s["holdout_buffer"]))), "dev leaks holdout"
    assert not set(s["holdout"]) & set(s["development"]), "holdout in development"
    assert not set(s["holdout"]) & set(s["holdout_buffer"]), "holdout in buffer"
