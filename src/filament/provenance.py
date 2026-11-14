"""来歴の記録用の小さな関数群: ハッシュ、git、実行環境。"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from pathlib import Path


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_obj(obj) -> str:
    """JSON化できる値の正規化ハッシュ（キー順に依存しない）。"""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def ids_hash(ids) -> str:
    """画像IDの集合のハッシュ（順序に依存しない）。"""
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()[:16]


def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.run(["git", *a], capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:  # noqa: BLE001 - 来歴が取れなくても実行は止めない
            return ""

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "src", "configs"))}


def env_state() -> dict:
    import numpy
    import torch

    out = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
           "numpy": numpy.__version__}
    try:
        import timm

        out["timm"] = timm.__version__
    except ImportError:
        pass
    try:
        import transformers

        out["transformers"] = transformers.__version__
    except ImportError:
        pass
    return out


def dinov3_provenance(variant: str) -> dict:
    """キャッシュ済みのtimm DINOv3の repo / revision / model.safetensors の sha256 / ライセンス名（offlineで読む）。"""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from huggingface_hub import hf_hub_download

    from .model_dino import TIMM_NAMES

    repo = "timm/" + TIMM_NAMES[variant]
    path = hf_hub_download(repo, "model.safetensors")
    return {"repo": repo, "revision": path.split("/snapshots/")[1].split("/")[0], "sha256": sha256_file(Path(path)),
            "license": "DINOv3 License (Meta, last updated 2025-08-19)"}
