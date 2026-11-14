"""Reproduce the submitted ten-model prediction from weights or saved seed maps.

The two five-model averages are computed separately, then averaged in float32.
The output directory records its inputs before allowing interrupted work to resume.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bundle", required=True)
    p.add_argument("--weights-root", action="append", default=[])
    p.add_argument("--repo", default=".")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--device", default="auto")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--ids-file", required=True)
    p.add_argument("--post")
    p.add_argument("--n", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--dry-g42")
    p.add_argument("--dry-g43")
    p.add_argument("--abort-after", type=int, default=10)
    p.add_argument("--abort-mean-seconds", type=float, default=25.0)
    p.add_argument("--abort-single-seconds", type=float, default=90.0)
    p.add_argument("--max-wall-seconds", type=float, default=0)
    args = p.parse_args()
    if bool(args.dry_g42) != bool(args.dry_g43):
        p.error("--dry-g42 and --dry-g43 must be supplied together")
    if args.n < 0 or args.threads < 1:
        p.error("--n must be nonnegative and --threads must be positive")
    return args


def probability(path):
    value = np.load(path, allow_pickle=False).astype(np.float32)
    if value.ndim != 2 or not np.isfinite(value).all() or value.min() < 0 or value.max() > 1:
        raise ValueError(f"invalid probability map: {path}")
    return value


def mean_seed_maps(first, second):
    if first.shape != second.shape:
        raise ValueError(f"probability shapes differ: {first.shape} != {second.shape}")
    # Preserve the exact arithmetic and dtype of the submitted pipeline.
    return ((first.astype(np.float32) + second.astype(np.float32)) / 2).astype(np.float32)


def prepare_output(out, identity):
    """Do not reuse results from another input, software or execution mode."""
    manifest = out / "run_inputs.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != identity:
            raise ValueError("output belongs to different inputs; use a new --out-dir")
    elif any((out / name).exists() for name in ("probs_g42", "probs_g43", "probs_ens10")):
        raise ValueError("existing probability files have no run_inputs.json; use a new --out-dir")
    else:
        manifest.write_text(json.dumps(identity, indent=1, sort_keys=True) + "\n")


def save_probability(path, value):
    tmp = path.with_name(path.stem + ".tmp.npy")
    np.save(tmp, value)
    tmp.replace(path)
    path.with_suffix(".sha256").write_text(sha256(path) + "\n")


def reusable_probability(path):
    if not path.exists():
        return False
    digest = path.with_suffix(".sha256")
    if not digest.exists() or digest.read_text().strip() != sha256(path):
        raise ValueError(f"cached probability is incomplete or changed: {path}")
    probability(path)
    return True


def run_command(command, cwd, log=None):
    result = subprocess.run(command, capture_output=True, text=True, cwd=cwd)
    if log:
        log.write_text(result.stdout + result.stderr)
    if result.returncode:
        raise RuntimeError(f"{Path(command[1]).name} failed: {result.stderr[-1000:]}")
    return result.stdout


def main(args):
    started = time.time()
    bundle, repo, out = (Path(p).resolve() for p in (args.bundle, args.repo, args.out_dir))
    images = Path(args.images_dir).resolve()
    post = Path(args.post).resolve() if args.post else bundle / "A2/post.json"
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / "src"))
    ids = [line.strip() for line in Path(args.ids_file).read_text().splitlines() if line.strip()]
    ids = ids[: args.n] if args.n else ids
    if not ids or len(ids) != len(set(ids)) or any(Path(i).name != i for i in ids):
        raise ValueError("image IDs must be nonempty, unique file names")

    def status(state, **fields):
        (out / "status.json").write_text(
            json.dumps(
                {
                    "state": state,
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "elapsed_seconds": round(time.time() - started),
                    **fields,
                },
                ensure_ascii=False,
                indent=1,
            )
        )

    def wall_check():
        if args.max_wall_seconds and time.time() - started > args.max_wall_seconds:
            raise TimeoutError("maximum wall time exceeded")

    selector = bundle / "C_10m/final_selector_10m.pkl"
    selector_info = json.loads((bundle / "C_10m/final_selector_10m.json").read_text())
    if sha256(selector) != selector_info["sha256_pkl"]:
        raise ValueError("selector SHA-256 mismatch")
    wm = json.loads((bundle / "weights_manifest_10m.json").read_text())
    groups = {"g42": wm["seed42_members"], "g43": wm["seed43_members"]}
    paths = {}
    dry = bool(args.dry_g42)
    if dry:
        sources = {"g42": Path(args.dry_g42).resolve(), "g43": Path(args.dry_g43).resolve()}
        input_hashes = {
            g: {i: sha256(directory / f"{i}.npy") for i in ids} for g, directory in sources.items()
        }
    else:
        for group, members in groups.items():
            paths[group] = []
            for member in members:
                path = next(
                    (
                        root / member["path"]
                        for root in [bundle, *map(Path, args.weights_root)]
                        if (root / member["path"]).exists()
                    ),
                    None,
                )
                if path is None or sha256(path) != member["sha256"]:
                    raise ValueError(f"missing weight or SHA-256 mismatch: {member['path']}")
                paths[group].append(path.resolve())
        input_hashes = {
            "images": {i: sha256(images / f"{i}.jpeg") for i in ids},
            "weights": {g: [m["sha256"] for m in members] for g, members in groups.items()},
            "manifests": {
                g: [sha256((bundle / m["manifest"]).resolve()) for m in members]
                for g, members in groups.items()
            },
        }

    # Code and library versions affect a resumed prediction, even when inputs match.
    code = sorted((repo / "src/filament").glob("*.py")) + [
        repo / "scripts/reinfer_10m.py",
        repo / "scripts/apply_selector.py",
        repo / "scripts/candidate_select.py",
    ]
    import scipy
    import sklearn
    import torch

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "mps"
        if args.device == "auto" and torch.backends.mps.is_available()
        else args.device
        if args.device != "auto"
        else "cpu"
    )
    hardware = {
        "device": device.type,
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
    }
    if not dry:
        import timm

        hardware["timm"] = timm.__version__
    identity = {
        "ids": ids,
        "dry": dry,
        "hardware": hardware,
        "input_hashes": input_hashes,
        "selector": sha256(selector),
        "post": sha256(post),
        "device": args.device,
        "threads": args.threads,
        "libraries": {
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "code": {str(p.relative_to(repo)): sha256(p) for p in code},
    }
    prepare_output(out, identity)
    (out / "ids.txt").write_text("\n".join(ids) + "\n")

    if not dry:
        from filament import engine, native_infer
        from filament.data import read_gray

        torch.set_num_threads(args.threads)
        env = {
            "device": device.type,
            "gpu_name": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "cuda_capability": list(torch.cuda.get_device_capability(0))
            if device.type == "cuda"
            else None,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if device.type == "cuda" else None,
            "timm": timm.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit-learn": sklearn.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "threads": args.threads,
            "n_images": len(ids),
        }
        (out / "env.json").write_text(json.dumps(env, indent=1))
        times = {}
        sources = {g: out / f"probs_{g}" for g in groups}
        for group, directory in sources.items():
            directory.mkdir(exist_ok=True)
            todo = [i for i in ids if not reusable_probability(directory / f"{i}.npy")]
            if not todo:
                continue
            models, cfg, _ = native_infer.load_weights(paths[group], device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            durations = []
            for k, image_id in enumerate(todo, 1):
                t0 = time.time()
                prob = engine.predict(
                    models, read_gray(images / f"{image_id}.jpeg"), device, cfg
                ).astype(np.float32)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                if device.type == "mps":
                    torch.mps.synchronize()
                dt = time.time() - t0
                if (
                    prob.ndim != 2
                    or not np.isfinite(prob).all()
                    or prob.min() < 0
                    or prob.max() > 1
                ):
                    raise ValueError(f"invalid probability: {group} {image_id}")
                save_probability(directory / f"{image_id}.npy", prob)
                durations.append(dt)
                status(
                    "inferring",
                    group=group,
                    done=k,
                    total=len(todo),
                    mean_seconds=round(sum(durations) / k, 2),
                )
                print(group, k, len(todo), image_id, round(dt, 2), "s", flush=True)
                if dt > args.abort_single_seconds or (
                    k == args.abort_after and sum(durations) / k > args.abort_mean_seconds
                ):
                    raise TimeoutError(f"inference speed limit exceeded: {group} {image_id}")
                if k % 10 == 0:
                    wall_check()
            times[group] = {
                "images": len(durations),
                "mean_seconds": round(sum(durations) / len(durations), 2),
                "total_seconds": round(sum(durations), 1),
                "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20)
                if device.type == "cuda"
                else None,
            }
            del models
            if device.type == "cuda":
                torch.cuda.empty_cache()
        (out / "timing.json").write_text(json.dumps(times, indent=1))
    else:
        (out / "env.json").write_text(json.dumps({"dry_run": True}))
        print("Using saved seed probability maps", flush=True)

    ensemble = out / "probs_ens10"
    ensemble.mkdir(exist_ok=True)
    for image_id in ids:
        path = ensemble / f"{image_id}.npy"
        if not reusable_probability(path):
            save_probability(
                path,
                mean_seed_maps(
                    probability(sources["g42"] / f"{image_id}.npy"),
                    probability(sources["g43"] / f"{image_id}.npy"),
                ),
            )
    references = json.loads((bundle / "probs_reference_sha256.json").read_text())
    report = {}
    for group, directory in (*sources.items(), ("ens10", ensemble)):
        different = [i for i in ids if sha256(directory / f"{i}.npy") != references[group][i]]
        report[group] = {
            "images": len(ids),
            "bit_identical": len(ids) - len(different),
            "differ": different[:30],
        }
        if different:
            report[group]["note"] = (
                "Probability hashes differ; numerical differences are not measured."
            )
    (out / "probs_sha_comparison.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(
        "Probability SHA-256:",
        {g: f"{r['bit_identical']}/{r['images']}" for g, r in report.items()},
        flush=True,
    )
    wall_check()
    for name, extra in (
        ("C_10m", ["--selector", str(selector), "--selector-sha256", selector_info["sha256_pkl"]]),
        ("A2_10m", ["--post", str(post)]),
    ):
        result = run_command(
            [
                sys.executable,
                str(repo / "scripts/apply_selector.py"),
                "--probs",
                str(ensemble),
                "--ids-file",
                str(out / "ids.txt"),
                "--out",
                str(out / f"{name}.csv"),
                *extra,
            ],
            repo,
            out / f"{name}.log",
        )
        print(name, (result.strip().splitlines() or [""])[-1], flush=True)
    result = run_command(
        [
            sys.executable,
            str(repo / "scripts/compare_csv_detail.py"),
            str(out / "C_10m.csv"),
            str(bundle / "submission_C_10m.csv"),
            "--ids-file",
            str(out / "ids.txt"),
            "--label-new",
            "再生成 C(10モデル)",
            "--label-ref",
            "提出済み C(10モデル, 58204294)",
            "--json-out",
            str(out / "comparison_C_10m.json"),
            "--md-out",
            str(out / "COMPARISON_C_10m.md"),
        ],
        repo,
    )
    print(result[-2500:])
    status(
        "done",
        images=len(ids),
        probs={g: f"{r['bit_identical']}/{r['images']}" for g, r in report.items()},
    )


if __name__ == "__main__":
    args = parse_args()
    try:
        main(args)
    except Exception as exc:
        out = Path(args.out_dir).resolve()
        out.mkdir(parents=True, exist_ok=True)
        (out / "status.json").write_text(
            json.dumps(
                {
                    "state": "aborted" if isinstance(exc, TimeoutError) else "failed",
                    "reason": str(exc),
                },
                ensure_ascii=False,
                indent=1,
            )
        )
        sys.exit(str(exc))
