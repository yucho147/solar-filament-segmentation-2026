"""保存した学習履歴からfold別の学習・検証損失を描く。"""
from __future__ import annotations

import csv
from pathlib import Path


def plot_histories(paths: list[Path], output: Path) -> None:
    if not paths:
        raise ValueError("no history files")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    for path in paths:
        with Path(path).open(newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            raise ValueError(f"empty history: {path}")
        p = Path(path)
        label = "/".join(p.parts[-4:-1])
        axes[0].plot([int(r["step"]) for r in rows], [float(r["train_loss"]) for r in rows], label=label)
        valid = [r for r in rows if r["validation_loss"] != ""]
        if valid:
            axes[1].plot([int(r["step"]) for r in valid], [float(r["validation_loss"]) for r in valid],
                         marker="o", markersize=3, label=label)
    for ax, title in zip(axes, ("Training loss (smoothed crops)", "Validation loss (full images)")):
        ax.set(title=title, xlabel="Optimizer updates", ylabel="Loss")
        ax.grid(alpha=.25)
        if ax.lines:
            ax.legend(fontsize="small")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)


def _read(path: Path) -> list[dict]:
    with Path(path).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty history: {path}")
    return rows


def _series(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    pts = [(int(r["step"]), float(r[key])) for r in rows if r.get(key, "") not in ("", None)]
    return [p[0] for p in pts], [p[1] for p in pts]


def _shown_lr_columns(columns: list[str]) -> list[str]:
    """学習率の列(lr_<group>)のうち、図に出すもの。層別の多数のencoder groupは最浅・最深だけにする。"""
    lr_cols = [c for c in columns if c.startswith("lr_") and c != "lr"]
    blocks = sorted(c for c in lr_cols if c.startswith("lr_enc_block"))
    return [c for c in lr_cols if c not in blocks] + (blocks[:1] + blocks[-1:] if len(blocks) > 1 else blocks)


def plot_fold_curves(runs: list[dict], output: Path, require_train_fixed: bool = True) -> None:
    """foldごとの学習曲線。runs の各要素: {label, history: csv, summary: .aug.jsonのパス or None}。

    上段: 横軸=optimizer更新数。内側fit固定画像の損失(train_fixed_loss)と内側tuneの損失(validation_loss)は
    同じ関数・同じ設定で評価した値。拡張つき学習バッチの平滑化損失(train_loss)は別系列(灰色の点線)で、直接比較しない。
    下段: 各parameter groupが実際に使った学習率(対数軸)。縦線: 最良更新数(緑)、停止更新数(赤)、warmup終了(青)、schedule終端(橙)。
    """
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not runs:
        raise ValueError("no runs")
    fig, axes = plt.subplots(2, len(runs), figsize=(5.2 * len(runs), 7.5), constrained_layout=True, squeeze=False)
    for j, run in enumerate(runs):
        rows = _read(run["history"])
        tx, tv = _series(rows, "train_fixed_loss")
        vx, vv = _series(rows, "validation_loss")
        if require_train_fixed and vx and not tx:
            raise ValueError(f"{run['label']}: validation loss exists but the comparable train loss is missing")
        if require_train_fixed and tx and [*tx] != [*vx]:
            raise ValueError(f"{run['label']}: train and validation losses are not evaluated at the same updates")
        ax, ax2 = axes[0, j], axes[1, j]
        ex, ev = _series(rows, "train_loss")
        ax.plot(ex, ev, color="0.6", ls=":", label="train batches (augmented, EMA)")
        if tx:
            ax.plot(tx, tv, marker="o", ms=3, label="inner-fit fixed images (no aug)")
        if vx:
            ax.plot(vx, vv, marker="o", ms=3, label="inner-tune (no aug)")
        ax.set(title=run["label"], xlabel="Optimizer updates", ylabel="BCE+Dice loss")
        for col in _shown_lr_columns(list(rows[0])):
            lx, lv = _series(rows, col)
            ax2.plot(lx, lv, label=col[3:])
        ax2.set(xlabel="Optimizer updates", ylabel="Learning rate used", yscale="log")
        if run.get("summary") and Path(run["summary"]).exists():
            info = json.loads(Path(run["summary"]).read_text())
            es, lr = info.get("early_stopping", {}), info.get("lr", {})
            marks = [(es.get("best_step"), "tab:green", "best"), (es.get("stopped_step"), "tab:red", "stop"),
                     (lr.get("warmup_steps"), "tab:blue", "warmup end"),
                     (lr.get("scheduler_total_steps") if lr.get("holds_final_lr_after") else None, "tab:orange", "schedule end")]
            for step, color, name in marks:
                if step:
                    for a in (ax, ax2):
                        a.axvline(step, color=color, ls="--", lw=1, label=name if a is ax else None)
        for a in (ax, ax2):
            a.grid(alpha=.25)
            if a.lines:
                a.legend(fontsize="x-small")
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=130)
    plt.close(fig)
