"""任意: 提出モデルの 1 fold(外側再学習 + EMA 0.999)を、保存した設定・分割・選択記録から再度学習する経路(再学習は既定で実行しない)。
既定は「検査と計画の表示」だけ(dry-run)。--run を付けたときだけ、事前学習重みの取得 → 画像キャッシュ作成 → 学習(GPU、約 1.2〜2 時間/fold)を行う。
学習の出力は作業ディレクトリ(既定 <bundle>/outputs/train_repro)に新しく作り、提出モデルの weights/ は上書きしない(既存の出力があれば止まる)。
確認できること(検査する): 分割ハッシュ・入れ子分割ハッシュ・選択記録の SHA-256・学習画像 ID のハッシュ・解決後の設定ハッシュが、提出モデルの manifest と一致すること。
確認できないこと(この経路は「提出モデルの学習の完全な再現」ではない): 学習時の git commit(manifest の git.commit が空)、乱数・GPU・ライブラリ差によるビット一致、
fold1 が複数回の学習(source_run=ema_s42f1_run2)から選ばれた経緯、学習後の評価の再現。再学習した重みの SHA-256 は提出モデルと一致しないのが通常。
使い方: python scripts/train_fold.py --fold 0 [--data-root <MAGFiLO_1.0_Kaggle_2026>] [--workdir DIR] [--run]"""
import argparse, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path

BUNDLE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BUNDLE / "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--fold", type=int, required=True, choices=range(5))
ap.add_argument("--data-root", default=os.environ.get("EMA5_DATA_ROOT", ""), help="配布された MAGFiLO_1.0_Kaggle_2026 ディレクトリ(train/ を含む)")
ap.add_argument("--workdir", default=str(BUNDLE / "outputs" / "train_repro"))
ap.add_argument("--name", default="repro")
ap.add_argument("--run", action="store_true", help="実際に学習する(GPU を使う)。付けなければ検査と計画の表示のみ")
ap.add_argument("--json-out", default="")
a = ap.parse_args()

k = a.fold; work = Path(a.workdir).resolve(); name = f"{a.name}_f{k}"
T = BUNDLE / "training"; checks = []


def sha(p):
    h = hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()


def check(label, ok, detail=""):
    checks.append({"check": label, "ok": bool(ok), "detail": detail}); print(("OK  " if ok else "NG  ") + label, detail)


man = json.load(open(BUNDLE / f"weights/fold{k}/ema_0.999/manifest.json"))
split = json.load(open(T / "split.json")); nested = json.load(open(T / "nested_split.json"))
sel_path = T / f"selection_fold{k}.json"
check("分割ハッシュが提出モデルの manifest と一致", split["hash"] == man["split"], f"{split['hash']}")
check("入れ子分割ハッシュが manifest と一致", nested["hash"] == man["nested_hash"], f"{nested['hash']}")
check("選択記録の SHA-256 が manifest と一致", sha(sel_path) == man["selection_file_sha256"], sha(sel_path)[:16])
from filament.provenance import ids_hash, sha256_obj
check("学習画像 ID(manifest.fit)のハッシュが一致", ids_hash(man["fit"]) == man["fit_ids_hash"], f"{len(man['fit'])} 枚 {man['fit_ids_hash']}")
dev = set(split["development"]); fold_eval = set(split["folds"][k]["eval"])
check("学習画像は開発画像のみで、fold の評価画像・最終 holdout・buffer を含まない", set(man["fit"]) <= dev and not (set(man["fit"]) & (fold_eval | set(split["holdout"]) | set(split["holdout_buffer"]))))
import yaml
from filament import engine
d = yaml.safe_load((BUNDLE / "configs/native_dinov3_l_cos.yaml").read_text()); d["workers"] = 8; d["extra"] = {"ema": {"decay": 0.999, "extra_decays": [0.995]}}
cfg_hash = sha256_obj(engine.cfg_dict(engine.Config(**d)))[:16]
check("解決後の設定ハッシュが manifest と一致(configs/native_dinov3_l_cos.yaml + workers=8 + EMA 0.999/0.995)", cfg_hash == man["config_hash"], cfg_hash)
steps = man["refit_steps"]; check("再学習の更新数が選択記録と manifest で一致", steps == man["selection_refit_steps"], f"{steps}")
root = Path(a.data_root) if a.data_root else None
data_ok = bool(root and (root / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json").exists() and (root / "train/train_images").exists())
check("配布データ(--data-root の train/)がある", data_ok, str(root) if root else "未指定(EMA5_DATA_ROOT または --data-root)")
import torch
cuda = torch.cuda.is_available(); check("CUDA の GPU がある(学習には必要。提出モデルは RTX 3090 / bf16 で学習)", cuda, torch.cuda.get_device_name(0) if cuda else "なし")
print("\n不明のままの事項: 学習時の git commit(manifest.git.commit は空)。この複製コードのハッシュは CHECKSUMS.json。再学習した重みは提出モデルと SHA-256 が一致しない前提。")
out_dir = work / "outputs/filament/nested" / name
cmd = [sys.executable, "-m", "filament.cli_eval", "refit", "--config", str(BUNDLE / "configs/native_dinov3_l_cos.yaml"), "--name", name, "--fold", str(k),
       "-o", "workers=8", "-o", "extra={ema: {decay: 0.999, extra_decays: [0.995]}}", "--refit-steps", str(steps)]
plan = {"workdir": str(work), "output": str(out_dir / "outer" / f"fold{k}"), "refit_steps": steps, "fit_images": len(man["fit"]), "stored_train_seconds": man["resources"]["train_seconds"],
        "stored_peak_gpu_mb": man["resources"]["peak_memory_mb"], "command": " ".join(cmd), "cwd": str(work), "PYTHONPATH": str(BUNDLE / "src")}
print("\n計画:", json.dumps(plan, ensure_ascii=False, indent=1))
fixed_ok = all(c["ok"] for c in checks[:6])
if not a.run:
    print("\n--run なしのため学習しない(検査と計画のみ)。"); res = {"ran": False, "checks": checks, "plan": plan}
else:
    if not (fixed_ok and data_ok and cuda): sys.exit("検査に通らない項目(上の NG)があるため学習しない")
    if out_dir.exists(): sys.exit(f"{out_dir} が既にある。上書きしない")
    # 作業ディレクトリを整える: 分割・入れ子分割・選択記録は提出時のものを複製(ハッシュは上で検査済み)。配布データは symlink。
    (work / "outputs/filament/nested" / name / f"fold{k}/inner0").mkdir(parents=True)
    shutil.copy2(T / "split.json", work / "outputs/filament/split.json"); shutil.copy2(T / "nested_split.json", work / "outputs/filament/nested/nested_split.json")
    shutil.copy2(sel_path, out_dir / f"fold{k}/inner0/selection.json")
    (work / "configs/filament").mkdir(parents=True, exist_ok=True); shutil.copy2(T / "usage_history.yaml", work / "configs/filament/usage_history.yaml")   # refit が評価記録に添付する使用履歴
    dd = work / "data/raw/filament-segmentation-2026"; dd.mkdir(parents=True, exist_ok=True)
    if not (dd / "MAGFiLO_1.0_Kaggle_2026").exists(): os.symlink(root.resolve(), dd / "MAGFiLO_1.0_Kaggle_2026")
    env = {**os.environ, "PYTHONPATH": str(BUNDLE / "src")}
    pre = ("from huggingface_hub import hf_hub_download as h; import hashlib; p=h('timm/vit_large_patch16_dinov3.lvd1689m','model.safetensors',revision='" + man["pretrained_weights"]["revision"] + "');"
           "assert hashlib.sha256(open(p,'rb').read()).hexdigest()=='" + man["pretrained_weights"]["sha256"] + "', '事前学習重みの SHA-256 が違う'; print('pretrained OK', p)")
    subprocess.run([sys.executable, "-c", pre], check=True, cwd=work, env=env)
    subprocess.run([sys.executable, "-c", "from filament import cache; cache.build(1, with_sets=False); print('cache OK')"], check=True, cwd=work, env=env)
    subprocess.run(cmd, check=True, cwd=work, env=env)
    new = json.load(open(out_dir / f"outer/fold{k}/ema_0.999/manifest.json"))
    cmp = {x: {"stored": man[x], "new": new[x]} for x in ("split", "nested_hash", "config_hash", "fit_ids_hash", "refit_steps")}
    cmp["weights_sha256"] = {"stored": man["weights_sha256"], "new": new["weights_sha256"], "identical": man["weights_sha256"] == new["weights_sha256"]}
    cmp["git"] = {"stored": man["git"], "new": new["git"]}
    res = {"ran": True, "checks": checks, "plan": plan, "comparison_with_submitted_manifest": cmp}
    print(json.dumps(cmp, indent=1, ensure_ascii=False))
if a.json_out: Path(a.json_out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
