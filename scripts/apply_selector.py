"""学習済みの選別器 C を確率地図に適用し、提出形式のCSVを書く(提出はしない)。事前登録: candidate_select_preregistration.json の final_selector_plan。
入力: 確率の保存先(画像ごとの .npy、または dev 用に npz とキー)、選別器 pkl、出力CSV。thr 0.4 の候補(面積100以上)に s=r_hat x q_hat を付け s>=tau を採用。
使い方: uv run python scripts/apply_selector.py --probs DIR --selector final_selector.pkl --out out.csv [--npz FILE --keys a,b]"""
import argparse, hashlib, json, pickle, sys
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
sys.path.insert(0, "scripts"); sys.path.insert(0, "src")
import candidate_select as cs
from filament import submission
from filament.postprocess import EIGHT, check_disjoint

ap = argparse.ArgumentParser(); ap.add_argument("--probs"); ap.add_argument("--npz"); ap.add_argument("--keys"); ap.add_argument("--selector"); ap.add_argument("--post", help="選別器の代わりに固定の後処理(post.json)を適用(A2のCSV用)"); ap.add_argument("--out", required=True)
ap.add_argument("--allow-sklearn-mismatch", action="store_true"); ap.add_argument("--selector-sha256", help="選別器 pkl の期待 SHA-256。省略時は <選別器の親の親>/CANDIDATE_MAP.json の C.postprocess.selector_sha256"); ap.add_argument("--allow-unverified-selector", action="store_true", help="SHA-256 を検査せずに読む(信頼できる出所のときだけ)"); ap.add_argument("--ids-file", help="全テスト画像のID(空予測も含めて対象を明示)。省略時は入力のID")
a = ap.parse_args()
from filament import native_infer
def verified_selector(path):
    """pickle は読み込み時に任意のコードを実行できる。読む前に SHA-256 を期待値と照合し、不一致・期待値なしなら読まずに止まる。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""): h.update(b)
    got = h.hexdigest(); want = a.selector_sha256
    if want is None:
        cm = Path(path).resolve().parent.parent / "CANDIDATE_MAP.json"
        if cm.exists(): want = json.load(open(cm))["C"]["postprocess"]["selector_sha256"]
    if a.allow_unverified_selector: print("警告: 選別器の SHA-256 を検査せずに読む", got); return
    if want is None: sys.exit(f"選別器の期待 SHA-256 がない({Path(path).resolve().parent.parent / 'CANDIDATE_MAP.json'} も --selector-sha256 もない)。読まずに止まる。信頼できる出所なら --allow-unverified-selector")
    if got != want: sys.exit(f"選別器 {path} の SHA-256 が期待値と違う(実際 {got[:16]}…、期待 {want[:16]}…)。pickle は読まずに止まる")
    print("選別器の SHA-256 を検査(一致):", got[:16] + "…")
if a.selector: verified_selector(a.selector)
sel = pickle.load(open(a.selector, "rb")) if a.selector else None; tau = sel["tau"] if sel else None
if sel is not None:   # 学習時の scikit-learn の版と一致しなければ止まる(版が違うと保存した選別器を読めない/結果が変わりうる)
    import sklearn
    info_path = Path(a.selector).with_suffix(".json")
    want = json.load(open(info_path))["sklearn"] if info_path.exists() else None
    if want is None or want != sklearn.__version__:
        if not a.allow_sklearn_mismatch:
            sys.exit(f"scikit-learn の版が学習時 {want} と実行時 {sklearn.__version__} で違う(または {info_path} がない)。--allow-sklearn-mismatch で無視できるが結果は保証されない")
        print("警告: scikit-learn の版の不一致を無視", want, sklearn.__version__)
post = native_infer.load_post(Path(a.post)) if a.post else None
assert (sel is None) != (post is None), "--selector か --post のどちらか一方"
if a.npz:
    z = np.load(a.npz); keys = a.keys.split(",") if a.keys else sorted(z.files); get = lambda k: z[k].astype(np.float32)
else:
    keys = sorted(p.stem for p in Path(a.probs).glob("*.npy") if not p.name.endswith(".tmp.npy")); get = lambda k: np.load(Path(a.probs) / f"{k}.npy").astype(np.float32)


def masks_for(prob):
    if post is not None:
        m = native_infer.masks_from_prob(prob, post); return m, len(m)
    nodes = cs.nodes_from_prob(prob, [])["nodes"]
    if not nodes: return np.zeros((0, *prob.shape), bool), 0
    X = np.array([nd["f"] for nd in nodes]); s = np.clip(sel["mr"].predict(X), 0, 1) * np.clip(sel["mq"].predict(X), 0, 1)
    keep = [nd["lab"] for j, nd in enumerate(nodes) if nd["level"] == cs.LV04 and s[j] >= tau]
    lab, _ = ndi.label(prob > cs.THR[cs.LV04], structure=EIGHT)
    m = np.stack([lab == i for i in keep]) if keep else np.zeros((0, *prob.shape), bool)
    check_disjoint(m); return m, len(keep)


def gen():
    for k in keys:
        m, n = masks_for(get(k)); print(k, n, flush=True); yield k, m
ids = [l.strip() for l in open(a.ids_file)] if a.ids_file else keys
n = submission.write_stream(gen(), Path(a.out), ids)
print("rows", n, "verify", submission.verify(Path(a.out), ids), "tau", tau)
