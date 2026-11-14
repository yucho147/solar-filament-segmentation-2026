"""2 つの提出形式 CSV の詳細比較(CPUのみ)。バイト一致、行数・画像数、画像ごとの個体数の一致、IoU による個体の対応(最大 IoU の分布、>=0.999・>=0.99・>0.5 の割合、片方にしか対応がない個体)、予測 0 件の画像の集合を JSON/Markdown にする。
使い方: python scripts/compare_csv_detail.py NEW.csv REF.csv --ids-file test_ids.txt [--label-new 再生成 --label-ref 提出済み] [--json-out x.json] [--md-out x.md]"""
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools import mask as mu

ap = argparse.ArgumentParser(); ap.add_argument("new"); ap.add_argument("ref"); ap.add_argument("--ids-file", required=True); ap.add_argument("--label-new", default="新"); ap.add_argument("--label-ref", default="参照")
ap.add_argument("--json-out"); ap.add_argument("--md-out"); a = ap.parse_args()
ids = [l.strip() for l in open(a.ids_file) if l.strip()]
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    d = defaultdict(list)
    for r in csv.DictReader(open(p, newline="")): d[r["filament_id"].rsplit("_", 1)[0]].append({"size": [2048, 2048], "counts": r["segmentation_rle"].encode()})
    return d
A, B = load(a.new), load(a.ref); best_a, best_b, same_n, same_rle, imgs_diff = [], [], 0, 0, []
for i in ids:
    x, y = A.get(i, []), B.get(i, [])
    if len(x) == len(y): same_n += 1
    if [r["counts"] for r in x] == [r["counts"] for r in y]: same_rle += 1
    else: imgs_diff.append((i, len(x), len(y)))
    if x and y:
        iou = np.asarray(mu.iou(x, y, [0] * len(y))).reshape(len(x), len(y)); best_a += iou.max(1).tolist(); best_b += iou.max(0).tolist()
    else: best_a += [0.0] * len(x); best_b += [0.0] * len(y)
ba, bb = np.array(best_a), np.array(best_b)
st = lambda v: {"n": int(len(v)), "ge_0.999": int((v >= 0.999).sum()), "ge_0.99": int((v >= 0.99).sum()), "gt_0.5": int((v > 0.5).sum()), "le_0.5(対応なし)": int((v <= 0.5).sum()), "mean_best_iou": round(float(v.mean()), 4) if len(v) else None, "median_best_iou": round(float(np.median(v)), 4) if len(v) else None}
res = {"new": a.new, "ref": a.ref, "sha256_new": sha(a.new), "sha256_ref": sha(a.ref), "byte_identical": (sha(a.new) == sha(a.ref)) if len(ids) >= 180 else None, "byte_identical_note": None if len(ids) >= 180 else "対象画像が 180 枚未満(部分集合)のためファイル全体のバイト一致は比較しない", "rows_new": sum(len(A.get(i, [])) for i in ids), "rows_ref": sum(len(B.get(i, [])) for i in ids),
       "images": len(ids), "images_with_rows_new": sum(1 for i in ids if A.get(i)), "images_with_rows_ref": sum(1 for i in ids if B.get(i)), "images_same_instance_count": same_n, "images_all_rle_identical": same_rle,
       "empty_images_new": [i for i in ids if not A.get(i)], "empty_images_ref": [i for i in ids if not B.get(i)], "empty_sets_equal": [i for i in ids if not A.get(i)] == [i for i in ids if not B.get(i)],
       "new_instances_vs_ref": st(ba), "ref_instances_vs_new": st(bb), "images_with_difference_head": imgs_diff[:20]}
if a.json_out: Path(a.json_out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
md = [f"# CSV 比較: {a.label_new} と {a.label_ref}", "", f"- バイト一致: **{res['byte_identical'] if res['byte_identical'] is not None else '（部分集合のため比較しない）'}**（SHA-256 {res['sha256_new'][:16]}… / {res['sha256_ref'][:16]}…）", f"- 行数: {res['rows_new']} / {res['rows_ref']}、予測のある画像: {res['images_with_rows_new']} / {res['images_with_rows_ref']}（全 {res['images']} 枚）",
      f"- 個体数が同じ画像: {same_n}/{len(ids)}、RLE が全て同一の画像: {same_rle}/{len(ids)}", f"- 予測 0 件の画像: {a.label_new} {res['empty_images_new']} / {a.label_ref} {res['empty_images_ref']}（集合が{'同じ' if res['empty_sets_equal'] else '違う'}）",
      f"- {a.label_new}の個体の最大 IoU: {res['new_instances_vs_ref']}", f"- {a.label_ref}の個体の最大 IoU: {res['ref_instances_vs_new']}"]
if a.md_out: Path(a.md_out).write_text("\n".join(md) + "\n")
print("\n".join(md))
