"""提出 CSV の独立検査。プロジェクトのコード(filament.*)を使わず、標準ライブラリ・numpy・pycocotools だけで検査する。
検査: 見出し(filament_id,segmentation_rle)、行末(CRLF)、ID 形式 <画像ID>_<連番>、連番が 1..n の連続、画像 ID がテスト画像の集合に含まれる、RLE が復号できる、2048x2048、空でない、個体間の画素の重なりが 0、再符号化で同じ RLE、
既存の提出済み CSV と書式(見出し・行末・RLE の符号化方式)が一致。使い方: python independent_csv_check.py <csv> --test-dir <テスト画像dir> [--ref 提出済みCSV] [--out json]"""
import argparse, csv, json, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from pycocotools import mask as mu

ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--test-dir", required=True); ap.add_argument("--ref"); ap.add_argument("--out"); a = ap.parse_args()
raw = Path(a.csv).read_bytes(); lines = raw.split(b"\r\n"); res = {"file": a.csv, "bytes": len(raw), "ends_with_crlf": raw.endswith(b"\r\n"), "has_lf_only_lines": b"\n" in raw.replace(b"\r\n", b"")}
rows = list(csv.reader(raw.decode().splitlines())); res["header"] = rows[0]; body = rows[1:]; res["n_rows"] = len(body); res["bad_width"] = sum(len(r) != 2 for r in body)
test_ids = sorted(p.stem for p in Path(a.test_dir).glob("*.*") if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".fits"))
res["n_test_images_found"] = len(test_ids); byimg = defaultdict(list); bad_id = dup = 0; seen = set()
for fid, rle in body:
    m = re.fullmatch(r"(.+)_(\d+)", fid)
    if not m: bad_id += 1; continue
    if fid in seen: dup += 1
    seen.add(fid); byimg[m.group(1)].append((int(m.group(2)), rle))
res.update({"bad_id_format": bad_id, "duplicate_ids": dup, "images_with_rows": len(byimg), "images_not_in_test": sorted(set(byimg) - set(test_ids)), "test_images_without_rows": sorted(set(test_ids) - set(byimg))})
res["numbering_not_1_to_n"] = sum(sorted(n for n, _ in v) != list(range(1, len(v) + 1)) for v in byimg.values())
fail = empty = nonbin = overlap_imgs = reenc_diff = 0; areas = []
for img, v in byimg.items():
    acc = None
    for n, rle in sorted(v):
        r = {"size": [2048, 2048], "counts": rle.encode()}
        try: m = mu.decode(r)
        except Exception: fail += 1; continue
        if m.shape != (2048, 2048): fail += 1; continue
        if m.sum() == 0: empty += 1
        if m.max() > 1: nonbin += 1
        areas.append(int(m.sum()))
        if mu.encode(np.asfortranarray(m))["counts"] != r["counts"]: reenc_diff += 1
        acc = m.astype(np.int16) if acc is None else acc + m
    if acc is not None and acc.max() > 1: overlap_imgs += 1
res.update({"rle_decode_failures": fail, "empty_masks": empty, "non_binary": nonbin, "images_with_pixel_overlap": overlap_imgs, "reencode_mismatch": reenc_diff, "area_min": min(areas), "area_median": int(np.median(areas)), "area_max": max(areas)})
if a.ref:
    rr = Path(a.ref).read_bytes(); rrows = list(csv.reader(rr.decode().splitlines()))
    res["format_same_as_ref"] = {"header": rrows[0] == rows[0], "crlf": rr.endswith(b"\r\n") == raw.endswith(b"\r\n"), "ref_rows": len(rrows) - 1, "ref_images_with_rows": len({r[0].rsplit("_", 1)[0] for r in rrows[1:]})}
res["ok"] = all([res["header"] == ["filament_id", "segmentation_rle"], res["ends_with_crlf"], not res["has_lf_only_lines"], res["bad_width"] == 0, bad_id == 0, dup == 0, not res["images_not_in_test"], res["numbering_not_1_to_n"] == 0, fail == 0, empty == 0, nonbin == 0, overlap_imgs == 0, reenc_diff == 0, res["n_test_images_found"] == 180])
print(json.dumps(res, indent=1, ensure_ascii=False))
if a.out: Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
sys.exit(0 if res["ok"] else 1)
