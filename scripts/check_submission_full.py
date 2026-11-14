"""提出形式CSVの独立検査（提出はしない）: 全テスト画像の対応、ID形式、RLEのデコード、個体間の画素重複、空予測、列構造を、
既存の filament.submission.verify とは別のコードで確認する。使い方: uv run python scripts/check_submission_full.py CSV [--out result.json]"""
import argparse, csv, json, re, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, "src")
from pycocotools import mask as mu
from filament.data import list_test_images

ap = argparse.ArgumentParser(); ap.add_argument("csv"); ap.add_argument("--out"); a = ap.parse_args()
test_ids = sorted(p.stem for p in list_test_images())
raw = Path(a.csv).read_bytes()
res = {"file_bytes": len(raw), "has_quote_chars": (b'"' in raw) or (b"'" in raw), "crlf": b"\r\n" in raw}
with open(a.csv, newline="") as f:
    rd = csv.reader(f); header = next(rd); rows = list(rd)
res["header"] = header; res["n_rows"] = len(rows); res["bad_row_width"] = sum(len(r) != 2 for r in rows)
fid = [r[0] for r in rows]
res["duplicate_filament_id"] = len(fid) - len(set(fid))
pat = re.compile(r"^(\d{14}[A-Za-z]{2})_(\d+)$")
bad = [x for x in fid if not pat.match(x)]; res["bad_id_format"] = len(bad)
by = defaultdict(list)
for r in rows:
    m = pat.match(r[0])
    if m: by[m.group(1)].append((int(m.group(2)), r[1]))
res["unknown_image_ids"] = sorted(set(by) - set(test_ids)); res["images_with_rows"] = len(by)
res["images_without_rows"] = sorted(set(test_ids) - set(by)); res["n_test_images"] = len(test_ids)
res["row_numbering_not_1_to_n"] = sum(sorted(i for i, _ in v) != list(range(1, len(v) + 1)) for v in by.values())
res["rows_per_image"] = dict(sorted(Counter(len(v) for v in by.values()).items()))
dec_fail = empty_mask = overlap_imgs = non_binary = 0; overlap_px = 0; areas = []
for img, items in by.items():
    acc = np.zeros((2048, 2048), np.uint16)
    for _, c in items:
        try:
            m = mu.decode({"size": [2048, 2048], "counts": c.encode()})
        except Exception:
            dec_fail += 1; continue
        if m.max() > 1: non_binary += 1
        s = int(m.sum()); areas.append(s); empty_mask += s == 0; acc += m
    o = int((acc > 1).sum()); overlap_px += o; overlap_imgs += o > 0
res.update({"rle_decode_failures": dec_fail, "empty_masks": empty_mask, "non_binary_masks": non_binary, "images_with_pixel_overlap": overlap_imgs,
            "overlap_pixels_total": overlap_px, "mask_area_min_median_max": [min(areas), int(np.median(areas)), max(areas)]})
res["ok"] = all([res["bad_row_width"] == 0, res["duplicate_filament_id"] == 0, res["bad_id_format"] == 0, not res["unknown_image_ids"], dec_fail == 0,
                 empty_mask == 0, overlap_px == 0, not res["has_quote_chars"]])
res["note"] = "空予測の画像数は images_without_rows。0枚でも空予測が受理される証明にはならず、逆にあっても不受理の証拠ではない(公式に記載なし)"
print(json.dumps(res, indent=1, ensure_ascii=False))
if a.out: Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
