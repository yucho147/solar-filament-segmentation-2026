# Solar Filament Segmentation Challenge 2026 — ten-model DINOv3 ensemble with a learned instance selector

Instance segmentation of solar filaments in GONG H-alpha images (MAGFiLO), scored by Panoptic Quality over annotation sets.
**Submitted solution ("C")**: ten fine-tuned DINOv3 ViT-L/16 segmentation models (2 seeds × 5 folds, EMA 0.999) → mean probability map → learned selector that keeps the connected components expected to match annotators → COCO-RLE instances. Public leaderboard score: 0.38 (two decimals).
The 4-page technical report is in [`report/main.pdf`](report/main.pdf).

## Pipeline (all steps are in the notebook `notebooks/pipeline_10m.ipynb`)
| Step | What happens | Code |
| --- | --- | --- |
| 1. Input | 8-bit grayscale H-alpha image, native 2048×2048 (no resizing; scaled to [0,1]) | `src/filament/data.py` (`read_gray`) |
| 2. Model | DINOv3 ViT-L/16 encoder (4 intermediate layers) + high-resolution convolutional branch + progressive decoder → 1 logit/pixel (trained on the mean of annotation sets; BCE + soft Dice, EMA 0.999) | `src/filament/model_dino.py`, `native.py`, `engine.py`, `configs/` |
| 3. Inference | overlapping 512 px tiles (50 % overlap), 3 views (identity, h-flip, v-flip), uniform tile weights; 5 fold models per seed averaged, then the two seeds averaged (ten-model mean, float32) | `src/filament/engine.py` (`predict`), `native_infer.py`, `scripts/reinfer_10m.py` |
| 4. Selector C | 8-connected components of p > 0.4 (area ≥ 100 px); two gradient-boosted regressors on 11 label-free descriptors estimate expected matched IoU; keep components with score ≥ τ = 0.25 | `scripts/candidate_select.py`, `scripts/apply_selector.py`, `C_10m/` |
| 5. Output | disjoint instances → compressed COCO RLE, one row per instance (`submission_C_10m.csv` format) | `src/filament/submission.py` |
| (Fixed-rule alternative A2) | components with area ≥ 400 px and peak ≥ 0.995; not the submitted solution | `A2/post.json` |

An executed example with outputs (2 test images on an Apple-silicon Mac, MPS) is in `notebooks/executed_example_mps_2images.ipynb`.

## Quick start
```bash
pip install -r requirements.txt                 # tested with Python 3.11 (pinned versions; scikit-learn 1.8.0 is required to load the selector)
python scripts/fetch_weights.py                 # downloads the ten weight files (12.2 GB) and verifies SHA-256  — see "Weights"
# put the competition test images (JPEG) under data/test_images/ (or set EMA10_IMAGES_DIR), then open notebooks/pipeline_10m.ipynb
```
Command-line equivalent of the notebook:
```bash
python scripts/reinfer_10m.py --bundle . --repo . --out-dir out_ens10 --images-dir data/test_images --ids-file test_ids.txt --post A2/post.json --device cuda
```
It checks every weight and the selector against SHA-256 values, infers the two five-model groups, averages them, applies C (and A2), and compares the result with `submission_C_10m.csv`.
No training is needed; no file outside this repository (other than the weights and the competition images) is required.

**Resources (measured on one RTX 3090, CUDA 12.8):** 16.3 s/image per five-model pass → about 98 min for the 180 test images (two passes), 7.6 GB GPU memory per pass, about 2.9 GB of temporary probability files per pass. CPU-only/Mac inference works but is slow (measured on an Apple-silicon Mac with MPS: about 218 s per image per five-model pass, i.e. > 20 h for 180 images). The scripts stop automatically if one image takes > 90 s on a GPU-sized budget; raise `--abort-single-seconds` / `--abort-mean-seconds` (or `EMA10_ABORT_*` in the notebook) on slower devices.
**Reproducibility (measured):** with the original inference code, on an RTX 3090 (CUDA 12.8, torch 2.8.0), the regenerated probabilities were bit-identical to the stored reference (`probs_reference_sha256.json`) and the CSV was byte-identical to `submission_C_10m.csv` (180/180 images). The refactored inference code has not yet been checked end-to-end on the full GPU dataset. On other GPUs/precisions small probability differences are expected; the notebook then reports instance matching (IoU) rather than byte equality.

## Weights
The ten weight files (about 1.14 GiB each, fp32) are hosted in the Hugging Face model repository identified by `weights_source.json` and fetched by `scripts/fetch_weights.py`, which verifies each file against `weights_manifest_10m.json`. (A GitHub copy of the code, without weights, links here.) The per-weight training manifests (`*/ema_0.999/manifest.json`) are in this repository.
Notes on provenance (details in `PROVENANCE.md`): the manifest of the seed-43 fold-3 weights was lost and reconstructed (flagged in `weights_manifest_10m.json`); the training git commit is unknown, so the original training runs cannot be reproduced exactly (GPU training is also non-deterministic). Inference from the released weights is reproducible as described above.

## Training (not needed for reproduction)
`scripts/train_fold.py` and `configs/` describe the training procedure; `training/` holds the split, nested split, and inner-selection records (image IDs and hashes). The training data (MAGFiLO train set) is not included.

## Repository layout
`src/filament/` library · `scripts/` inference, selector, checks, training · `configs/` model configs · `C_10m/` selector (pickle + SHA-256 + training records) · `A2/` fixed-rule post-processing · `notebooks/` pipeline notebook · `report/` technical report · `reference/`-style files at the root (`submission_C_10m.csv`, `probs_reference_sha256.json`, `test_ids.txt`).

## License and notices
Code: MIT (`LICENSE`). The weights derive from DINOv3 (Meta, DINOv3 License) and were trained on competition data; see `NOTICE.md` before reuse.

### Re-running and checking inference

`reinfer_10m.py` records the image IDs, input and weight hashes, relevant code, device and library versions in `out_ens10/run_inputs.json`. A restart in the same directory reuses a probability map only after checking its accompanying SHA-256. If inputs or the environment change, use a new output directory. Older output directories without this record are not reused. The hash comparison and CSV comparison describe differences; a `done` status means the pipeline completed, not that its output is identical to the submitted CSV.

Run `python -m unittest discover -s tests -v` for portable inference and cache checks. In the development workspace, saved-probability regression checks additionally compare the earlier implementation and a five-model submitted CSV. Neither test replaces an end-to-end ten-model GPU check.

After a full CUDA inference on the reference environment, run `python scripts/verify_10model_regression.py out_ens10 .` from the bundle root. This requires all 180 images and checks all three probability maps and the submitted C CSV byte-for-byte; saved-map mode cannot pass.
