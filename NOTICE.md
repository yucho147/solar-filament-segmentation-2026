# Notices

- **Code** in this repository is released under the MIT License (`LICENSE`).
- **Model weights** (hosted separately, see `weights_source.json`) are fine-tuned from the pretrained DINOv3 ViT-L/16 encoder
  (`timm/vit_large_patch16_dinov3.lvd1689m`). The DINOv3 weights are provided by Meta under the *DINOv3 License*; the fine-tuned weights should be treated as
  subject to the same terms. The DINOv3 License is included in `LICENSE-DINOv3.md`; redistribution of the DINOv3 materials or their derivatives is subject to that license. Please review it before reuse. This repository does not include the DINOv3 weights themselves; the
  pretrained weights are downloaded by `timm` from the Hugging Face Hub only when training from scratch.
- **Data**: the competition data (GONG H-alpha images and MAGFiLO annotations) are not included and remain subject to the competition rules and the MAGFiLO/GONG data
  terms (the competition data may be used for non-commercial purposes). Fine-tuned weights and the selector were trained on those data.
- The `C_10m/final_selector_10m.pkl` file is a pickle. It is checked against a SHA-256 value before loading; do not load pickles from untrusted sources.
