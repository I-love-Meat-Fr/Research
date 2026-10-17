# `src/` — Unified Training Pipeline

> Cross-domain finger vein recognition pipeline combining 3 SOTA models.

## What's Here

| File | Purpose |
|------|---------|
| `datasets.py` | Unified dataset loader (FV-USM + FingerVeinSyn-5M) |
| `train.py` | Training + evaluation pipeline for 3 models |
| `quick_start.py` | Run smoke tests on all 3 models |
| `third_party/` | Cloned source code from public repos |

## Architecture

```
src/
├── datasets.py              # Unified dataset loader
├── train.py                 # Main training pipeline
├── quick_start.py           # Quick smoke test runner
├── third_party/
│   ├── FDT/                 # SCUT-BIP-Lab/FDT (TIFS 2025)
│   ├── GLA-FD/              # liangying-Ke/GLA-FD (IEEE Access 2025)
│   └── ILCNN/               # liangying-Ke/Finger-vein-recognition (Bioengineering 2023)
└── results/                 # Output metrics, JSON logs
```

## Models

| Model | Source | Status | Used As |
|-------|--------|--------|---------|
| **FDT** | [SCUT-BIP-Lab/FDT](https://github.com/SCUT-BIP-Lab/FDT) | TIFS 2025 | Transformer baseline |
| **GLA-FD** | [liangying-Ke/GLA-FD](https://github.com/liangying-Ke/GLA-FD) | IEEE Access 2025 | ViT baseline |
| **ILCNN** | [liangying-Ke/Finger-vein-recognition](https://github.com/liangying-Ke/Finger-vein-recognition) | Bioengineering 2023 | Lightweight baseline |

## Quick Start

### 1. Setup (one-time)

```bash
# PyTorch (CPU version, ~200MB) — install via PyPI (NOT pytorch.org)
pip3 install --break-system-packages --user torch torchvision \
    --index-url https://pypi.org/simple/

# Other dependencies
pip3 install --break-system-packages --user \
    scikit-learn pandas matplotlib seaborn pillow opencv-python \
    tqdm ptflops pytorch-model-summary
```

### 2. Smoke Test (verify pipeline)

```bash
# Test all 3 models (1 epoch each, quick)
python3 src/quick_start.py

# Test specific model
PYTHONPATH=src python3 src/train.py --model fdt --epochs 1 --quick_test
```

### 3. Full Training

```bash
# FDT (most expensive)
PYTHONPATH=src python3 src/train.py --model fdt --epochs 20

# GLA-FD
PYTHONPATH=src python3 src/train.py --model gla-fd --epochs 20

# ILCNN (fastest)
PYTHONPATH=src python3 src/train.py --model ilcnn --epochs 20
```

### 4. With Synthetic Pretraining

```bash
PYTHONPATH=src python3 src/train.py --model fdt --epochs 10 --pretrain
```

## Dataset Structure

```
data/
├── external/
│   └── Published_database_FV-USM_Dec2013/
│       ├── 1st_session/raw_data/{user}_{finger}/0{idx}.jpg
│       └── 2nd_session/raw_data/{user}_{finger}/0{idx}.jpg
└── raw/
    └── fingerveinsyn5m_example/  # 100 sample synthetic images
        ├── {identity}_{seq}_{variant}.png
        └── annotations/
```

## Expected Output

After running, `results/{model}/results.json` contains:

```json
{
  "model": "fdt",
  "epochs": 1,
  "num_params": 11280280,
  "final_test": {
    "eer": 0.2840,
    "hter": 0.2680,
    "auc": 0.8059,
    "threshold": 1.0,
    "num_pairs": 500
  }
}
```

## Hardware Notes

- Tested on: Intel i5-13420H, RTX 3050 Mobile (4GB), 15GB RAM
- ✅ **GPU working** (after Secure Boot disabled + `pip install torch --index-url https://download.pytorch.org/whl/cu121`)
- GPU training: ~10-30 it/s with batch_size=8, image_size=112
- 1 epoch ≈ 10-25 seconds (GPU mode)

## Known Limitations

1. ~~GPU disabled~~: ✅ FIXED — GPU training now working (RTX 3050 CUDA enabled)
2. **Synthetic data**: Only 100 sample images (need to download full 5M from Kaggle)
3. **Model wrappers**: Currently use simplified versions
   - FDT: ResNet18 backbone (paper uses Dynamic Transformer)
   - GLA-FD: Falls back to simple version
   - ILCNN: Direct import

## Next Steps

1. ✅ ~~Fix NVIDIA driver for GPU training~~ (DONE)
2. Download full FingerVeinSyn-5M dataset
3. Use original architectures (not simplified backbones)
4. Add ablation studies (data size, architecture, pretrain strategy)

---

## Streamlit Demo App (`src/e2e_app.py`)

An end-to-end GUI that wraps the trained ResNet-18 + ArcFace backbone in a
browser-friendly research demo. Use it to:

* **Register** one or more users with multiple finger-vein samples.
* **Verify** a probe image against a claimed identity (1:1 matching) using
  cosine similarity against a per-user centroid gallery.
* **Browse** the SQLite enrollment database.
* **Inspect** the latest offline evaluation metrics (EER, intra/inter
  cosine means, margin, etc.).

### Run

```bash
streamlit run src/e2e_app.py \
    --server.headless true \
    --server.port 8501 \
    --server.address 127.0.0.1
```

Open `http://127.0.0.1:8501/` in your browser.

### Required Artifacts

The app expects the following files (all created by the training pipeline):

| Path | Purpose |
|------|---------|
| `src/results/fvusm_resnet18_arcface_v1/best.pth` | Trained checkpoint (ResNet-18 encoder + ArcFace head) |
| `src/results/fvusm_resnet18_arcface_v1/eval_results.json` | Offline EER / cosine stats used for the decision threshold |
| `src/results/enrollment.db` | SQLite store for enrolled users + embeddings (created on first save) |

### How Verification Works

1. **Preprocess** the probe image: grayscale → resize to 128×128 →
   `(x − 0.5) / 0.5` (identical to the training preprocessing).
2. **Embed** with the ResNet-18 backbone (128-D, L2-normalised — matches
   the gallery pipeline used by `compute_eer` in `train_fvusm.py`).
3. **Compare** the probe embedding to the user's centroid gallery using
   cosine similarity (dot product on unit vectors).
4. **Decide** by comparing the score to the threshold derived from
   `eval_results.json` — specifically the midpoint of
   `pos_mean` (intra-class) and `neg_mean` (inter-class) cosine means,
   which approximates the Equal-Error-Rate operating point.

For the current `fvusm_resnet18_arcface_v1` checkpoint
(`EER ≈ 0.108 %`) the threshold is ≈ **0.515**.

### Architecture Alignment

The model class `ResNet18Backbone` inside `e2e_app.py` mirrors the
backbone defined in `train_fvusm.py` and `eval_fvusm.py`
exactly: `backbone` (an `nn.Sequential` of conv blocks) → `embed`
(Linear 512→128) → `bn` (BatchNorm1d). The checkpoint stores these
under the top-level key `ckpt['backbone']`; the separate ArcFace head
under `ckpt['head']` is intentionally **not** loaded because the demo
only compares embeddings, never classifies.

If you retrain and change the architecture, update both
`ResNet18Backbone` and the `load_model` function to match.

### Known Quirks

* `meta['threshold']` in the loaded checkpoint is the raw EER value
  (≈ 0.001) and is therefore *not* used as the decision boundary; the
  app prefers the much more interpretable `(pos_mean + neg_mean) / 2`
  from the eval JSON.
* The `@st.cache_resource` decorator means the model is loaded once per
  Streamlit server lifetime. To reload after a checkpoint change, restart
  the server (the cache invalidates only on source-file changes).
