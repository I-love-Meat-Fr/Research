# Finger Vein E2E Pipeline

Clean, self-contained end-to-end pipeline for finger vein biometric recognition.

## What it does

1. **Enrollment** — Register a new person by uploading 3+ finger vein images → deep-learning embeddings are extracted and stored in SQLite
2. **Identification** — Upload a query image → the system finds the best match among all enrolled users using cosine similarity

## Architecture

- **Deep Learning Model**: ResNet18 backbone + ArcFace head (pre-trained checkpoint from `src/results/`)
- **Enrollment DB**: SQLite — stores raw float32 embedding blobs per sample
- **Center embedding**: Per-user mean of all samples, re-normalized to unit length — used for fast 1:N matching
- **No proprietary DLLs** — runs entirely on open-source PyTorch model

## Quick Start

```bash
cd tools/FVVeinPipeline
pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- Streamlit 1.30+
- Pre-trained checkpoint: `src/results/baseline_resnet18_arcface_expD/best_model.pth`

## Directory Structure

```
tools/FVVeinPipeline/
├── app.py                 ← main Streamlit app
├── services/
│   ├── __init__.py
│   ├── deep_learning.py   ← ResNet18+ArcFace inference engine
│   └── enrollment_db.py   ← SQLite enrollment database
├── requirements.txt
└── README.md
```

## Pipeline Flow

```
User uploads 3+ images
        ↓
Load image → Grayscale → Resize 224×224 → Normalize
        ↓
Extract 128-dim embedding via ResNet18+ArcFace
        ↓
Store: user_id | embedding_blob | sample_idx | source
        ↓
[Identification]
Query image → extract embedding
        ↓
Cosine similarity vs every user's center embedding
        ↓
Rank by score → top-K results → MATCH/UNKNOWN verdict
```

## Threshold

Decision threshold controls the trade-off between false matches and false rejections:

| Threshold | Effect |
|-----------|--------|
| Lower (0.30) | More matches, higher false acceptance |
| Higher (0.60) | Fewer matches, higher false rejection |

Default: 0.40 (tuned on FV-USM test set, EER ≈ 1%).
