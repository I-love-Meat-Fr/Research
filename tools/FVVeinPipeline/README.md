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
├── app.py                       <- main Streamlit app
├── calibrate_thresholds.py      <- recommended threshold finder
├── services/
│   ├── __init__.py
│   ├── deep_learning.py         <- ResNet18+ArcFace inference engine
│   │                               + Mahalanobis open-set identifier
│   └── enrollment_db.py         <- SQLite enrollment database
│                                   + per-user covariance for Mahalanobis
├── scripts/
│   ├── batch_enroll.py          <- bulk-enroll every person under decoded/
│   ├── smoke_test_identify.py   <- smoke test for the cosine identifier
│   └── smoke_test_mahalanobis.py<- smoke test for the Mahalanobis identifier
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

Identification uses a **3-gate decision rule** (all must pass):

| Gate | Formula | Purpose | Default |
|------|---------|---------|---------|
| **1. Similarity** | `top1_score ≥ ident_threshold` | Reject low-quality matches | 0.55 |
| **2. Margin (absolute)** | `top1 − top2 ≥ margin_threshold` | Reject matches where top-1 is barely above top-2 | 0.08 |
| **3. Ratio (relative)** | `top1 / top2 ≥ ratio_threshold` | Reject impostors when all scores are bunched in a tight band (margin ≈ 0 but top-1 is still relatively highest) | 1.02 |

**Why 3 gates?**
- Gate 1 alone fails because high scores (0.85-0.92) appear for impostors too.
- Gates 1+2 fail when the gallery is dense — all scores collapse to a tight band and `top1 − top2 ≈ 0` even for genuine matches.
- Gate 3 catches that case: even when margin is 0.001, if top1 = 0.92 and top2 = 0.91 the ratio 0.92/0.91 ≈ 1.011 still discriminates.

### Calibrating thresholds

Use `calibrate_thresholds.py` to find optimal values from your gallery:

```bash
python calibrate_thresholds.py
# Reads enrolled samples from results/enrollment.db,
# splits them into gallery (70%) + query (30%),
# computes genuine + impostor score distributions,
# and prints the recommended thresholds.
```

Output is also written to `results/calibration.json`.

### Matching strategies

The Identification tab offers 5 strategies:

| Strategy | What it does | When to use |
|----------|--------------|-------------|
| **Center (mean)** | Compare query to per-user mean embedding | Small galleries (≤5 users), diverse samples per user |
| **Re-rank (max)** | Top-N candidates via centers, then re-rank by **best** raw sample | **Default** — robust to noisy enrollment samples |
| **Re-rank (top-2 mean)** | Same as above but uses top-2 samples averaged | Slightly smoother, less aggressive than max |
| **Mahalanobis (open-set)** | Statistical open-set: per-user mean + covariance + chi-square test | **Recommended when impostor scores overlap with genuine** — gives a real p-value for "is this query from user X?" |
| **Mahalanobis + Re-rank** | Two-stage: re-rank candidates by best raw sample, then verify the top-N with Mahalanobis + chi-square | Best of both worlds — robust to noisy enrollment **and** statistically grounded rejection |

#### Mahalanobis open-set identification

The classical 3-gate decision rule (similarity + margin + ratio) is a hand-tuned heuristic. It works most of the time but it has no statistical interpretation — there's no "is this a real match" p-value, only "did we beat three magic numbers".

The Mahalanobis strategy replaces all of that with a **proper statistical test**:

For each enrolled user we keep:

- The user's **mean** embedding `μ_u` (the "center", same as before)
- The user's **covariance** `Σ_u` (a new quantity that captures how the user's samples scatter around their center)
- Its **regularized inverse** `Σ_u⁻¹` (needed for the formula below)

For each query embedding `x` and each user `u`, we compute:

```
d_M(x, μ_u)  =  sqrt( (x - μ_u)^T  Σ_u⁻¹  (x - μ_u) )
```

This is the **Mahalanobis distance**. It's just like Euclidean distance, but it scales each dimension by the user's own spread. A user with tightly clustered enrollment samples will accept only queries very close to the center; a user with more diverse samples will accept a wider region.

If `x` really comes from user `u`, then `d_M²` follows a **chi-square distribution** with `D` degrees of freedom (`D` = embedding dimension, typically 128). So we can run a **one-sample chi-square hypothesis test**:

```
H0: x ~ N(μ_u, Σ_u)   (x really belongs to user u)
Reject H0 if  chi2.sf(d_M², D)  <  alpha
```

When we reject H0 we call the query `UNKNOWN`. The threshold `alpha` is configurable from the sidebar (default `0.01` = 1% false-reject rate at the user's region boundary).

**Why this helps with score overlap:**

| Problem with classical 3-gate | How Mahalanobis fixes it |
|--------------------------------|--------------------------|
| Threshold is global — tuned to average user | Each user gets a personal "region" derived from their own data |
| Cosine 0.8 to imposter looks like cosine 0.8 to genuine | Mahalanobis automatically down-weights directions where the genuine user has high variance |
| No way to know "how confident" we are | The chi-square test produces a real p-value |
| Threshold tuning requires manual sweeps | Just pick `alpha` from theory (0.01 is the standard choice) |

**Practical tips:**

- **3-5 enrollment samples per user** give the best Mahalanobis results. With 1 sample the covariance is identity (the test falls back to cosine behaviour).
- If you see lots of false rejects, raise `alpha` (e.g. to 0.05).
- If you see false accepts, lower `alpha` (e.g. to 0.001).
- The `Minimum cosine (coarse floor)` slider adds a safety belt on top of the chi-square test. Setting it to `0.3` means "if cosine < 0.3 the user is rejected immediately" — useful when you want a cheap, interpretable first gate.

**References:**

- PMAL: Open Set Recognition via Robust Prototype Mining (AAAI 2021) — uses minimum point-to-set Mahalanobis distance for rejection.
- Sub-Center ArcFace (ECCV 2020) — empirically motivated our use of multi-sample enrollment.
- "An Evaluation of Score Level Fusion Approaches" (arXiv 1805.10666) — confirms the value of statistical vs hand-tuned decision rules.
