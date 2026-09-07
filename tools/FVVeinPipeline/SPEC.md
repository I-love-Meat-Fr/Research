# Finger Vein E2E Pipeline — SPEC

## 1. Overview

A clean, self-contained **end-to-end finger vein recognition pipeline** with two core operations:

1. **Enrollment (Đăng ký)**: Save 3 finger-vein images for a person → extract deep-learning embedding → store in SQLite gallery
2. **Identification (Quét mạch máu)**: Import a query image → compute embedding → find best match in gallery → report who it is

No proprietary DLL dependencies. Runs entirely on open-source models (ResNet18 + ArcFace, pre-trained on finger-vein data).

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Streamlit UI                        │
│  [Enrollment Tab]        [Identification Tab]        │
└──────────────┬──────────────────────┬──────────────┘
               │                      │
        ┌──────▼──────┐       ┌──────▼───────────────┐
        │  Enrollment  │       │   Identification     │
        │  Service     │       │   Service            │
        └──────┬──────┘       └──────┬───────────────┘
               │                      │
        ┌──────▼──────┐       ┌──────▼───────────────┐
        │ EnrollmentDB │       │  DeepLearningEngine  │
        │   (SQLite)   │       │  (ResNet18+ArcFace)  │
        └─────────────┘       └──────────────────────┘
```

---

## 3. Data Model

### SQLite — `enrollment.db`

**Table: `users`**
| Column       | Type    | Constraints            |
|--------------|---------|------------------------|
| user_id      | TEXT    | PRIMARY KEY            |
| name         | TEXT    | nullable               |
| created_at   | TEXT    | ISO 8601 timestamp     |
| num_samples  | INTEGER | default 0              |

**Table: `embeddings`**
| Column       | Type    | Constraints                         |
|--------------|---------|-------------------------------------|
| id           | INTEGER | PRIMARY KEY AUTOINCREMENT           |
| user_id      | TEXT    | NOT NULL, FK → users(user_id)       |
| embedding    | BLOB    | NOT NULL, float32 raw bytes         |
| sample_idx   | INTEGER | NOT NULL (0,1,2 for 3 samples)     |
| source_path  | TEXT    | nullable (original filename)        |
| created_at   | TEXT    | ISO 8601 timestamp                  |

### Gallery (runtime)
- All user embeddings are loaded at startup
- A **center embedding** per user is computed: mean of all samples → re-normalized to unit length
- This center is used for fast 1:N cosine similarity search

---

## 4. Core Services

### 4.1 Enrollment Service

**`enroll_user(user_id, name, images: list[Image])`**

1. Validate ≥ 3 images (strict check; minimum is 3)
2. For each image:
   a. Preprocess: grayscale → resize → normalize
   b. Extract embedding via `DeepLearningEngine.extract(img)`
3. Store each embedding in `embeddings` table with `sample_idx`
4. Update `users.num_samples`
5. Return success / error

### 4.2 Deep Learning Engine

**Model**: ResNet18 backbone + ArcFace head
- Trained checkpoint: `results/baseline_resnet18_arcface_expD/best_model.pth`
- Embedding dimension: 128
- Input size: 224×224 grayscale
- Output: L2-normalized 128-dim vector

**`extract(image: PIL.Image) → np.ndarray`** — single image to embedding
**`extract_batch(images: list[PIL.Image]) → list[np.ndarray]`** — batch extraction
**`verify(emb1, emb2, threshold) → dict`** — 1:1 verification

### 4.3 Identification Service

**`identify(query_image, top_k=5, threshold=0.4) → list[dict]`**

1. Extract query embedding
2. Compute cosine similarity with every user's center embedding
3. Rank by score descending
4. Return top-K with scores, verdicts, and metadata
5. Best match if score > threshold → IDENTIFIED
6. Best match if score ≤ threshold → UNKNOWN

---

## 5. UI (Streamlit)

### 5.1 Enrollment Tab
- User ID input (required)
- Name input (optional)
- Multi-image uploader: "Upload ≥ 3 finger vein images"
- Preview thumbnails of uploaded images
- "Enroll" button → progress → success/error message
- Gallery table showing enrolled users with sample count and creation date

### 5.2 Identification Tab
- Single image uploader: "Query image"
- Preview of query image
- "Identify" button → progress → results
- Top-3 match cards: user_id, name, score, MATCH/UNKNOWN badge
- Score bar chart (all enrolled users)
- Large verdict banner: "✅ Identified: [name] ([user_id])" or "❌ Unknown person"

### 5.3 Sidebar
- Checkpoint path (default: `results/baseline_resnet18_arcface_expD/best_model.pth`)
- Decision threshold slider (default: from checkpoint or 0.40)
- Database path (default: `results/enrollment.db`)
- Gallery count display
- "Clear DB" danger button

---

## 6. File Structure

```
tools/FVVeinPipeline/
├── SPEC.md                    ← this file
├── app.py                     ← main Streamlit app
├── services/
│   ├── __init__.py
│   ├── deep_learning.py       ← ResNet18+ArcFace model, inference
│   └── enrollment_db.py       ← SQLite enrollment DB
├── requirements.txt
└── README.md
```

---

## 7. Acceptance Criteria

- [ ] Enrollment: 3 images → stored in DB → visible in user table
- [ ] Identification: 1 query → cosine sim against all users → ranked results
- [ ] Threshold: similarity > threshold → MATCH, else → NO MATCH
- [ ] UI: clean Streamlit, no proprietary DLLs required
- [ ] No IplImage struct references (removed from old code)
- [ ] Self-contained: runs with `streamlit run app.py`
- [ ] GPU support (cuda if available, fallback to cpu)
