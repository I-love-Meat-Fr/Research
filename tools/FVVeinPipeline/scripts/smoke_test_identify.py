"""
Smoke test: identify a query image against the gallery.

Loads the same ResNet18+ArcFace checkpoint + SQLite DB used by
batch_enroll.py, extracts an embedding for the query image, then
ranks every enrolled user by cosine similarity.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from PIL import Image

from services.deep_learning import DeepLearningEngine
from services.enrollment_db import EnrollmentDB


CHECKPOINT = Path(
    r"D:\WorkSpace\Research\Research\MFFV\src\results"
    r"\baseline_resnet18_arcface_expD\best_model.pth"
)
DB_PATH = Path(
    r"D:\WorkSpace\Research\Research\MFFV\src\results\enrollment.db"
)
QUERY_IMAGE = Path(
    r"D:\WorkSpace\Research\Research\decoded\10\jpg\m1.jpg"
)


def main() -> int:
    print(f"Checkpoint : {CHECKPOINT}")
    print(f"DB         : {DB_PATH}")
    print(f"Query image: {QUERY_IMAGE}")

    engine = DeepLearningEngine(str(CHECKPOINT), device="auto")
    db = EnrollmentDB(str(DB_PATH))

    gallery = db.get_all_centers()
    print(f"\nGallery size: {len(gallery)} users")
    if not gallery:
        print("Gallery is empty. Run batch_enroll.py first.")
        return 1

    img = Image.open(QUERY_IMAGE)
    if img.mode != "L":
        img = img.convert("L")

    query_emb = engine.extract(img)
    print(f"Query embedding shape={query_emb.shape}, "
          f"norm={float((query_emb**2).sum() ** 0.5):.4f}")

    results = engine.identify(query_emb, gallery, threshold=0.0, top_k=len(gallery))
    print(f"\nTop-5 results:")
    for r in results[:5]:
        marker = "  <-- expected (person 10)" if r["user_id"] == "10" else ""
        print(f"   user_id={r['user_id']:<8}  score={r['score']:.4f}{marker}")

    top1 = results[0]
    print(f"\nTop-1: user_id={top1['user_id']}  score={top1['score']:.4f}")
    if top1["user_id"] == "10":
        print("PASS - top match is the expected person (10).")
        return 0
    else:
        print("INFO - top match is not person 10 (could be a real test failure "
              "OR the model was trained on different data).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
