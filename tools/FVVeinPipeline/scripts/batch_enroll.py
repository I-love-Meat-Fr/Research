"""
Batch Enrollment Script — Option A
====================================
Walk `decoded/<person_id>/jpg/m*.jpg`, extract a ResNet18+ArcFace embedding
per file, and write everything into the SQLite enrollment database.

Layout assumed (relative to repo root D:\\WorkSpace\\Research\\Research):

    decoded/
        10/jpg/m1.jpg, m2.jpg, m3.jpg     (person 10)
        100/jpg/m1.jpg, m2.jpg, m3.jpg    (person 100)
        ...

Default checkpoint:
    src/results/baseline_resnet18_arcface_expD/best_model.pth

Default DB:
    src/results/enrollment.db

Usage::

    cd tools/FVVeinPipeline
    python scripts/batch_enroll.py                          # use all defaults
    python scripts/batch_enroll.py --decoded ../decoded     # custom decoded root
    python scripts/batch_enroll.py --limit 5                # enroll only first 5 people
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make services importable when run as a script
HERE = Path(__file__).resolve().parent
PIPELINE_ROOT = HERE.parent
sys.path.insert(0, str(PIPELINE_ROOT))

from services.deep_learning import DeepLearningEngine  # noqa: E402
from services.enrollment_db import EnrollmentDB  # noqa: E402


# Hard-coded paths matching the on-disk repo layout.
#   decoded/    is a sibling of MFFV/ at D:\WorkSpace\Research\Research\decoded
#   src/        lives under MFFV/                         D:\WorkSpace\Research\Research\MFFV\src
DEFAULT_DECODED_ROOT = Path(r"D:\WorkSpace\Research\Research\decoded")
DEFAULT_CHECKPOINT = Path(
    r"D:\WorkSpace\Research\Research\MFFV\src\results"
    r"\baseline_resnet18_arcface_expD\best_model.pth"
)
DEFAULT_DB_PATH = Path(
    r"D:\WorkSpace\Research\Research\MFFV\src\results\enrollment.db"
)


def discover_people(decoded_root: Path) -> dict[str, list[Path]]:
    """
    Return {person_id: [list of jpg paths]} for every person folder under
    decoded_root that contains a jpg/ subfolder.
    """
    people: dict[str, list[Path]] = {}
    if not decoded_root.exists():
        return people

    for person_dir in sorted(decoded_root.iterdir()):
        if not person_dir.is_dir():
            continue
        jpg_dir = person_dir / "jpg"
        if not jpg_dir.is_dir():
            continue
        jpgs = sorted(jpg_dir.glob("m*.jpg"))
        if not jpgs:
            continue
        people[person_dir.name] = jpgs

    return people


def enroll_one_person(
    engine: DeepLearningEngine,
    db: EnrollmentDB,
    person_id: str,
    image_paths: list[Path],
) -> tuple[bool, int, str]:
    """Enroll a single person. Returns (ok, num_saved, message)."""
    from PIL import Image

    pil_images: list[Image.Image] = []
    failed: list[str] = []
    for p in image_paths:
        try:
            img = Image.open(p)
            if img.mode != "L":
                img = img.convert("L")
            pil_images.append(img)
        except Exception as e:
            failed.append(f"{p.name}: {e}")

    if len(pil_images) < 1:
        return False, 0, f"no loadable images ({len(failed)} failed)"

    try:
        embeddings = engine.extract_batch(pil_images)
    except Exception as e:
        return False, 0, f"embedding failed: {e}"

    db.upsert_user(person_id, name="")
    saved = 0
    for i, (emb, p) in enumerate(zip(embeddings, image_paths)):
        if db.add_embedding(person_id, emb, sample_idx=i, source_path=str(p)):
            saved += 1

    msg = f"saved {saved}/{len(pil_images)} samples"
    if failed:
        msg += f"; skipped {len(failed)} unreadable"
    return True, saved, msg


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch-enroll all people from decoded/")
    parser.add_argument(
        "--decoded",
        type=Path,
        default=DEFAULT_DECODED_ROOT,
        help="Root folder containing <person_id>/jpg/m*.jpg subfolders.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Path to ResNet18+ArcFace .pth checkpoint.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="SQLite enrollment database (created if missing).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If >0, enroll only the first N people (useful for smoke-tests).",
    )
    args = parser.parse_args()

    decoded_root: Path = args.decoded.resolve()
    checkpoint: Path = args.checkpoint.resolve()
    db_path: Path = args.db.resolve()

    print(f"[1/4] Decoded root : {decoded_root}")
    print(f"[1/4] Checkpoint   : {checkpoint}")
    print(f"[1/4] Database     : {db_path}")

    if not checkpoint.exists():
        print(f"❌ Checkpoint not found: {checkpoint}", file=sys.stderr)
        return 1

    print("\n[2/4] Loading deep-learning model...")
    t0 = time.time()
    engine = DeepLearningEngine(str(checkpoint), device="auto")
    print(f"      loaded in {time.time()-t0:.1f}s - {engine.meta['embedding_dim']}D, "
          f"{engine.device.upper()}, img_size={engine.img_size}")

    print("\n[3/4] Discovering people...")
    people = discover_people(decoded_root)
    if not people:
        print(f"❌ No <person_id>/jpg/m*.jpg folders found under {decoded_root}",
              file=sys.stderr)
        return 1

    items = list(people.items())
    if args.limit > 0:
        items = items[: args.limit]
    print(f"      found {len(people)} people — enrolling {len(items)}")

    db = EnrollmentDB(str(db_path))

    print("\n[4/4] Enrolling...")
    t0 = time.time()
    ok_count = 0
    fail_count = 0
    for idx, (pid, jpgs) in enumerate(items, start=1):
        ok, saved, msg = enroll_one_person(engine, db, pid, jpgs)
        status = "OK " if ok and saved > 0 else "ERR"
        print(f"   [{idx:>4}/{len(items)}] person={pid:<6}  files={len(jpgs):<3}  "
              f"{status} {msg}")
        if ok and saved > 0:
            ok_count += 1
        else:
            fail_count += 1

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Done in {elapsed:.1f}s — {ok_count} enrolled, {fail_count} failed")
    info = db.db_info()
    print(f"DB now holds {info['num_users']} users / {info['num_embeddings']} embeddings "
          f"({info['size_bytes']/1024:.1f} KB)")
    print("=" * 60)
    return 0 if fail_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
