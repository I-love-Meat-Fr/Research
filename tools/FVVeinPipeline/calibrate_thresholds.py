"""Calibration Tool for Identification Thresholds."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from services.deep_learning import DeepLearningEngine
from services.enrollment_db import EnrollmentDB


def load_engine(weights_path):
    try:
        engine = DeepLearningEngine(model_path=weights_path)
        engine.load()
        return engine
    except Exception as e:
        print(f"[warn] Could not load real model ({e}). Using stub.")
        engine = DeepLearningEngine.__new__(DeepLearningEngine)
        engine.model = None
        engine.device = "cpu"
        return engine


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def collect_pairs(engine, db, test_ratio):
    gallery, queries = {}, {}
    for u in db.list_users():
        embs = db.get_user_embeddings(u["user_id"])
        if not embs:
            continue
        vectors = [e[1] for e in embs]
        n = len(vectors)
        cut = max(1, int(n * (1 - test_ratio)))
        gallery[u["user_id"]] = vectors[:cut]
        queries[u["user_id"]] = vectors[cut:] if cut < n else vectors[:1]

    gallery_centers = {}
    for uid, embs in gallery.items():
        c = np.mean(np.stack(embs), axis=0)
        gallery_centers[uid] = c / (np.linalg.norm(c) + 1e-12)

    genuine_scores, impostor_scores = [], []
    genuine_margins, impostor_margins = [], []
    genuine_ratios, impostor_ratios = [], []

    for q_user, q_embs in queries.items():
        sims = {uid: cosine(q_embs[0], c) for uid, c in gallery_centers.items()}
        sorted_users = sorted(sims.keys(), key=lambda u: sims[u], reverse=True)
        top1 = sims[sorted_users[0]]
        top2 = sims[sorted_users[1]] if len(sorted_users) > 1 else 0.0
        margin = top1 - top2
        ratio = top1 / top2 if top2 > 1e-6 else float("inf")

        if sorted_users[0] == q_user:
            genuine_scores.append(top1)
            genuine_margins.append(margin)
            genuine_ratios.append(ratio)
        else:
            impostor_scores.append(top1)
            impostor_margins.append(margin)
            impostor_ratios.append(ratio)

    return (genuine_scores, impostor_scores,
            genuine_margins, impostor_margins,
            genuine_ratios, impostor_ratios)


def find_best_threshold(genuine, impostor, name):
    if not genuine or not impostor:
        return 0.5
    candidates = sorted(set(genuine + impostor))
    best_thr, best_gap = 0.5, -1
    for thr in candidates:
        n_gen_above = sum(1 for s in genuine if s >= thr)
        n_imp_below = sum(1 for s in impostor if s < thr)
        gap = n_gen_above + n_imp_below
        if gap > best_gap:
            best_gap = gap
            best_thr = thr
    print(f"  {name}: genuine[{min(genuine):.4f}, {max(genuine):.4f}] "
          f"vs impostor[{min(impostor):.4f}, {max(impostor):.4f}] "
          f"-> thr={best_thr:.4f}")
    return best_thr


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", default="weights/best_model.pth")
    p.add_argument("--db", default="results/enrollment.db")
    p.add_argument("--test-ratio", type=float, default=0.3)
    args = p.parse_args()

    print(f"[1/4] Loading engine from {args.weights}...")
    engine = load_engine(args.weights)

    print(f"[2/4] Loading database from {args.db}...")
    db = EnrollmentDB(db_path=args.db)
    info = db.db_info()
    print(f"      users: {info['num_users']}, embeddings: {info['num_embeddings']}")
    if info["num_users"] < 3:
        print("[error] Need at least 3 enrolled users.")
        sys.exit(1)

    print(f"[3/4] Building genuine + impostor score distributions...")
    genuine, impostor, gm, im, gr, ir = collect_pairs(engine, db, args.test_ratio)
    print(f"      genuine queries: {len(genuine)}, impostor queries: {len(impostor)}")
    if not genuine or not impostor:
        print("[error] Not enough data.")
        sys.exit(1)

    print("[4/4] Recommended thresholds:")
    sim_thr = find_best_threshold(genuine, impostor, "score  ")
    mar_thr = find_best_threshold(gm, im, "margin ")
    rat_thr = find_best_threshold(gr, ir, "ratio  ")

    print()
    print("=" * 60)
    print("Copy these into the Streamlit sidebar:")
    print(f"   Identification threshold : {sim_thr:.3f}")
    print(f"   Margin threshold         : {mar_thr:.3f}")
    print(f"   Ratio threshold          : {rat_thr:.3f}")
    print("=" * 60)

    Path("results").mkdir(exist_ok=True)
    with open("results/calibration.json", "w") as f:
        json.dump({
            "ident_threshold": sim_thr,
            "margin_threshold": mar_thr,
            "ratio_threshold": rat_thr,
            "n_genuine": len(genuine),
            "n_impostor": len(impostor),
        }, f, indent=2)
    print(f"\nSaved to results/calibration.json")


if __name__ == "__main__":
    main()
