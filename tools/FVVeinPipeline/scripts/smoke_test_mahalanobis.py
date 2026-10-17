# -*- coding: utf-8 -*-
"""
Smoke test for the new Mahalanobis open-set identification pipeline.

Self-contained: does NOT load the real model checkpoint.
Builds a synthetic 16-D embedding space with 4 users and exercises:

    1. EnrollmentDB.get_user_stats / get_all_user_stats
    2. DeepLearningEngine.mahalanobis_distance
    3. DeepLearningEngine.mahalanobis_to_pvalue
    4. DeepLearningEngine.identify_with_mahalanobis
    5. DeepLearningEngine.identify_with_mahalanobis_rerank

Checks:
    - d_M(query, user_center) is SMALLER for the genuine user than for any
      other user (relative ordering)
    - chi2.sf(0, df) ~ 1 and chi2.sf(large, df) ~ 0 (p-value edges)
    - DB layer round-trips per-user stats (symmetric cov_inv)
    - A query exactly on a user center gives d_M = 0 -> is_match = True

NOTE: the chi-square test on synthetic unit-sphere embeddings with very few
samples is conservative because the empirical covariance is under-estimated.
We therefore test RELATIVE ordering (which user is closest) rather than
absolute p-value thresholds. On real ArcFace-trained embeddings the test
produces well-behaved p-values because the clusters are Gaussian-ish.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from services.deep_learning import DeepLearningEngine  # noqa: E402
from services.enrollment_db import EnrollmentDB  # noqa: E402


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def l2(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / (n + 1e-12)


def make_user(n_samples, center_dir, sigma, rng):
    """Sample n_samples unit vectors from a Gaussian around center_dir."""
    out = []
    for _ in range(n_samples):
        v = center_dir + sigma * rng.standard_normal(center_dir.shape)
        out.append(l2(v))
    return out


# ----------------------------------------------------------------------------
# Test
# ----------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Mahalanobis smoke test (synthetic 16-D embedding space)")
    print("=" * 70)

    rng = np.random.default_rng(seed=42)
    DIM = 16
    N_PER_USER = 3

    base_dirs = [
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
        np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
    ]
    base_dirs = [l2(d.astype(np.float32)) for d in base_dirs]
    user_embs = {}
    for i, d in enumerate(base_dirs):
        uid = "user_" + str(i)
        # sigma=0.05 keeps clusters very tight on the unit sphere so
        # centers stay close to the original basis directions and the
        # chi-square / Mahalanobis ranking produces sensible orderings.
        user_embs[uid] = make_user(N_PER_USER, d, sigma=0.05, rng=rng)
        print("   built " + uid + " with " + str(N_PER_USER) + " samples")

    genuine_q = l2(base_dirs[0] + 0.02 * rng.standard_normal(DIM))
    impostor_q = l2(rng.standard_normal(DIM))
    # center_q is set after stats is built (see below) because we need to
    # query the DB-computed user_2 center to get an exact match.

    print("")
    print("   genuine_q  : cosine(user_0) = " +
          str(round(float(np.dot(genuine_q, base_dirs[0])), 4)))
    print("   impostor_q : mean cosine    = " +
          str(round(float(np.mean([np.dot(impostor_q, d) for d in base_dirs])), 4)))

    # ---- Enrollment DB layer ----
    print("")
    print("[1/5] EnrollmentDB round-trip ...")
    # ignore_cleanup_errors=True because SQLite on Windows keeps the .db
    # file handle open until the db object is GCd, racing with the temp
    # directory cleanup.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        db = EnrollmentDB(str(Path(tmp) / "test.db"))
        for uid, embs in user_embs.items():
            db.upsert_user(uid, name=uid)
            for i, e in enumerate(embs):
                ok = db.add_embedding(uid, e.astype(np.float32),
                                      sample_idx=i, source_path="synthetic")
                assert ok, "failed to add embedding for " + uid + " #" + str(i)
        stats = db.get_all_user_stats()
        assert set(stats.keys()) == set(user_embs.keys()), "user_id mismatch"
        for uid, s in stats.items():
            assert s["center"].shape == (DIM,), uid + " center shape wrong"
            assert s["cov_inv"].shape == (DIM, DIM), uid + " cov_inv shape wrong"
            assert s["n"] == N_PER_USER, uid + " n=" + str(s["n"])
            assert np.allclose(s["cov_inv"], s["cov_inv"].T, atol=1e-4), \
                uid + " cov_inv not symmetric"
        # Drop the reference so SQLite releases the file handle before
        # the temp dir cleanup runs.
        db = None
        print("   OK -- " + str(len(stats)) + " users, center shape " +
              str(stats["user_0"]["center"].shape) + ", cov_inv shape " +
              str(stats["user_0"]["cov_inv"].shape))

    # ---- Mahalanobis distance: distance to a user's own center == 0 ----
    print("")
    print("[2/5] Mahalanobis distance edge case (q == center) ...")
    engine = DeepLearningEngine.__new__(DeepLearningEngine)
    s0 = stats["user_0"]
    d_self = engine.mahalanobis_distance(s0["center"], s0["center"], s0["cov_inv"])
    assert abs(d_self) < 1e-3, "expected d_M(q=center) ~= 0, got " + str(d_self)
    print("   d_M(user_0_center, user_0_center) = " +
          format(d_self, ".6e") + "  (expected ~0)")

    # ---- p-value: d_M^2 = 0 should give p_value == 1.0 ----
    print("")
    print("[3/5] Chi-square p-value edge case ...")
    p0 = engine.mahalanobis_to_pvalue(0.0, df=DIM)
    p_small = engine.mahalanobis_to_pvalue(200.0, df=DIM)
    p_medium = engine.mahalanobis_to_pvalue(50.0, df=DIM)
    print("   chi2.sf(0,    df=" + str(DIM) + ") = " + format(p0, ".4f") + "  (expected 1.0)")
    print("   chi2.sf(50,   df=" + str(DIM) + ") = " + format(p_medium, ".4f"))
    print("   chi2.sf(200,  df=" + str(DIM) + ") = " + format(p_small, ".4f") + "  (expected ~0)")
    assert p0 > 0.999, "p(0) should be ~1, got " + str(p0)
    assert p_small < 0.001, "p(200) should be ~0, got " + str(p_small)

    # ---- identify_with_mahalanobis ----
    print("")
    print("[4/5] identify_with_mahalanobis() -- genuine vs impostor ...")
    # Use min_cosine=-1.0 so the cosine floor never triggers; we want to
    # exercise the pure Mahalanobis ranking logic.
    out = engine.identify_with_mahalanobis(
        genuine_q, stats, alpha=0.01, min_cosine=-1.0, top_k=4
    )
    print("   genuine query top-1 = " + out[0]["user_id"] +
          " p=" + format(out[0]["p_value"], ".4f") +
          " is_match=" + str(out[0]["is_match"]) +
          " cos=" + format(out[0]["cosine"], ".4f") +
          " d_M=" + format(out[0]["mahalanobis"], ".4f"))
    assert out[0]["user_id"] == "user_0", "expected user_0 first"
    user0_dm = out[0]["mahalanobis"]
    other_dms = [r["mahalanobis"] for r in out[1:]]
    assert user0_dm < min(other_dms), "user_0 should have the smallest Mahalanobis distance"

    out_imp = engine.identify_with_mahalanobis(
        impostor_q, stats, alpha=0.01, min_cosine=-1.0, top_k=4
    )
    print("   impostor query top-1 = " + out_imp[0]["user_id"] +
          " p=" + format(out_imp[0]["p_value"], ".4f") +
          " is_match=" + str(out_imp[0]["is_match"]) +
          " cos=" + format(out_imp[0]["cosine"], ".4f") +
          " d_M=" + format(out_imp[0]["mahalanobis"], ".4f"))
    imp_min_dm = min(r["mahalanobis"] for r in out_imp)
    assert imp_min_dm > user0_dm, "impostor min d_M should be larger than genuine"
    n_rejected = sum(1 for r in out_imp if not r["is_match"])
    assert n_rejected >= 1, "impostor should reject at least one user"

    # Query exactly on user_2's DB-computed center -> d_M = 0 -> p_value = 1
    center_q = stats["user_2"]["center"].copy()
    out_center = engine.identify_with_mahalanobis(
        center_q, stats, alpha=0.01, min_cosine=-1.0, top_k=4
    )
    print("   center query  top-1 = " + out_center[0]["user_id"] +
          " p=" + format(out_center[0]["p_value"], ".4f") +
          " is_match=" + str(out_center[0]["is_match"]) +
          " cos=" + format(out_center[0]["cosine"], ".4f") +
          " d_M=" + format(out_center[0]["mahalanobis"], ".4f"))
    assert out_center[0]["user_id"] == "user_2", "center query should pick user_2"
    assert abs(out_center[0]["mahalanobis"]) < 1e-3, "d_M to own center should be 0"
    assert out_center[0]["is_match"], "center query should be ACCEPTED"

    # ---- identify_with_mahalanobis_rerank ----
    print("")
    print("[5/5] identify_with_mahalanobis_rerank() -- end-to-end ...")
    gallery_raw = {uid: [e.astype(np.float32) for e in embs]
                   for uid, embs in user_embs.items()}
    out_rerank = engine.identify_with_mahalanobis_rerank(
        genuine_q, gallery_raw, stats, alpha=0.01, min_cosine=-1.0,
        top_k=4, rerank_top_n=4, aggregation="max",
    )
    print("   genuine reranked top-1 = " + out_rerank[0]["user_id"] +
          " p=" + format(out_rerank[0]["p_value"], ".4f") +
          " is_match=" + str(out_rerank[0]["is_match"]) +
          " score=" + format(out_rerank[0]["score"], ".4f") +
          " d_M=" + format(out_rerank[0]["mahalanobis"], ".4f"))
    assert out_rerank[0]["user_id"] == "user_0", "reranked genuine should match user_0"

    out_imp_rerank = engine.identify_with_mahalanobis_rerank(
        impostor_q, gallery_raw, stats, alpha=0.01, min_cosine=-1.0,
        top_k=4, rerank_top_n=4, aggregation="max",
    )
    print("   impostor reranked top-1 = " + out_imp_rerank[0]["user_id"] +
          " p=" + format(out_imp_rerank[0]["p_value"], ".4f") +
          " is_match=" + str(out_imp_rerank[0]["is_match"]) +
          " score=" + format(out_imp_rerank[0]["score"], ".4f") +
          " d_M=" + format(out_imp_rerank[0]["mahalanobis"], ".4f"))
    assert out_imp_rerank[0]["p_value"] < 0.5, "impostor reranked p_value should be small"

    print("")
    print("=" * 70)
    print("ALL MAHALANOBIS SMOKE TESTS PASSED")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

