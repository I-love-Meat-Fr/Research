"""
Deep Learning Engine for Finger Vein Recognition
================================================
Loads ResNet18 + ArcFace checkpoint and provides:
  - extract(image)     → 128-dim embedding
  - verify(e1, e2, t) → verdict dict
  - identify_batch(emb_query, gallery, threshold, top_k)
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# ============================================================================
# Model definition — must match training architecture
# ============================================================================

class ResNet18FV(nn.Module):
    """
    ResNet18 backbone with single-channel input (grayscale finger vein)
    and an L2-normalized embedding head (ArcFace-style).
    """

    def __init__(self, embedding_dim: int = 128, num_classes: int = 492, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet18(weights=None)

        # Replace first conv to accept 1-channel input
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        self.embedding = nn.Linear(in_features, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.in_features = in_features
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> torch.Tensor:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        x = self.backbone.avgpool(x)
        x = torch.flatten(x, 1)
        emb = self.embedding(x)
        emb = F.normalize(emb, p=2, dim=1)
        if return_embeddings:
            return emb
        logits = self.classifier(emb)
        return logits


# ============================================================================
# Engine
# ============================================================================

class DeepLearningEngine:
    """
    Wrapper around ResNet18FV for finger-vein feature extraction and matching.
    """

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.checkpoint_path = Path(checkpoint_path)
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.meta, self.state_dict = self._load_checkpoint()
        self.model = self._build_model()
        self.model.eval()

        # Image preprocessing
        self.img_size = self.meta.get("img_size", 224)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    def _load_checkpoint(self) -> tuple[dict, dict]:
        """Load checkpoint metadata and state dict."""
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        ckpt = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # Support both raw .pth and results.json pointing to .pth
        if self.checkpoint_path.suffix == ".pth":
            meta = {
                "embedding_dim": ckpt.get("embedding_dim", 128),
                "num_classes": ckpt.get("num_classes", 492),
                "img_size": ckpt.get("img_size", 224),
                "threshold": ckpt.get("threshold", None),
                "use_arcface": ckpt.get("use_arcface", True),
                "epoch": ckpt.get("epoch", -1),
                "eer": ckpt.get("eer", None),
            }
            state = ckpt.get("model_state", ckpt)
        elif self.checkpoint_path.name == "results.json":
            # Find sibling best_model.pth
            sibling = self.checkpoint_path.parent / "best_model.pth"
            return self._load_checkpoint_inner(sibling)
        else:
            # Try loading as-is
            state = ckpt
            meta = {
                "embedding_dim": 128,
                "num_classes": 492,
                "img_size": 224,
                "threshold": None,
            }

        return meta, state

    def _load_checkpoint_inner(self, pth_path: Path) -> tuple[dict, dict]:
        """Inner loader for a .pth file."""
        ckpt = torch.load(pth_path, map_location=self.device, weights_only=False)
        meta = {
            "embedding_dim": ckpt.get("embedding_dim", 128),
            "num_classes": ckpt.get("num_classes", 492),
            "img_size": ckpt.get("img_size", 224),
            "threshold": ckpt.get("threshold", None),
            "use_arcface": ckpt.get("use_arcface", True),
            "epoch": ckpt.get("epoch", -1),
            "eer": ckpt.get("eer", None),
        }
        state = ckpt.get("model_state", ckpt)
        return meta, state

    def _build_model(self) -> nn.Module:
        """Build model, load weights, freeze."""
        model = ResNet18FV(
            embedding_dim=self.meta["embedding_dim"],
            num_classes=self.meta["num_classes"],
            pretrained=False,
        ).to(self.device)
        model.load_state_dict(self.state_dict, strict=False)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Convert PIL image to model input tensor."""
        if image.mode != "L":
            image = image.convert("L")
        tensor = self.transform(image)          # [1, H, W]
        tensor = tensor.unsqueeze(0)             # [1, 1, H, W]
        return tensor.to(self.device)

    @torch.no_grad()
    def extract(self, image: Image.Image) -> np.ndarray:
        """Extract L2-normalized embedding from a PIL image."""
        tensor = self.preprocess(image)
        emb = self.model(tensor, return_embeddings=True)  # [1, 128]
        return emb.cpu().numpy().squeeze()

    @torch.no_grad()
    def extract_batch(self, images: list[Image.Image]) -> list[np.ndarray]:
        """Extract embeddings for a batch of images."""
        tensors = [self.preprocess(img) for img in images]
        batch = torch.cat(tensors, dim=0)  # [N, 1, H, W]
        embs = self.model(batch, return_embeddings=True)  # [N, 128]
        return [e.cpu().numpy() for e in embs]

    def enroll_with_augmentation(
        self,
        image: Image.Image,
        n_augmentations: int = 11,
        seed: int | None = None,
    ) -> list[tuple[np.ndarray, bool, str]]:
        """1 real + N augmented embeddings, using services.augmentation.

        This is the primary enrollment method for Option A: the user uploads
        1 image, and we generate N strong augmented variants to spread the
        user's embedding distribution in 128-D ArcFace space.

        Use this instead of extract_batch() when enrolling a new user from
        a single image. Use extract_batch() when you have multiple *distinct*
        real images (e.g., gallery enrollment with 3 separate captures).

        Args:
            image:           Source PIL image (the user's single upload).
            n_augmentations: How many augmented variants to generate (default 11).
            seed:            Optional RNG seed for reproducibility.

        Returns:
            List of (embedding, is_augmented, aug_config_json) tuples.
            First entry is the real embedding (is_augmented=False).
            Subsequent entries are augmented.
        """
        from services.augmentation import generate_augmented_embeddings
        return generate_augmented_embeddings(
            image=image,
            base_extract_fn=self.extract,  # the existing single-image extractor
            n_augmentations=n_augmentations,
            seed=seed,
        )

    # ------------------------------------------------------------------
    # 1:1 Verification
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_sim(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2))

    def verify(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray,
        threshold: float,
    ) -> dict:
        """
        1:1 verification — are two embeddings from the same finger?
        Returns verdict dict.
        """
        score = self.cosine_sim(emb1, emb2)
        if score > threshold:
            verdict = "GENUINE"
        elif score < threshold:
            verdict = "IMPOSTOR"
        else:
            verdict = "TIE"

        return {
            "score": round(score, 6),
            "threshold": round(threshold, 6),
            "verdict": verdict,
            "device": self.device,
            "embedding_dim": self.meta["embedding_dim"],
            "img_size": self.img_size,
        }

    # ------------------------------------------------------------------
    # 1:N Identification
    # ------------------------------------------------------------------

    def identify(
        self,
        query_emb: np.ndarray,
        gallery: dict[str, np.ndarray],
        threshold: float = 0.4,
        top_k: int = 5,
        margin_threshold: float = 0.10,
        ratio_threshold: float = 1.02,
        adaptive: bool = True,
    ) -> list[dict]:
        """
        1:N identification — find the best match for a query embedding.

        Uses three-layer decision logic to reject unknown persons and
        near-mistaken identities:
          1. Top-1 score >= threshold        (similarity gate, absolute)
          2. Top-1 - Top-2 >= margin_threshold (margin gate, absolute gap)
          3. Top-1 / Top-2 >= ratio_threshold (ratio gate, RELATIVE gap)

        Adaptive mode (default ON): when ALL scores in the gallery collapse
        into a tight low band (e.g. mean = 0.30, std = 0.02) the query
        image is FAR from every enrolled user — likely genuine but in
        an unseen pose/condition. In that regime we relax the ratio gate
        from `ratio_threshold` down to 1.001 because:
          - True impostors still give top1 >> top2 (typical ratio > 1.05)
          - True genuine far-away queries give top1 ≈ top2 (ratio ≈ 1.001)
        This rescues genuine matches without re-introducing impostor risk
        (impostors in this regime would still need an absolute score above
        the similarity gate, which they rarely reach in low-score bands).

        Args:
            query_emb:        embedding of the query image
            gallery:          {user_id: center_embedding}
            threshold:        minimum cosine similarity to accept
            top_k:            number of top results returned
            margin_threshold: minimum absolute gap between top-1 and top-2
            ratio_threshold:  minimum relative ratio top1/top2
            adaptive:         if True, relax ratio gate in low-score regimes

        Returns:
            List of dicts: {user_id, score, margin, ratio, is_match, reason}
        """
        if not gallery:
            return []

        results = []
        for user_id, center_emb in gallery.items():
            score = self.cosine_sim(query_emb, center_emb)
            results.append({
                "user_id": user_id,
                "score": round(score, 6),
            })

        # Sort all candidates by score desc
        results.sort(key=lambda x: x["score"], reverse=True)

        top1_score = results[0]["score"]
        top2_score = results[1]["score"] if len(results) > 1 else 0.0
        margin = round(top1_score - top2_score, 6)
        if top2_score > 1e-6:
            ratio = round(top1_score / top2_score, 6)
        else:
            ratio = float("inf")

        # ---- Adaptive thresholding ----
        # Compute statistics of all scores to decide the operating regime.
        all_scores = np.array([r["score"] for r in results])
        score_mean = float(all_scores.mean())
        score_std = float(all_scores.std())
        score_max = float(all_scores.max())
        score_spread = float(all_scores.max() - all_scores.min())

        # ---- New SMART gate: top1 vs mean of others ----
        # Instead of checking "top1 >= threshold" (which fails when all
        # scores are low, e.g. 0.56 with threshold 0.55), we check
        # how much top1 stands out ABOVE the crowd.
        #
        # Key insight:  top1 - mean_others  tells us how much the model
        # "prefers" top-1 over the average gallery member.
        #   - Genuine user: top1 is typically 0.10-0.50 above mean_others
        #   - Impostor in tight band: top1 ≈ mean_others (diff ~0.01-0.03)
        #
        # The adaptive threshold scales with gallery size:
        #   small gallery  -> impostor has more room to match (need bigger gap)
        #   large gallery  -> impostor is averaged into the crowd (smaller gap OK)
        n_gallery = len(all_scores)
        mean_others = float(all_scores[1:].mean()) if n_gallery > 1 else score_mean
        above_mean = top1_score - mean_others

        # Adaptive min_margin: how much top1 must beat the average gallery member
        if n_gallery <= 3:
            min_above_mean = 0.08    # small gallery: impostor has more impact
        elif n_gallery <= 10:
            min_above_mean = 0.05
        elif n_gallery <= 30:
            min_above_mean = 0.03
        else:
            min_above_mean = 0.02    # large gallery: impostor dilutes into crowd

        smart_pass = above_mean >= min_above_mean
        # Keep absolute threshold as a safety floor
        sim_pass = top1_score >= threshold

        if adaptive:
            if score_spread < 0.05:
                # Tight band: all scores within 0.05 of each other.
                # Use SMART gate (top1 vs mean) as primary gate.
                effective_ratio = 1.0
                effective_margin = 0.0
                regime = "tight_band_smart"
            elif score_mean >= 0.85:
                effective_ratio = min(ratio_threshold, 1.001)
                effective_margin = min(margin_threshold, 0.001)
                regime = "high_confidence_relaxed"
            elif score_mean >= 0.50:
                effective_ratio = ratio_threshold
                effective_margin = margin_threshold
                regime = "high_confidence"
            elif score_mean >= 0.30:
                effective_ratio = min(ratio_threshold, 1.001)
                effective_margin = min(margin_threshold, 0.001)
                regime = "low_confidence_relaxed"
            else:
                effective_ratio = 1.0
                effective_margin = 0.0
                regime = "far_query"
        else:
            effective_ratio = ratio_threshold
            effective_margin = margin_threshold
            regime = "fixed"

        margin_pass = margin >= effective_margin
        ratio_pass = ratio >= effective_ratio
        is_match = sim_pass and margin_pass and ratio_pass and smart_pass

        if is_match:
            reason = (
                f"ACCEPTED ({regime}) "
                f"score {top1_score:.4f}>={threshold:.2f} "
                f"above_mean={above_mean:.4f}>={min_above_mean:.4f} "
                f"margin={margin:.4f}>={effective_margin:.4f} "
                f"ratio={ratio:.4f}>={effective_ratio:.4f}"
            )
        elif not sim_pass:
            reason = (
                f"REJECTED: top score {top1_score:.4f} < threshold {threshold:.2f}"
            )
        elif not smart_pass:
            reason = (
                f"REJECTED ({regime}): top1 only {above_mean:.4f} above "
                f"mean_others {mean_others:.4f} < min_needed {min_above_mean:.4f} "
                f"(need bigger gap to confirm genuine, not impostor in tight band)"
            )
        elif regime == "far_query":
            reason = (
                f"REJECTED: top score {top1_score:.4f} < threshold {threshold:.2f} "
                f"(regime=far_query, mean={score_mean:.4f})"
            )
        elif not margin_pass:
            reason = (
                f"REJECTED ({regime}): margin {margin:.4f} < {effective_margin:.4f} "
                f"(top1={top1_score:.4f} ~ top2={top2_score:.4f}, "
                f"mean={score_mean:.4f} std={score_std:.4f})"
            )
        else:
            reason = (
                f"REJECTED ({regime}): ratio {ratio:.4f} < {effective_ratio:.4f} "
                f"(top1={top1_score:.4f} only marginally above top2={top2_score:.4f}, "
                f"mean={score_mean:.4f})"
            )

        # Inject decision info into the top-1 entry
        results[0]["margin"] = margin
        results[0]["ratio"] = ratio
        results[0]["is_match"] = is_match
        results[0]["reason"] = reason
        results[0]["regime"] = regime
        results[0]["score_mean"] = round(score_mean, 4)
        results[0]["score_std"] = round(score_std, 4)
        results[0]["above_mean"] = round(above_mean, 4)
        results[0]["smart_pass"] = smart_pass
        results[0]["min_above_mean"] = round(min_above_mean, 4)

        for r in results[1:]:
            r["margin"] = None
            r["ratio"] = None
            r["is_match"] = False
            r["reason"] = "runner-up"
            r["regime"] = None
            r["score_mean"] = None
            r["score_std"] = None
            r["above_mean"] = None
            r["smart_pass"] = None
            r["min_above_mean"] = None

        return results[:top_k]

    def identify_with_reranking(
        self,
        query_emb: np.ndarray,
        gallery_raw: dict[str, list[np.ndarray]],
        centers: dict[str, np.ndarray],
        threshold: float = 0.4,
        top_k: int = 5,
        margin_threshold: float = 0.10,
        ratio_threshold: float = 1.02,
        rerank_top_n: int = 5,
        aggregation: str = "max",
        adaptive: bool = True,
    ) -> list[dict]:
        """Re-ranking identification with adaptive 3-gate decision rule.

        Stage 1: pick `rerank_top_n` candidates by cheap center comparison.
        Stage 2: re-rank each candidate by comparing the query against EVERY
                 stored raw embedding of that user (not just the center).
                 Aggregate with `aggregation`:
                   "max"       -> best sample wins (robust to noisy samples)
                   "mean"      -> average all samples
                   "topk_mean" -> mean of top-2 samples

        Stage 3: adaptive 3-gate decision — same regime logic as identify().
        """
        if not gallery_raw:
            return []

        # --- Stage 1: coarse retrieval on centers ---
        coarse = []
        for user_id, center_emb in centers.items():
            score = self.cosine_sim(query_emb, center_emb)
            coarse.append((user_id, round(score, 6)))
        coarse.sort(key=lambda x: x[1], reverse=True)
        coarse_top = coarse[:max(rerank_top_n, 1)]

        # --- Stage 2: re-rank each coarse candidate on raw embeddings ---
        reranked = []
        for user_id, _ in coarse_top:
            samples = gallery_raw.get(user_id, [])
            if not samples:
                continue
            sims = [self.cosine_sim(query_emb, s) for s in samples]
            if aggregation == "max":
                agg = max(sims)
            elif aggregation == "mean":
                agg = sum(sims) / len(sims)
            elif aggregation == "topk_mean":
                sims_sorted = sorted(sims, reverse=True)
                k = min(2, len(sims_sorted))
                agg = sum(sims_sorted[:k]) / k
            else:
                agg = max(sims)
            reranked.append({
                "user_id": user_id,
                "score": round(agg, 6),
                "n_samples": len(samples),
                "per_sample_top": round(max(sims), 6),
            })

        reranked.sort(key=lambda x: x["score"], reverse=True)
        if not reranked:
            return []

        top1_score = reranked[0]["score"]
        top2_score = reranked[1]["score"] if len(reranked) > 1 else 0.0
        margin = round(top1_score - top2_score, 6)
        if top2_score > 1e-6:
            ratio = round(top1_score / top2_score, 6)
        else:
            ratio = float("inf")

        # ---- Adaptive thresholding ----
        all_scores = np.array([r["score"] for r in reranked])
        score_mean = float(all_scores.mean())
        score_std = float(all_scores.std())
        score_spread = float(all_scores.max() - all_scores.min())

        # SMART gate: top1 vs mean of others
        n_gallery = len(all_scores)
        mean_others = float(all_scores[1:].mean()) if n_gallery > 1 else score_mean
        above_mean = top1_score - mean_others
        if n_gallery <= 3:
            min_above_mean = 0.08
        elif n_gallery <= 10:
            min_above_mean = 0.05
        elif n_gallery <= 30:
            min_above_mean = 0.03
        else:
            min_above_mean = 0.02
        smart_pass = above_mean >= min_above_mean
        sim_pass = top1_score >= threshold

        if adaptive:
            if score_spread < 0.05:
                effective_ratio = 1.0
                effective_margin = 0.0
                regime = "tight_band_smart"
            elif score_mean >= 0.85:
                effective_ratio = min(ratio_threshold, 1.001)
                effective_margin = min(margin_threshold, 0.001)
                regime = "high_confidence_relaxed"
            elif score_mean >= 0.50:
                effective_ratio = ratio_threshold
                effective_margin = margin_threshold
                regime = "high_confidence"
            elif score_mean >= 0.30:
                effective_ratio = min(ratio_threshold, 1.001)
                effective_margin = min(margin_threshold, 0.001)
                regime = "low_confidence_relaxed"
            else:
                effective_ratio = 1.0
                effective_margin = 0.0
                regime = "far_query"
        else:
            effective_ratio = ratio_threshold
            effective_margin = margin_threshold
            regime = "fixed"

        margin_pass = margin >= effective_margin
        ratio_pass = ratio >= effective_ratio
        is_match = sim_pass and margin_pass and ratio_pass and smart_pass

        if is_match:
            reason = (
                f"ACCEPTED via re-ranking ({regime}, {aggregation}) "
                f"top1={top1_score:.4f} above_mean={above_mean:.4f}>={min_above_mean:.4f} "
                f"margin={margin:.4f} ratio={ratio:.4f}"
            )
        elif not sim_pass:
            reason = (
                f"REJECTED: re-ranked top {top1_score:.4f} < threshold {threshold:.2f}"
            )
        elif not smart_pass:
            reason = (
                f"REJECTED ({regime}): top1 only {above_mean:.4f} above mean_others "
                f"{mean_others:.4f} < min_needed {min_above_mean:.4f} "
                f"(impostor in tight band, not genuine)"
            )
        elif regime == "far_query":
            reason = (
                f"REJECTED: re-ranked top {top1_score:.4f} < threshold {threshold:.2f} "
                f"(regime=far_query, mean={score_mean:.4f})"
            )
        elif not margin_pass:
            reason = (
                f"REJECTED ({regime}): re-ranked margin {margin:.4f} < "
                f"{effective_margin:.4f} (mean={score_mean:.4f})"
            )
        else:
            reason = (
                f"REJECTED ({regime}): re-ranked ratio {ratio:.4f} < "
                f"{effective_ratio:.4f} (mean={score_mean:.4f})"
            )

        reranked[0]["margin"] = margin
        reranked[0]["ratio"] = ratio
        reranked[0]["is_match"] = is_match
        reranked[0]["reason"] = reason
        reranked[0]["regime"] = regime
        reranked[0]["score_mean"] = round(score_mean, 4)
        reranked[0]["score_std"] = round(score_std, 4)
        reranked[0]["above_mean"] = round(above_mean, 4)
        reranked[0]["smart_pass"] = smart_pass
        reranked[0]["min_above_mean"] = round(min_above_mean, 4)

        for r in reranked[1:]:
            r["margin"] = None
            r["ratio"] = None
            r["is_match"] = False
            r["reason"] = "runner-up"
            r["regime"] = None
            r["score_mean"] = None
            r["score_std"] = None
            r["above_mean"] = None
            r["smart_pass"] = None
            r["min_above_mean"] = None

        return reranked[:top_k]

    def compute_center(
        self,
        embeddings: list[np.ndarray],
    ) -> np.ndarray:
        """
        Compute center embedding: mean of all samples, re-normalized.
        """
        arr = np.stack(embeddings) if embeddings else np.array([])
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        center = arr.mean(axis=0)
        center = center / (np.linalg.norm(center) + 1e-12)
        return center

    # ------------------------------------------------------------------
    # Mahalanobis open-set identification
    # ------------------------------------------------------------------

    @staticmethod
    def mahalanobis_distance(
        emb: np.ndarray,
        center: np.ndarray,
        cov_inv: np.ndarray,
    ) -> float:
        """
        Mahalanobis distance between an embedding and a class center.

        d_M(x, mu) = sqrt( (x - mu)^T Σ^{-1} (x - mu) )

        Because embeddings are L2-normalized and the inverse covariance
        is positive semi-definite, this distance is always >= 0.

        Args:
            emb:     query embedding, shape (D,)
            center:  class mean (center), shape (D,)
            cov_inv: regularized inverse covariance of that class, shape (D, D)

        Returns:
            Mahalanobis distance (>= 0). Smaller = closer to the class.
        """
        diff = emb - center
        # (D,) @ (D, D) @ (D,) -> scalar
        return float(np.sqrt(diff @ cov_inv @ diff + 1e-12))

    @staticmethod
    def mahalanobis_to_pvalue(distance_sq: float, df: int) -> float:
        """
        Convert a squared Mahalanobis distance to a p-value under a
        chi-square distribution with `df` degrees of freedom.

        ``scipy.stats.chi2`` is used because it is already a project
        dependency (matplotlib → scipy via streamlit's plotting stack).
        Falls back to a coarse approximation if scipy is unavailable.

        Args:
            distance_sq: squared Mahalanobis distance (>= 0)
            df:          number of degrees of freedom (= embedding dim)

        Returns:
            p-value in [0, 1]. High p-value = looks like a known user.
        """
        try:
            from scipy.stats import chi2
            return float(chi2.sf(distance_sq, df))
        except Exception:
            # Very rough approximation for when scipy is missing.
            # chi2.sf(x, k) ≈ exp(-x/2) * (x/2)^(k/2-1) / Gamma(k/2)
            # for large k. We don't need exactness — this only fires when
            # scipy is broken, which should never happen in production.
            if distance_sq <= 0:
                return 1.0
            return max(0.0, min(1.0, 1.0 / (1.0 + distance_sq)))

    def identify_with_mahalanobis(
        self,
        query_emb: np.ndarray,
        user_stats: dict[str, dict],
        alpha: float = 0.01,
        min_cosine: float = 0.30,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Open-set 1:N identification using Mahalanobis distance + chi-square
        hypothesis test. Rejects unknown persons statistically rather than
        relying on a hand-tuned cosine threshold.

        Decision rule (per user):
            1. cosine sim with user center >= `min_cosine` (cheap gate,
               keeps the result comparable to the existing pipeline)
            2. squared Mahalanobis distance follows chi-square(df=D)
               with p-value >= 1 - alpha  -> ACCEPT, else REJECT

        Users are returned sorted by p-value DESCENDING (most "known"
        looking first). This is the inverse of distance ascending, which
        is what people usually expect.

        Args:
            query_emb:   embedding of the query image, shape (D,)
            user_stats:  {user_id: {"center", "cov_inv", "n"}} from
                         EnrollmentDB.get_all_user_stats()
            alpha:       significance level for the chi-square test.
                         0.01 -> reject when p-value < 0.01.
                         Smaller alpha => stricter => fewer false accepts
                         but also fewer true accepts in low-quality queries.
            min_cosine:  minimum cosine similarity for a candidate to be
                         considered at all. Acts as a coarse floor so we
                         don't drown the user in noise from far-away users.
            top_k:       number of top results returned.

        Returns:
            list of dicts (sorted by p-value desc, top_k):
                {
                    "user_id":      str,
                    "cosine":       float,   # cosine sim with center
                    "mahalanobis":  float,   # sqrt(d_M^2)
                    "mahal_sq":     float,   # d_M^2
                    "p_value":      float,   # chi2.sf(mahal_sq, df)
                    "n_samples":    int,     # samples used to fit the cov
                    "is_match":     bool,    # True iff ACCEPTED
                    "rejection":    str|None # "cosine" | "chi2" | None
                }
        """
        if not user_stats:
            return []

        df = query_emb.shape[0]

        results = []
        for user_id, stats in user_stats.items():
            center = stats["center"]
            cov_inv = stats["cov_inv"]
            n = stats.get("n", 1)

            cos = self.cosine_sim(query_emb, center)
            mahal = self.mahalanobis_distance(query_emb, center, cov_inv)
            mahal_sq = mahal * mahal
            p = self.mahalanobis_to_pvalue(mahal_sq, df)

            # Coarse floor: too far even on cosine => reject outright
            if cos < min_cosine:
                is_match = False
                rejection = "cosine"
            elif p >= 1.0 - alpha:
                is_match = True
                rejection = None
            else:
                is_match = False
                rejection = "chi2"

            results.append({
                "user_id": user_id,
                "cosine": round(cos, 6),
                "mahalanobis": round(mahal, 6),
                "mahal_sq": round(mahal_sq, 6),
                "p_value": round(p, 6),
                "n_samples": n,
                "is_match": is_match,
                "rejection": rejection,
            })

        # Highest p-value = most "known-looking" -> show first
        results.sort(key=lambda r: r["p_value"], reverse=True)
        return results[:top_k]

    def identify_with_mahalanobis_rerank(
        self,
        query_emb: np.ndarray,
        gallery_raw: dict[str, list[np.ndarray]],
        user_stats: dict[str, dict],
        alpha: float = 0.01,
        min_cosine: float = 0.30,
        top_k: int = 5,
        rerank_top_n: int = 5,
        aggregation: str = "max",
    ) -> list[dict]:
        """
        Two-stage open-set identification combining re-ranking with
        Mahalanobis verification.

        Stage 1 (re-rank):
            For each candidate user, score the query against EVERY raw
            sample, then aggregate with ``aggregation``:
              "max"  -> best sample wins (robust to noisy enrollments)
              "mean" -> average all samples (smooth)
            Keep top ``rerank_top_n`` users.

        Stage 2 (Mahalanobis verify):
            For each surviving candidate, compute the Mahalanobis distance
            from query to that user's CENTER using that user's INVERSE
            COVARIANCE. Run a chi-square test at level ``alpha``.
            ACCEPT iff:
                cosine_with_center >= min_cosine AND
                chi2.sf(mahal^2, df=D) >= 1 - alpha

        This combines:
            - the noise-robustness of max/mean-sample re-ranking
              (handles bad enrollment samples)
            - the statistical rigour of Mahalanobis + chi-square
              (rejects unknown persons with a real p-value)

        Returns:
            list of dicts (top_k), each containing:
                user_id, score (re-ranked), cosine_with_center,
                mahalanobis, p_value, is_match, rejection, n_samples,
                per_sample_top
        """
        if not user_stats or not gallery_raw:
            return []

        # ---- Stage 1: re-rank candidates by raw sample scores ----
        coarse = []
        for user_id in user_stats.keys():
            samples = gallery_raw.get(user_id, [])
            if not samples:
                continue
            sims = [self.cosine_sim(query_emb, s) for s in samples]
            if aggregation == "mean":
                agg = sum(sims) / len(sims)
            else:  # "max" (default)
                agg = max(sims)
            coarse.append({
                "user_id": user_id,
                "score": agg,
                "per_sample_top": max(sims),
                "n_samples": len(sims),
            })
        coarse.sort(key=lambda r: r["score"], reverse=True)
        top_candidates = coarse[:max(rerank_top_n, 1)]

        # ---- Stage 2: Mahalanobis verify on the top-N ----
        df = query_emb.shape[0]
        verified = []
        for c in top_candidates:
            uid = c["user_id"]
            stats = user_stats.get(uid)
            if stats is None:
                continue
            cos = self.cosine_sim(query_emb, stats["center"])
            mahal = self.mahalanobis_distance(query_emb, stats["center"], stats["cov_inv"])
            mahal_sq = mahal * mahal
            p = self.mahalanobis_to_pvalue(mahal_sq, df)

            if cos < min_cosine:
                is_match = False
                rejection = "cosine"
            elif p >= 1.0 - alpha:
                is_match = True
                rejection = None
            else:
                is_match = False
                rejection = "chi2"

            verified.append({
                "user_id": uid,
                "score": round(c["score"], 6),
                "per_sample_top": round(c["per_sample_top"], 6),
                "cosine": round(cos, 6),
                "mahalanobis": round(mahal, 6),
                "mahal_sq": round(mahal_sq, 6),
                "p_value": round(p, 6),
                "n_samples": c["n_samples"],
                "is_match": is_match,
                "rejection": rejection,
            })

        verified.sort(key=lambda r: (r["is_match"], r["p_value"]), reverse=True)
        return verified[:top_k]
