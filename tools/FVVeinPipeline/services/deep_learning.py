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
    ) -> list[dict]:
        """
        1:N identification — find the best match for a query embedding.

        Args:
            query_emb: embedding of the query image
            gallery:  {user_id: center_embedding}
            threshold: decision threshold
            top_k:    number of top results to return

        Returns:
            List of dicts: {user_id, score, is_match} sorted by score desc
        """
        results = []
        for user_id, center_emb in gallery.items():
            score = self.cosine_sim(query_emb, center_emb)
            results.append({
                "user_id": user_id,
                "score": round(score, 6),
                "is_match": score >= threshold,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

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
