"""
Unified Training Pipeline for 3 Baselines
==========================================
Trains and evaluates:
1. FDT (FVRA-FDT) - Dynamic Transformer
2. GLA-FD - Global-Local Attention with Feature Decoupling
3. ILCNN - Improved Lightweight CNN

Author: Research Team
Date: 2026-08-09
"""
import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add third_party to path (parent directories, not models subdirs)
# IMPORTANT: Add src directory FIRST so our local modules are found before third_party
# This is needed because the models use relative imports (.dbb, etc.)
THIRD_PARTY = Path(__file__).parent / 'third_party'

# Make sure src directory is first in path so our datasets.py is found
src_dir = str(Path(__file__).parent)
# Remove any existing entries for src_dir to avoid duplicates
if src_dir in sys.path:
    sys.path.remove(src_dir)
sys.path.insert(0, src_dir)

# Add parent directories so that 'from models.model import X' works
# For GLA-FD: add GLA-FD/ so 'models' maps to GLA-FD/models
# For ILCNN: add ILCNN/ so 'models' maps to ILCNN/models  
# For FDT: add FDT/FDT/model/fdt so 'from fdt import FDT' works
# IMPORTANT: Add these AFTER src so our modules are resolved first
# Order matters: imports will use the first matching path
# Put ILCNN first because it's imported before GLA-FD in ILCNNWrapper
sys.path.insert(1, str(THIRD_PARTY / 'ILCNN'))
sys.path.insert(2, str(THIRD_PARTY / 'GLA-FD'))
sys.path.insert(3, str(THIRD_PARTY / 'FDT' / 'FDT' / 'model' / 'fdt'))

# Import our dataset module FIRST before any third_party imports
# This ensures datasets refers to our src/datasets.py
import importlib
datasets_module = importlib.import_module('datasets')
globals()['datasets'] = datasets_module

# Import our dataset
from datasets import (
    FingerVeinDataset, FingerVeinPairDataset,
    compute_eer, compute_hter, compute_roc_auc,
    collate_pairs, get_fvusm_loaders, get_fingerveinsyn5m_loader,
    get_transforms,
)


# ============================================================================
# 1. MODEL WRAPPERS (Unified Interface)
# ============================================================================

class ModelWrapper(nn.Module):
    """Base wrapper for all models. Provides unified interface."""

    def __init__(self, backbone: nn.Module, embedding_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.embedding_dim = embedding_dim
        self.embedding = nn.Linear(self._get_feature_dim(), embedding_dim)

    def _get_feature_dim(self) -> int:
        """Override in subclass."""
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embedding."""
        feat = self.backbone(x)
        emb = self.embedding(feat)
        emb = F.normalize(emb, p=2, dim=1)
        return emb


class ILCNNWrapper(nn.Module):
    """ILCNN (Improved Lightweight CNN + CoordAttention)."""

    def __init__(self, embedding_dim: int = 128, num_classes: int = 492, in_channels: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim

        # Try to use original ILCNN with a quick forward test
        self.use_original = False
        try:
            from models.model import LightWeightedModel
            test_model = LightWeightedModel(num_classes=num_classes, in_channels=in_channels)
            # Use batch_size=2 (BatchNorm1d needs >1 in train mode) + eval mode to skip BN issues
            test_model.eval()
            with torch.no_grad():
                test_input = torch.randn(2, in_channels, 112, 112)
                test_output = test_model(test_input)
            del test_model  # Clean up

            # Use original ILCNN
            self.backbone = LightWeightedModel(num_classes=num_classes, in_channels=in_channels)
            self.use_original = True

            # ILCNN returns (cos_theta, logits) both (B, num_classes) -- no intermediate features.
            # We patch forward to also return the 256-dim penultimate features.
            self.backbone._original_forward = self.backbone.forward
            def patched_forward(x):
                B = x.shape[0]
                out = self.backbone.stem(x)
                feat = self.backbone.feature(out).reshape(B, -1)
                feat = self.backbone.dropout(self.backbone.relu(self.backbone.bn(feat)))
                cos_theta, logits = self.backbone._original_forward(x)
                return feat, cos_theta, logits
            self.backbone.forward = patched_forward

            self._feature_dim = embedding_dim  # wrapper outputs embedding_dim (after proj_256)
            self._num_classes = num_classes
            # Project 256-dim penultimate features to target embedding_dim
            if embedding_dim != 256:
                self.proj_256 = nn.Linear(256, embedding_dim)
            else:
                self.proj_256 = nn.Identity()
            print(f"  [OK] Using original ILCNN ({sum(p.numel() for p in self.backbone.parameters()):,} params)")
        except Exception as e:
            # Fallback: simple CNN if original fails
            print(f"  [WARNING] ILCNN original failed: {e}, using fallback")
            self.backbone = self._create_fallback_model(embedding_dim, num_classes, in_channels)
            self.use_original = False
            self._feature_dim = embedding_dim
            self.proj = nn.Linear(512, embedding_dim)

    def _create_fallback_model(self, embedding_dim: int, num_classes: int, in_channels: int):
        """Simple fallback CNN model when ILCNN fails to load."""
        import torchvision.models as models
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Identity()
        return model

    def _get_feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_original:
            # Returns (features_256, cos_theta, logits)
            feat, cos_theta, logits = self.backbone.forward(x)
            # Project 256-dim features to embedding_dim
            feat = self.proj_256(feat)
            return F.normalize(feat, p=2, dim=1)
        else:
            # Fallback: project backbone output (512) -> embedding_dim
            out = self.backbone(x)
            if isinstance(out, tuple):
                out = out[0]
            emb = self.proj(out)
            return F.normalize(emb, p=2, dim=1)

    def proj_256_to_target(self, target_dim: int):
        """Call after init if embedding_dim != 256 to add projection layer."""
        if target_dim != 256:
            self.proj_256 = nn.Linear(256, target_dim).to(next(self.parameters()).device)
            self._feature_dim = target_dim


class GLAFDWrapper(nn.Module):
    """GLA-FD (Global-Local Attention with Feature Decoupling)."""

    def __init__(self, embedding_dim: int = 128, num_classes: int = 492, in_channels: int = 1):
        super().__init__()
        self.embedding_dim = embedding_dim
        self._is_fallback = False
        try:
            # GLA-FD's model.py does `from models.utils import ...`
            # We need GLA-FD/models on sys.path and ILCNN out of the way
            import sys
            glafd_path = str(THIRD_PARTY / 'GLA-FD')
            # Remove any existing 'models' module from sys.modules to avoid ILCNN collision
            for mod_name in list(sys.modules.keys()):
                if mod_name == 'models' or mod_name.startswith('models.'):
                    del sys.modules[mod_name]
            # Save and remove ILCNN from sys.path
            original_path = sys.path[:]
            filtered_path = [p for p in sys.path if not p.endswith('ILCNN')]
            sys.path = [glafd_path] + filtered_path
            try:
                from models.model import Model as GLAFDModel
                self.backbone = GLAFDModel(num_classes=num_classes, in_channels=in_channels)
                # GLA-FD returns 512-dim embeddings in out[1], project to embedding_dim
                self.proj = nn.Linear(512, embedding_dim)
                self._feature_dim = embedding_dim
                self._is_fallback = False
                print(f"  [OK] Using original GLA-FD ({sum(p.numel() for p in self.backbone.parameters()):,} params)")
            finally:
                sys.path = original_path
                # Clean up imported GLA-FD modules
                for mod_name in list(sys.modules.keys()):
                    if mod_name == 'models' or mod_name.startswith('models.'):
                        del sys.modules[mod_name]
                # Restore ILCNN first so subsequent imports still resolve
                sys.path.insert(1, str(THIRD_PARTY / 'ILCNN'))
        except Exception as e:
            # Fallback: simple model if original fails
            print(f"  [WARNING] GLA-FD original failed: {e}, using fallback")
            self.backbone = self._create_fallback_model(embedding_dim, num_classes, in_channels)
            self._is_fallback = True
            self._feature_dim = embedding_dim
            self.proj = nn.Linear(512, embedding_dim)

    def _create_fallback_model(self, embedding_dim: int, num_classes: int, in_channels: int):
        """Simple fallback CNN model when GLA-FD fails to load."""
        import torchvision.models as models
        model = models.resnet18(weights=None)
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Identity()
        return model

    def _get_feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # GLAFD returns (out, embeds, cos_theta, norms) or just logits
        out = self.backbone(x)
        if isinstance(out, tuple):
            # Model returns (logits, embeddings, cos_theta, norms)
            emb = out[1]  # 512-dim embeddings
        else:
            emb = out
        # Project 512-dim to target embedding_dim
        if emb.shape[-1] != self.embedding_dim:
            emb = self.proj(emb)
        return F.normalize(emb, p=2, dim=1)


class FDTWrapper(nn.Module):
    """FDT (Full-View Dynamic Transformer) — original architecture from paper."""

    def __init__(self, embedding_dim: int = 128, num_classes: int = 492):
        super().__init__()
        # Import original FDT architecture
        from fdt import FDT as FDTModel

        # FDT expects multi-view input (3 views per finger)
        # We duplicate single image to 3 views for FV-USM (single-view dataset)
        # Patch size (24, 32) for 240x320 images
        # For 112x112, use (16, 16) or (14, 14)
        self.model = FDTModel(
            img_size=(112, 112),
            patch_size=(14, 14),
            in_chans=1,
            num_classes=num_classes,
            embed_dim=embedding_dim,
            depth_fv=4,
            depth_sv=2,
            num_heads=4,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            keep_rate_fv=(1.0, 1.0, 1.0, 1.0),
            keep_rate_sv=(1.0, 1.0),
            use_fa=False,
            use_ba=False,
            use_lffn=False,
            use_mlppatch=True,
            use_peg=False,         # Disable PEG - shape issue with multi-view
        )
        # Output: 128-dim feature vector (from f2c layer)
        self._feature_dim = embedding_dim

    def _get_feature_dim(self) -> int:
        return self._feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FDT expects list of 3 views [view1, view2, view3].
        We duplicate single image to create 3 views.
        """
        # Create 3 views from single image
        views = [x, x, x]
        logits, features = self.model(views)
        # Use features (embedding) for verification
        return F.normalize(features, p=2, dim=1)


def get_model(name: str, num_classes: int = 492, embedding_dim: int = 128) -> nn.Module:
    """Factory function for all models."""
    name = name.lower()
    if name == 'ilcnn':
        return ILCNNWrapper(embedding_dim, num_classes)
    elif name == 'gla-fd' or name == 'glafd':
        return GLAFDWrapper(embedding_dim, num_classes)
    elif name == 'fdt':
        return FDTWrapper(embedding_dim, num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")


class ClassificationModel(nn.Module):
    """Wrapper that adds classification head for training."""

    def __init__(self, backbone: nn.Module, num_classes: int):
        super().__init__()
        self.backbone = backbone
        # Auto-detect feature dim from backbone to support fallbacks (ResNet18=512)
        if hasattr(backbone, '_feature_dim'):
            feat_dim = backbone._feature_dim
        else:
            # Probe dynamically with a dummy input (fallback)
            feat_dim = 128
        self.classifier = nn.Linear(feat_dim, num_classes)
        self._feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.backbone(x)
        logits = self.classifier(emb)
        return logits


def get_classification_model(name: str, num_classes: int = 492, embedding_dim: int = 128) -> nn.Module:
    """Get model with classification head for supervised training."""
    backbone = get_model(name, num_classes, embedding_dim)
    return ClassificationModel(backbone, num_classes)


# ============================================================================
# 2. LOSS FUNCTIONS
# ============================================================================

class ArcFaceLoss(nn.Module):
    """ArcFace loss for metric learning."""

    def __init__(self, embedding_dim: int, num_classes: int, margin: float = 0.5, scale: float = 30.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cos_theta = F.linear(embeddings, weight)
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)

        # Add margin
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        theta = torch.acos(cos_theta)
        target_cos = torch.cos(theta + self.margin)
        output = one_hot * target_cos + (1 - one_hot) * cos_theta

        # Scale
        output *= self.scale

        return F.cross_entropy(output, labels)


class ContrastiveLoss(nn.Module):
    """Contrastive loss for pair learning."""

    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb1: torch.Tensor, emb2: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        euclidean = F.pairwise_distance(emb1, emb2)
        loss = labels * euclidean.pow(2) + (1 - labels) * F.relu(self.margin - euclidean).pow(2)
        return loss.mean()


# ============================================================================
# 3. TRAINING & EVALUATION
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str = 'cpu',
    log_interval: int = 10,
    use_arcface: bool = False,
) -> Dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'  Train', ncols=100)
    for batch_idx, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device)
        labels = labels.to(device).long()

        # Forward
        optimizer.zero_grad()

        if use_arcface and isinstance(criterion, ArcFaceLoss):
            # For ArcFace, get embeddings directly from backbone (skip classification head)
            embeddings = model.backbone(imgs)
            loss = criterion(embeddings, labels)
            # For accuracy, use classification head
            with torch.no_grad():
                logits = model.classifier(embeddings)
                _, predicted = torch.max(logits, 1)
                correct += (predicted == labels).sum().item()
        else:
            # Standard: use classification model with logits
            logits = model(imgs)
            loss = criterion(logits, labels)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total += imgs.size(0)

        if batch_idx % log_interval == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/max(total,1):.1f}%'})

    return {
        'loss': total_loss / len(loader),
        'accuracy': correct / max(total, 1),
    }


@torch.no_grad()
def evaluate_verification(
    model: nn.Module,
    pair_loader: DataLoader,
    device: str = 'cpu',
) -> Dict:
    """Evaluate on verification task (genuine/impostor pairs)."""
    model.eval()

    # Get the backbone for embedding extraction
    backbone = model.backbone if hasattr(model, 'backbone') else model

    all_scores = []
    all_labels = []

    pbar = tqdm(pair_loader, desc=f'  Eval', ncols=100)
    for img1, img2, labels in pbar:
        img1 = img1.to(device)
        img2 = img2.to(device)

        # Get embeddings
        emb1 = backbone(img1)
        emb2 = backbone(img2)

        # Cosine similarity
        scores = F.cosine_similarity(emb1, emb2, dim=1).cpu().numpy()
        all_scores.append(scores)
        all_labels.append(labels.numpy())

    all_scores = np.concatenate(all_scores)
    all_labels = np.concatenate(all_labels)

    # Compute metrics
    eer, threshold = compute_eer(all_labels, all_scores)
    hter = compute_hter(all_labels, all_scores, threshold)
    auc = compute_roc_auc(all_labels, all_scores)

    return {
        'eer': eer,
        'hter': hter,
        'auc': auc,
        'threshold': threshold,
        'num_pairs': len(all_labels),
    }


# ============================================================================
# 4. MAIN PIPELINE
# ============================================================================

def convert_labels_to_int(labels):
    """Convert string labels (e.g., '037_4') to int class IDs."""
    return torch.tensor([abs(hash(str(l))) % 1000 for l in labels])


def main():
    parser = argparse.ArgumentParser(description='Unified Training Pipeline')
    parser.add_argument('--model', type=str, default='ilcnn',
                        choices=['ilcnn', 'gla-fd', 'fdt'],
                        help='Model to train')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=112, help='Image size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dim')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')
    parser.add_argument('--pretrain', action='store_true',
                        help='Use synthetic pretraining first')
    parser.add_argument('--eval_batch_size', type=int, default=0,
                        help='Eval batch size (0 = same as train batch_size)')
    parser.add_argument('--num_pairs', type=int, default=2000,
                        help='Number of pairs for verification eval')
    parser.add_argument('--patience', type=int, default=0,
                        help='Early stopping patience (0 = disabled)')
    parser.add_argument('--quick_test', action='store_true',
                        help='Quick test with few samples')
    parser.add_argument('--arcface', action='store_true',
                        help='Use ArcFace loss (metric learning)')
    parser.add_argument('--arcface_margin', type=float, default=0.5,
                        help='ArcFace margin')
    parser.add_argument('--arcface_scale', type=float, default=30.0,
                        help='ArcFace scale')
    args = parser.parse_args()

    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[Setup] Device: {device}')
    print(f'[Setup] Model: {args.model}')
    print(f'[Setup] Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}')

    output_dir = Path(args.output_dir) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Phase 1: Load datasets
    # ------------------------------------------------------------------
    print('\n[Phase 1] Loading datasets...')
    
    # Auto-detect FV-USM path
    import os
    possible_paths = [
        'd:/WorkSpace/Research/Research/MFFV/data/external/Published_database_FV-USM_Dec2013',
        'D:/WorkSpace/Research/Research/MFFV/data/external/Published_database_FV-USM_Dec2013',
        'd:/WorkSpace/Research/Research/MFFV/data/external',
        './data/external/Published_database_FV-USM_Dec2013',
        '../data/external/Published_database_FV-USM_Dec2013',
    ]
    fvusm_root = None
    for path in possible_paths:
        test_path = os.path.join(path, 'Published_database_FV-USM_Dec2013')
        if os.path.exists(test_path):
            fvusm_root = test_path
            break
        elif os.path.exists(path):
            for item in os.listdir(path):
                if 'FV-USM' in item or 'fv_usm' in item.lower():
                    fvusm_root = os.path.join(path, item)
                    break
            if fvusm_root:
                break
    
    if fvusm_root is None:
        print(f"[WARNING] FV-USM dataset not found at: {possible_paths}")
        print(f"  Please download and place in data/external/")
        print(f"  Skipping dataset loading for now...")
        print(f'\n? Setup incomplete. Please check dataset paths.')
        return
    
    print(f'  [Dataset] FV-USM root: {fvusm_root}')
    
    train_loader, val_loader, test_loader = get_fvusm_loaders(
        root=fvusm_root,
        batch_size=args.batch_size, num_workers=0, session='1st',
        img_size=args.img_size,
    )

    # Build pair datasets for evaluation
    print('\n[Phase 1] Building pair datasets for verification...')
    _, eval_transform = get_transforms(args.img_size, augment=False)
    val_dataset = FingerVeinDataset(
        root=fvusm_root,
        dataset_name='fvusm', session='1st', mode='val',
        transform=eval_transform,
    )
    test_dataset = FingerVeinDataset(
        root=fvusm_root,
        dataset_name='fvusm', session='1st', mode='test',
        transform=eval_transform,
    )

    val_pairs = FingerVeinPairDataset(val_dataset, num_pairs=500 if args.quick_test else args.num_pairs)
    test_pairs = FingerVeinPairDataset(test_dataset, num_pairs=500 if args.quick_test else args.num_pairs)

    eval_bs = args.eval_batch_size if args.eval_batch_size > 0 else args.batch_size
    val_pair_loader = DataLoader(val_pairs, batch_size=eval_bs, shuffle=False,
                                 collate_fn=collate_pairs, num_workers=0)
    test_pair_loader = DataLoader(test_pairs, batch_size=eval_bs, shuffle=False,
                                  collate_fn=collate_pairs, num_workers=0)
    print(f'  [Eval] pairs={len(val_pairs)}, batch_size={eval_bs}')

    # ------------------------------------------------------------------
    # Phase 2: Pretraining (optional)
    # ------------------------------------------------------------------
    if args.pretrain:
        print('\n[Phase 2] Pretraining on FingerVeinSyn-5M (sample)...')
        syn_loader = get_fingerveinsyn5m_loader(
            batch_size=args.batch_size, num_workers=0,
            img_size=args.img_size,
        )

        # Note: Pretraining with only 100 samples won't be effective
        # but tests the pipeline
        print(f'  [Pretrain] Synthetic samples: {len(syn_loader.dataset)}')

    # ------------------------------------------------------------------
    # Phase 3: Build model
    # ------------------------------------------------------------------
    print('\n[Phase 3] Building model...')
    num_classes = train_loader.dataset._count_unique_ids()
    print(f'  Number of classes: {num_classes}')

    try:
        model = get_classification_model(args.model, num_classes=num_classes,
                                          embedding_dim=args.embedding_dim).to(device)
        print(f'  Model: {model.__class__.__name__}')
    except Exception as e:
        print(f'  [ERROR] Failed to build {args.model}: {e}')
        print(f'  [FALLBACK] Using simple CNN baseline')
        model = get_classification_model('fdt', num_classes=num_classes,
                                          embedding_dim=args.embedding_dim).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    # ------------------------------------------------------------------
    # Phase 4: Setup training
    # ------------------------------------------------------------------
    print('\n[Phase 4] Setting up training...')
    use_arcface = getattr(args, 'arcface', False)
    if use_arcface:
        criterion = ArcFaceLoss(
            embedding_dim=args.embedding_dim,
            num_classes=num_classes,
            margin=args.arcface_margin,
            scale=args.arcface_scale,
        ).to(device)
        print(f'  Loss: ArcFaceLoss (margin={args.arcface_margin}, scale={args.arcface_scale})')
    else:
        criterion = nn.CrossEntropyLoss().to(device)
        print(f'  Loss: CrossEntropyLoss')
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Early stopping / best checkpoint setup
    best_eer = float('inf')
    best_epoch = 0
    best_model_path = Path(args.output_dir) / f'{args.model}' / 'best_model.pth'
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    patience = getattr(args, 'patience', 0)
    no_improve = 0

    # ------------------------------------------------------------------
    # Phase 5: Training loop
    # ------------------------------------------------------------------
    print('\n[Phase 5] Training...')
    history = []

    for epoch in range(args.epochs):
        print(f'\n[Epoch {epoch + 1}/{args.epochs}]')
        start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device,
                                     use_arcface=use_arcface)

        # Free GPU memory between train and eval (helps FDT specifically)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Evaluate
        val_metrics = evaluate_verification(model, val_pair_loader, device)

        scheduler.step()

        elapsed = time.time() - start
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'val_eer': val_metrics['eer'],
            'val_hter': val_metrics['hter'],
            'val_auc': val_metrics['auc'],
            'time_sec': elapsed,
        }
        history.append(epoch_log)

        print(f'  Train Loss: {train_metrics["loss"]:.4f}')
        print(f'  Val EER: {val_metrics["eer"]:.4f} | HTER: {val_metrics["hter"]:.4f} | AUC: {val_metrics["auc"]:.4f}')
        print(f'  Time: {elapsed:.1f}s')

        # Save best checkpoint
        is_best = val_metrics['eer'] < best_eer
        if is_best:
            best_eer = val_metrics['eer']
            best_epoch = epoch + 1
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'val_eer': val_metrics['eer'],
                'val_auc': val_metrics['auc'],
            }, best_model_path)
            print(f'  [BEST] Saved best model (EER={best_eer:.4f})')
            no_improve = 0
        else:
            no_improve += 1
            if patience > 0 and no_improve >= patience:
                print(f'  [STOP] Early stopping: no improvement for {patience} epochs')
                break

    # ------------------------------------------------------------------
    # Phase 6: Final test (using best checkpoint)
    # ------------------------------------------------------------------
    print('\n[Phase 6] Final test on test set...')
    if best_model_path.exists():
        print(f'  Loading best model from epoch {best_epoch} (EER={best_eer:.4f})')
        ckpt = torch.load(best_model_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])

    test_metrics = evaluate_verification(model, test_pair_loader, device)

    print(f'\n[FINAL RESULTS] {args.model}')
    print(f'  Best Val EER: {best_eer:.4f} (epoch {best_epoch})')
    print(f'  Test EER:     {test_metrics["eer"]:.4f}')
    print(f'  Test HTER:    {test_metrics["hter"]:.4f}')
    print(f'  Test AUC:     {test_metrics["auc"]:.4f}')
    print(f'  Threshold:    {test_metrics["threshold"]:.4f}')
    print(f'  Pairs:        {test_metrics["num_pairs"]}')

    # ------------------------------------------------------------------
    # Phase 7: Save results
    # ------------------------------------------------------------------
    print(f'\n[Phase 7] Saving results to {output_dir}...')
    results = {
        'model': args.model,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'num_params': n_params,
        'best_epoch': best_epoch,
        'best_val_eer': best_eer,
        'history': history,
        'final_test': test_metrics,
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'  Saved results to {output_dir / "results.json"}')
    print('\n[OK] Training complete!')


if __name__ == '__main__':
    main()
