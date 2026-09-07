"""
Baseline Finger Vein Recognition with ResNet18 + ArcFace
=======================================================
Simple, fast baseline: ResNet18 backbone + ArcFace loss

Usage:
    python src/baseline_resnet.py --epochs 10 --batch_size 32
    python src/baseline_resnet.py --epochs 10 --arcface --device cuda

Author: Research Team
Date: 2026-08-10
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from tqdm import tqdm

# Add current dir to path
sys.path.insert(0, str(Path(__file__).parent))

from datasets import (
    FingerVeinDataset, FingerVeinPairDataset,
    compute_eer, compute_hter, compute_roc_auc,
    collate_pairs, get_fvusm_loaders,
)


# ============================================================================
# 1. MODEL: ResNet18 for Finger Vein
# ============================================================================

class ResNet18FV(nn.Module):
    """
    ResNet18 adapted for Finger Vein Recognition.
    - Input: 1 channel (grayscale)
    - Output: 128-dim embedding
    - Optional classification head
    """

    def __init__(self, embedding_dim: int = 128, num_classes: int = 492, pretrained: bool = True):
        super().__init__()

        # Load ResNet18 pretrained
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet18(weights=None)

        # Modify first conv for 1-channel input
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        # Modify final layer for embedding
        in_features = self.backbone.fc.in_features  # 512
        self.backbone.fc = nn.Identity()

        # Embedding layer
        self.embedding = nn.Linear(in_features, embedding_dim)

        # Classification head (optional)
        self.classifier = nn.Linear(embedding_dim, num_classes)

        self.in_features = in_features
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> torch.Tensor:
        """
        Args:
            x: Input tensor [B, 1, H, W]
            return_embeddings: If True, return embeddings; else return logits
        Returns:
            Embeddings [B, embedding_dim] or logits [B, num_classes]
        """
        # Extract features
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

        # Embedding
        emb = self.embedding(x)
        emb = F.normalize(emb, p=2, dim=1)

        if return_embeddings:
            return emb

        # Classification
        logits = self.classifier(emb)
        return logits


# ============================================================================
# 2. ArcFace Loss
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

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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

        loss = F.cross_entropy(output, labels)
        return loss, output


# ============================================================================
# 3. Training Functions
# ============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    use_arcface: bool = False,
    log_interval: int = 10,
) -> Dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc='  Train', ncols=100)
    for batch_idx, (imgs, labels) in enumerate(pbar):
        imgs = imgs.to(device)
        labels = labels.to(device).long()

        optimizer.zero_grad()

        if use_arcface:
            # ArcFace: get embeddings, then loss
            embeddings = model(imgs, return_embeddings=True)
            loss, logits = criterion(embeddings, labels)
            with torch.no_grad():
                _, predicted = torch.max(logits, 1)
        else:
            # Standard CE loss
            logits = model(imgs, return_embeddings=False)
            loss = F.cross_entropy(logits, labels)
            _, predicted = torch.max(logits, 1)

        # Backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total += imgs.size(0)
        correct += (predicted == labels).sum().item()

        if batch_idx % log_interval == 0:
            acc = 100 * correct / max(total, 1)
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{acc:.1f}%'})

    return {
        'loss': total_loss / len(loader),
        'accuracy': correct / max(total, 1),
    }


@torch.no_grad()
def evaluate_verification(
    model: nn.Module,
    pair_loader: DataLoader,
    device: str,
) -> Dict:
    """Evaluate on verification task."""
    model.eval()
    all_scores = []
    all_labels = []

    pbar = tqdm(pair_loader, desc='  Eval', ncols=100)
    for img1, img2, labels in pbar:
        img1 = img1.to(device)
        img2 = img2.to(device)

        # Get embeddings
        emb1 = model(img1, return_embeddings=True)
        emb2 = model(img2, return_embeddings=True)

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
        'eer': float(eer),
        'hter': float(hter),
        'auc': float(auc),
        'threshold': float(threshold),
        'num_pairs': len(all_labels),
    }


@torch.no_grad()
def evaluate_classification(
    model: nn.Module,
    loader: DataLoader,
    device: str,
) -> Dict:
    """Evaluate on classification task."""
    model.eval()
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device).long()

        logits = model(imgs, return_embeddings=False)
        _, predicted = torch.max(logits, 1)

        total += imgs.size(0)
        correct += (predicted == labels).sum().item()

    return {
        'accuracy': correct / max(total, 1),
    }


# ============================================================================
# 4. Main Pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='ResNet18 Finger Vein Baseline')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=112, help='Image size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dimension')
    parser.add_argument('--arcface', action='store_true', help='Use ArcFace loss')
    parser.add_argument('--arcface_margin', type=float, default=0.5, help='ArcFace margin')
    parser.add_argument('--arcface_scale', type=float, default=30.0, help='ArcFace scale')
    parser.add_argument('--device', type=str, default='auto', help='Device: auto/cuda/cpu')
    parser.add_argument('--output_dir', type=str, default='results/baseline_resnet18',
                        help='Output directory')
    parser.add_argument('--session', type=str, default='1st', choices=['1st', '2nd', 'all'],
                        help='Dataset session to use')
    parser.add_argument('--num_pairs', type=int, default=5000, help='Number of pairs for eval')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--strong_augment', action='store_true', help='Use strong augmentation (blur + random erasing)')
    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print('=' * 70)
    print('RESNET18 + ArcFace FINGER VEIN BASELINE')
    print('=' * 70)
    print(f'Device: {device}')
    print(f'Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}')
    print(f'ArcFace: {args.arcface} (margin={args.arcface_margin}, scale={args.arcface_scale})')
    print(f'Session: {args.session}, Image size: {args.img_size}')
    print(f'Strong augment: {args.strong_augment}')
    print('=' * 70)

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load datasets
    # ------------------------------------------------------------------
    print('\n[1] Loading datasets...')
    train_loader, val_loader, test_loader = get_fvusm_loaders(
        batch_size=args.batch_size,
        num_workers=0,  # Set >0 for faster loading if you have multiple cores
        session=args.session,
        img_size=args.img_size,
        strong_augment=args.strong_augment,
    )

    num_classes = train_loader.dataset._count_unique_ids()
    print(f'  Classes: {num_classes}')
    print(f'  Train samples: {len(train_loader.dataset)}')
    print(f'  Val samples: {len(val_loader.dataset)}')
    print(f'  Test samples: {len(test_loader.dataset)}')

    # Build pair datasets
    print('\n[2] Building pair datasets...')
    _eval_root = str(Path(train_loader.dataset.root))
    val_dataset = FingerVeinDataset(
        root=_eval_root,
        dataset_name='fvusm', session=args.session, mode='val',
        transform=transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]),
    )
    test_dataset = FingerVeinDataset(
        root=_eval_root,
        dataset_name='fvusm', session=args.session, mode='test',
        transform=transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]),
    )

    val_pairs = FingerVeinPairDataset(val_dataset, num_pairs=args.num_pairs, seed=args.seed)
    test_pairs = FingerVeinPairDataset(test_dataset, num_pairs=args.num_pairs, seed=args.seed)

    val_pair_loader = DataLoader(val_pairs, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_pairs, num_workers=0)
    test_pair_loader = DataLoader(test_pairs, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=collate_pairs, num_workers=0)

    print(f'  Val pairs: {len(val_pairs)}')
    print(f'  Test pairs: {len(test_pairs)}')

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    print('\n[3] Building model...')
    model = ResNet18FV(
        embedding_dim=args.embedding_dim,
        num_classes=num_classes,
        pretrained=True,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Parameters: {n_params:,}')

    # ------------------------------------------------------------------
    # Setup training
    # ------------------------------------------------------------------
    print('\n[4] Setting up training...')
    if args.arcface:
        criterion = ArcFaceLoss(
            embedding_dim=args.embedding_dim,
            num_classes=num_classes,
            margin=args.arcface_margin,
            scale=args.arcface_scale,
        ).to(device)
        print(f'  Loss: ArcFaceLoss (margin={args.arcface_margin}, scale={args.arcface_scale})')
    else:
        criterion = None
        print(f'  Loss: CrossEntropyLoss')

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------
    print('\n[5] Training...')
    history = []
    best_eer = float('inf')
    best_model_path = output_dir / 'best_model.pth'

    for epoch in range(args.epochs):
        print(f'\n--- Epoch {epoch + 1}/{args.epochs} ---')
        start = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device,
            use_arcface=args.arcface,
        )

        # Evaluate
        val_metrics = evaluate_verification(model, val_pair_loader, device)
        val_cls_metrics = evaluate_classification(model, val_loader, device)

        scheduler.step()

        elapsed = time.time() - start

        # Log
        epoch_log = {
            'epoch': epoch + 1,
            'train_loss': train_metrics['loss'],
            'train_acc': train_metrics['accuracy'],
            'val_eer': val_metrics['eer'],
            'val_hter': val_metrics['hter'],
            'val_auc': val_metrics['auc'],
            'val_cls_acc': val_cls_metrics['accuracy'],
            'time_sec': elapsed,
        }
        history.append(epoch_log)

        print(f'  Train Loss: {train_metrics["loss"]:.4f} | Acc: {train_metrics["accuracy"]*100:.1f}%')
        print(f'  Val EER: {val_metrics["eer"]*100:.2f}% | HTER: {val_metrics["hter"]*100:.2f}% | AUC: {val_metrics["auc"]:.4f}')
        print(f'  Val Cls Acc: {val_cls_metrics["accuracy"]*100:.1f}%')
        print(f'  Time: {elapsed:.1f}s')

        # Save best model
        if val_metrics['eer'] < best_eer:
            best_eer = val_metrics['eer']
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'eer': best_eer,
                # Metadata for inference
                'embedding_dim': args.embedding_dim,
                'num_classes': num_classes,
                'img_size': args.img_size,
                'threshold': val_metrics['threshold'],
                'use_arcface': args.arcface,
                'arcface_margin': args.arcface_margin,
                'arcface_scale': args.arcface_scale,
            }, best_model_path)
            print(f'  [BEST] Saved best model (EER: {best_eer*100:.2f}%)')

    # ------------------------------------------------------------------
    # Final test
    # ------------------------------------------------------------------
    print('\n[6] Final test on test set...')

    # Load best model
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        print(f'  Loaded best model (from epoch {checkpoint["epoch"]+1})')

    test_metrics = evaluate_verification(model, test_pair_loader, device)
    test_cls_metrics = evaluate_classification(model, test_loader, device)

    print('\n' + '=' * 70)
    print('FINAL RESULTS')
    print('=' * 70)
    print(f'Model: ResNet18 + ArcFace={args.arcface}')
    print(f'Dataset: FV-USM Session {args.session}')
    print(f'Test EER:  {test_metrics["eer"]*100:.2f}%')
    print(f'Test HTER: {test_metrics["hter"]*100:.2f}%')
    print(f'Test AUC:  {test_metrics["auc"]:.4f}')
    print(f'Test Cls Acc: {test_cls_metrics["accuracy"]*100:.1f}%')
    print(f'Threshold: {test_metrics["threshold"]:.4f}')
    print(f'Pairs: {test_metrics["num_pairs"]}')
    print('=' * 70)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    print('\n[7] Saving results...')
    results = {
        'model': 'ResNet18',
        'arcface': args.arcface,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'embedding_dim': args.embedding_dim,
        'session': args.session,
        'num_params': n_params,
        'history': history,
        'final_test': test_metrics,
        'final_test_cls_acc': test_cls_metrics['accuracy'],
    }

    with open(output_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f'  Saved results to {output_dir / "results.json"}')
    print('\n✅ Training complete!')


if __name__ == '__main__':
    main()
