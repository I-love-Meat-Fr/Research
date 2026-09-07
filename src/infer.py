"""
Finger Vein Verification — Inference Script
============================================
End-to-end inference: Load checkpoint -> Extract embeddings -> Verify 1 pair

Usage:
    # Verify a single pair
    python src/infer.py --path1 "path/to/img1.png" --path2 "path/to/img2.png"

    # With custom checkpoint
    python src/infer.py --path1 "..." --path2 "..." --checkpoint results/baseline_resnet18_arcface_expD/best_model.pth

    # With custom threshold
    python src/infer.py --path1 "..." --path2 "..." --threshold 0.5

    # Interactive: enter paths interactively
    python src/infer.py --interactive

    # Batch: verify a list of pairs from CSV
    python src/infer.py --csv pairs.csv --col1 col_name1 --col2 col_name2
"""

import os
import sys
import csv
import json
import argparse
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image


# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from datasets import FingerVeinDataset


# ============================================================================
# MODEL: Must match ResNet18FV from baseline_resnet.py
# ============================================================================

class ResNet18FV(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 492, pretrained: bool = True):
        super().__init__()
        if pretrained:
            weights = ResNet18_Weights.DEFAULT
            self.backbone = resnet18(weights=weights)
        else:
            self.backbone = resnet18(weights=None)

        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
# HELPERS
# ============================================================================

def load_checkpoint(checkpoint_path: str, device: str = 'auto') -> Tuple[dict, dict]:
    """
    Load checkpoint. Tries .pth first, falls back to results.json.

    Returns (checkpoint_metadata, model_state_dict)
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    p = Path(checkpoint_path)
    if not p.exists():
        raise FileNotFoundError(f'Checkpoint not found: {checkpoint_path}')

    # Try .pth
    if p.suffix == '.pth' or p.suffix == '.pt':
        ckpt = torch.load(p, map_location=device, weights_only=False)
        meta = {
            'embedding_dim': ckpt.get('embedding_dim', 128),
            'num_classes': ckpt.get('num_classes', 492),
            'img_size': ckpt.get('img_size', 224),
            'threshold': ckpt.get('threshold', None),
            'use_arcface': ckpt.get('use_arcface', True),
            'epoch': ckpt.get('epoch', -1),
            'eer': ckpt.get('eer', None),
        }
        return meta, ckpt.get('model_state', ckpt)

    # Try results.json
    if p.name == 'results.json':
        with open(p) as f:
            data = json.load(f)
        # Look for best_model.pth alongside
        sibling = p.parent / 'best_model.pth'
        if sibling.exists():
            return load_checkpoint(str(sibling), device)
        raise FileNotFoundError(f'No best_model.pth found alongside {p}')

    raise ValueError(f'Unknown checkpoint format: {checkpoint_path}')


def build_model(meta: dict, device: str) -> nn.Module:
    """Build model and load weights from metadata."""
    model = ResNet18FV(
        embedding_dim=meta['embedding_dim'],
        num_classes=meta['num_classes'],
        pretrained=False,  # Weights loaded manually
    ).to(device)
    return model


def preprocess_image(
    img_path: str,
    img_size: int,
    device: str,
) -> torch.Tensor:
    """
    Load and preprocess a single finger vein image.

    Args:
        img_path: Path to image file (PNG, JPG, BMP, etc.)
        img_size: Target size (HxW)
        device: torch device

    Returns:
        Tensor [1, 1, img_size, img_size]
    """
    if not Path(img_path).exists():
        raise FileNotFoundError(f'Image not found: {img_path}')

    # Load image
    img = Image.open(img_path)

    # Convert to grayscale if needed
    if img.mode != 'L':
        img = img.convert('L')

    # Define transform
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    tensor = transform(img)  # [1, H, W]
    tensor = tensor.unsqueeze(0)  # [1, 1, H, W]
    return tensor.to(device)


@torch.no_grad()
def extract_embedding(
    model: nn.Module,
    tensor: torch.Tensor,
) -> np.ndarray:
    """Extract L2-normalized embedding from image tensor."""
    model.eval()
    emb = model(tensor, return_embeddings=True)  # [1, embedding_dim]
    return emb.cpu().numpy().squeeze()


@torch.no_grad()
def compute_similarity(
    emb1: np.ndarray,
    emb2: np.ndarray,
) -> float:
    """Compute cosine similarity between two embeddings."""
    return float(np.dot(emb1, emb2))


def verify_pair(
    model: nn.Module,
    path1: str,
    path2: str,
    threshold: float,
    img_size: int,
    device: str,
) -> dict:
    """
    Verify a single pair of finger vein images.

    Returns dict with:
        - score: cosine similarity
        - threshold: decision threshold
        - verdict: 'GENUINE', 'IMPOSTOR', or 'TIE'
        - label: human-readable message
    """
    t1 = preprocess_image(path1, img_size, device)
    t2 = preprocess_image(path2, img_size, device)

    emb1 = extract_embedding(model, t1)
    emb2 = extract_embedding(model, t2)

    score = compute_similarity(emb1, emb2)

    if score > threshold:
        verdict = 'GENUINE'
        label = f'  => MATCH (score {score:.4f} > threshold {threshold:.4f})'
    elif score < threshold:
        verdict = 'IMPOSTOR'
        label = f'  => NO MATCH (score {score:.4f} < threshold {threshold:.4f})'
    else:
        verdict = 'TIE'
        label = f'  => TIE (score {score:.4f} == threshold {threshold:.4f})'

    return {
        'score': score,
        'threshold': threshold,
        'verdict': verdict,
        'label': label,
        'path1': path1,
        'path2': path2,
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Finger Vein Verification — End-to-End Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single pair:
    python src/infer.py --path1 img/a.png --path2 img/b.png

  With custom checkpoint:
    python src/infer.py --path1 img/a.png --path2 img/b.png \\
        --checkpoint results/baseline_resnet18_arcface_expD/best_model.pth

  With custom threshold:
    python src/infer.py --path1 img/a.png --path2 img/b.png --threshold 0.4557

  Batch from CSV:
    python src/infer.py --csv pairs.csv --col1 img1 --col2 img2 \\
        --checkpoint results/baseline_resnet18_arcface_expD/best_model.pth

  Interactive mode:
    python src/infer.py --interactive
        """
    )
    parser.add_argument('--path1', type=str, help='Path to first image')
    parser.add_argument('--path2', type=str, help='Path to second image')
    parser.add_argument('--checkpoint', type=str,
                        default='src/results/baseline_resnet18_arcface_expD/best_model.pth',
                        help='Path to model checkpoint (.pth or results.json)')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Decision threshold (overrides checkpoint threshold)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: auto/cuda/cpu')
    parser.add_argument('--interactive', action='store_true',
                        help='Interactive mode: enter paths manually')
    parser.add_argument('--csv', type=str,
                        help='CSV file with pairs to verify')
    parser.add_argument('--col1', type=str, default='img1',
                        help='Column name for first image path in CSV')
    parser.add_argument('--col2', type=str, default='img2',
                        help='Column name for second image path in CSV')
    parser.add_argument('--verbose', action='store_true',
                        help='Print embedding norms and extra debug info')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f'Device: {device}')

    # Load checkpoint
    print(f'\nLoading checkpoint: {args.checkpoint}')
    meta, state_dict = load_checkpoint(args.checkpoint, device)
    print(f'  Model: ResNet18FV (embedding_dim={meta["embedding_dim"]}, num_classes={meta["num_classes"]})')
    print(f'  Image size: {meta["img_size"]}')
    print(f'  Trained epoch: {meta["epoch"] + 1}')
    print(f'  Trained EER: {meta["eer"]*100:.2f}%' if meta["eer"] else '  Trained EER: N/A')
    threshold = args.threshold if args.threshold is not None else meta['threshold']
    if threshold is None:
        print('  WARNING: No threshold in checkpoint and none provided. Using 0.5')
        threshold = 0.5
    print(f'  Decision threshold: {threshold:.4f}')

    # Build model
    model = build_model(meta, device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'  Model loaded successfully.')

    # ----------------------------------------------------------------
    # Single pair
    # ----------------------------------------------------------------
    if args.path1 and args.path2:
        print(f'\n--- Verifying Single Pair ---')
        print(f'  Image 1: {args.path1}')
        print(f'  Image 2: {args.path2}')

        result = verify_pair(model, args.path1, args.path2, threshold, meta['img_size'], device)
        print(f'\n  Cosine Similarity: {result["score"]:.6f}')
        print(f'  Threshold: {result["threshold"]:.6f}')
        print(f'  Decision: {result["verdict"]}')
        if args.verbose:
            print(f'\n  Path1: {result["path1"]}')
            print(f'  Path2: {result["path2"]}')
        print()
        return

    # ----------------------------------------------------------------
    # Batch from CSV
    # ----------------------------------------------------------------
    if args.csv:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            print(f'ERROR: CSV file not found: {args.csv}')
            return

        print(f'\n--- Batch Verification from CSV ---')
        print(f'  CSV: {csv_path}')
        print(f'  Columns: {args.col1}, {args.col2}')
        print(f'  Threshold: {threshold:.4f}')
        print()

        results = []
        genuine_count = 0
        impostor_count = 0

        with open(csv_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f'  Total pairs in CSV: {len(rows)}')
        print()

        for i, row in enumerate(rows):
            p1 = row.get(args.col1) or row.get('img1') or row.get('path1')
            p2 = row.get(args.col2) or row.get('img2') or row.get('path2')

            if not p1 or not p2:
                print(f'  [{i+1}] SKIP (missing path): {row}')
                continue

            try:
                result = verify_pair(model, str(p1).strip(), str(p2).strip(),
                                      threshold, meta['img_size'], device)
            except FileNotFoundError as e:
                print(f'  [{i+1}] FILE NOT FOUND: {e}')
                continue
            except Exception as e:
                print(f'  [{i+1}] ERROR: {e}')
                continue

            verdict_label = '[GENUINE] ' if result['verdict'] == 'GENUINE' else '[IMPOSTOR]'
            print(f'  [{i+1}] {verdict_label} score={result["score"]:.4f}  |  {Path(p1).name} vs {Path(p2).name}')

            if result['verdict'] == 'GENUINE':
                genuine_count += 1
            else:
                impostor_count += 1

            results.append(result)

        print()
        print(f'  --- Summary ---')
        print(f'  Total processed: {len(results)}')
        print(f'  GENUINE: {genuine_count}')
        print(f'  IMPOSTOR: {impostor_count}')

        # Save results
        out_csv = csv_path.parent / f'verification_results_{csv_path.stem}.csv'
        with open(out_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['img1', 'img2', 'score', 'threshold', 'verdict'])
            writer.writeheader()
            for r in results:
                writer.writerow({
                    'img1': r['path1'],
                    'img2': r['path2'],
                    'score': f'{r["score"]:.6f}',
                    'threshold': f'{r["threshold"]:.6f}',
                    'verdict': r['verdict'],
                })
        print(f'  Results saved to: {out_csv}')
        return

    # ----------------------------------------------------------------
    # Interactive
    # ----------------------------------------------------------------
    if args.interactive:
        print(f'\n--- Interactive Verification ---')
        print(f'  Image size: {meta["img_size"]}, Threshold: {threshold:.4f}')
        print(f'  Type "q" to quit.\n')

        while True:
            p1 = input('Path to image 1: ').strip()
            if p1.lower() in ('q', 'quit', 'exit'):
                print('Goodbye.')
                break
            p2 = input('Path to image 2: ').strip()
            if p2.lower() in ('q', 'quit', 'exit'):
                print('Goodbye.')
                break

            try:
                result = verify_pair(model, p1, p2, threshold, meta['img_size'], device)
            except FileNotFoundError as e:
                print(f'  ERROR: {e}\n')
                continue
            except Exception as e:
                print(f'  ERROR: {e}\n')
                continue

            print(f'  Cosine Similarity: {result["score"]:.6f}')
            print(f'  Decision: {result["verdict"]}\n')
        return

    # ----------------------------------------------------------------
    # No mode selected
    # ----------------------------------------------------------------
    print()
    print('ERROR: No mode selected.')
    print('  Use --path1 and --path2 for single pair,')
    print('  use --csv for batch, or')
    print('  use --interactive for interactive mode.')
    print()
    parser.print_help()


if __name__ == '__main__':
    main()
