#!/usr/bin/env python
"""
Comparison Runner: Train all 3 models with ArcFace and compare results.

Usage:
    cd src
    python compare_models.py --epochs 5 --quick
"""
import sys
sys.path.insert(0, '.')

import os
import time
import json
import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Import from train.py
from train import (
    get_model, get_classification_model,
    ArcFaceLoss, train_epoch, evaluate_verification,
)
from datasets import (
    FingerVeinDataset, FingerVeinPairDataset,
    compute_eer, compute_hter, compute_roc_auc,
    collate_pairs, get_fvusm_loaders,
)


def train_single_model(model_name: str, args, device: str) -> Dict:
    """Train a single model and return results."""
    print(f"\n{'='*70}")
    print(f"TRAINING: {model_name.upper()}")
    print(f"{'='*70}")

    # Auto-detect FV-USM path
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
        print(f"[ERROR] FV-USM dataset not found!")
        return None

    # Load datasets
    train_loader, val_loader, test_loader = get_fvusm_loaders(
        root=fvusm_root,
        batch_size=args.batch_size, num_workers=0, session='1st',
        img_size=args.img_size,
    )

    val_dataset = FingerVeinDataset(root=fvusm_root, dataset_name='fvusm', session='1st', mode='val')
    test_dataset = FingerVeinDataset(root=fvusm_root, dataset_name='fvusm', session='1st', mode='test')

    n_pairs = 500 if args.quick else 2000
    val_pairs = FingerVeinPairDataset(val_dataset, num_pairs=n_pairs)
    test_pairs = FingerVeinPairDataset(test_dataset, num_pairs=n_pairs)

    val_pair_loader = DataLoader(val_pairs, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_pairs, num_workers=0)
    test_pair_loader = DataLoader(test_pairs, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=collate_pairs, num_workers=0)

    # Build model
    num_classes = train_loader.dataset._count_unique_ids()
    print(f'  Classes: {num_classes}')

    try:
        model = get_classification_model(model_name, num_classes=num_classes,
                                         embedding_dim=args.embedding_dim).to(device)
    except Exception as e:
        print(f'  [ERROR] Failed to build {model_name}: {e}')
        return None

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Params: {n_params:,}')

    # Use ArcFace loss by default for comparison
    use_arcface = True
    if use_arcface:
        criterion = ArcFaceLoss(
            embedding_dim=args.embedding_dim,
            num_classes=num_classes,
            margin=args.arcface_margin,
            scale=args.arcface_scale,
        ).to(device)
    else:
        criterion = nn.CrossEntropyLoss()

    # FDT is 30x slower per epoch — cap at 3 to stay within reasonable time
    if model_name == 'fdt':
        max_epochs = min(args.epochs, 3)
        print(f'  [INFO] FDT capped at {max_epochs} epochs (full: {args.epochs})')
    else:
        max_epochs = args.epochs

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs)

    # Training loop
    history = []
    for epoch in range(max_epochs):
        print(f'\n[Epoch {epoch + 1}/{max_epochs}]')
        start = time.time()

        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device,
                                   use_arcface=use_arcface)
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

        print(f'  Loss: {train_metrics["loss"]:.4f} | '
              f'Val EER: {val_metrics["eer"]:.4f} | '
              f'AUC: {val_metrics["auc"]:.4f} | '
              f'Time: {elapsed:.1f}s')

    # Final test
    test_metrics = evaluate_verification(model, test_pair_loader, device)
    print(f'\n[FINAL TEST] {model_name}')
    print(f'  EER: {test_metrics["eer"]:.4f} | '
          f'HTER: {test_metrics["hter"]:.4f} | '
          f'AUC: {test_metrics["auc"]:.4f}')

    return {
        'model': model_name,
        'num_params': n_params,
        'epochs': max_epochs,
        'epochs_requested': args.epochs,
        'history': history,
        'final_test': test_metrics,
    }


def main():
    parser = argparse.ArgumentParser(description='Compare 3 Models')
    parser.add_argument('--models', nargs='+', default=['ilcnn', 'gla-fd', 'fdt'],
                        help='Models to compare')
    parser.add_argument('--epochs', type=int, default=10, help='Base epochs (FDT uses 3)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--img_size', type=int, default=112, help='Image size')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Embedding dim')
    parser.add_argument('--arcface_margin', type=float, default=0.5, help='ArcFace margin')
    parser.add_argument('--arcface_scale', type=float, default=30.0, help='ArcFace scale')
    parser.add_argument('--quick', action='store_true', help='Quick test (fewer pairs)')
    parser.add_argument('--output_dir', type=str, default='results/comparison')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[Setup] Device: {device} | Models: {args.models} | Epochs: {args.epochs}')

    # Create output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train all models
    all_results = {}
    for model_name in args.models:
        try:
            result = train_single_model(model_name, args, device)
            if result:
                all_results[model_name] = result
        except Exception as e:
            print(f'\n[ERROR] {model_name} failed: {e}')
            import traceback
            traceback.print_exc()

    # Save results
    output_file = output_dir / f'comparison_epochs{args.epochs}.json'
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f'\n[SAVED] Results to {output_file}')

    # Print comparison table
    print(f'\n{"="*70}')
    print('COMPARISON RESULTS')
    print(f'{"="*70}')
    print(f'{"Model":<10} {"Params":<12} {"Test EER":<10} {"Test HTER":<10} {"Test AUC":<10}')
    print('-' * 70)
    for model_name, result in all_results.items():
        m = result['final_test']
        print(f'{model_name:<10} '
              f'{result["num_params"]:<12,} '
              f'{m["eer"]:<10.4f} '
              f'{m["hter"]:<10.4f} '
              f'{m["auc"]:<10.4f}')
    print('=' * 70)


if __name__ == '__main__':
    main()
