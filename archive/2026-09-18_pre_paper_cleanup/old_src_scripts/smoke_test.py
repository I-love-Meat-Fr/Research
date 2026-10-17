#!/usr/bin/env python
"""
Quick smoke test for end-to-end pipeline (3 epoch with ArcFace).
Verifies all 3 models can train and evaluate.
"""
import sys
sys.path.insert(0, '.')

import os
import time
import json
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

from train import (
    get_model, get_classification_model,
    ArcFaceLoss, train_epoch, evaluate_verification,
)
from datasets import (
    FingerVeinDataset, FingerVeinPairDataset,
    compute_eer, compute_hter, compute_roc_auc,
    collate_pairs, get_fvusm_loaders,
)


def smoke_test():
    """Quick smoke test for end-to-end."""
    print('=' * 60)
    print('END-TO-END SMOKE TEST')
    print('=' * 60)

    # 1. Dataset loading
    print('\n[1] Loading dataset...')
    possible_paths = [
        'd:/WorkSpace/Research/Research/MFFV/data/external/Published_database_FV-USM_Dec2013',
        'D:/WorkSpace/Research/Research/MFFV/data/external/Published_database_FV-USM_Dec2013',
    ]
    fvusm_root = None
    for path in possible_paths:
        if os.path.exists(path):
            fvusm_root = path
            break

    if not fvusm_root:
        # Try relative
        candidates = [
            './data/external/Published_database_FV-USM_Dec2013',
            '../data/external/Published_database_FV-USM_Dec2013',
        ]
        for path in candidates:
            if os.path.exists(path):
                fvusm_root = path
                break

    assert fvusm_root, f"FV-USM dataset not found!"
    print(f'  Found: {fvusm_root}')

    train_loader, val_loader, test_loader = get_fvusm_loaders(
        root=fvusm_root, batch_size=8, num_workers=0, session='1st'
    )

    print(f'  Train: {len(train_loader.dataset)} samples')
    print(f'  Test: {len(test_loader.dataset)} samples')

    # Quick pair dataset for eval
    test_dataset = FingerVeinDataset(root=fvusm_root, dataset_name='fvusm',
                                     session='1st', mode='test')
    test_pairs = FingerVeinPairDataset(test_dataset, num_pairs=200)
    test_pair_loader = DataLoader(test_pairs, batch_size=8, shuffle=False,
                                  collate_fn=collate_pairs, num_workers=0)

    num_classes = train_loader.dataset._count_unique_ids()
    print(f'  Num classes: {num_classes}')

    # Auto-detect CUDA device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[Device] Using: {device}')
    if device == 'cuda':
        print(f'[Device] GPU: {torch.cuda.get_device_name(0)}')
        print(f'[Device] VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    models_to_test = ['ilcnn', 'gla-fd', 'fdt']

    all_results = {}
    for model_name in models_to_test:
        print(f'\n[2.{models_to_test.index(model_name)+1}] Testing {model_name}...')
        try:
            # Build model
            model = get_classification_model(model_name, num_classes=num_classes,
                                            embedding_dim=128).to(device)
            n_params = sum(p.numel() for p in model.parameters())
            print(f'  Model built: {n_params:,} params')

            # Build ArcFace
            criterion = ArcFaceLoss(
                embedding_dim=128, num_classes=num_classes,
                margin=0.5, scale=30.0
            ).to(device)
            optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            scheduler = CosineAnnealingLR(optimizer, T_max=2)

            # Train 1 epoch (quick)
            print(f'  Training 1 epoch...')
            train_metrics = train_epoch(model, train_loader, criterion, optimizer,
                                       device, use_arcface=True)
            print(f'  Train Loss: {train_metrics["loss"]:.4f}')

            # Evaluate
            print(f'  Evaluating...')
            test_metrics = evaluate_verification(model, test_pair_loader, device)
            print(f'  Test EER: {test_metrics["eer"]:.4f}')
            print(f'  Test AUC: {test_metrics["auc"]:.4f}')

            all_results[model_name] = {
                'params': n_params,
                'train_loss': train_metrics['loss'],
                'test_eer': test_metrics['eer'],
                'test_auc': test_metrics['auc'],
            }

            print(f'  OK: {model_name} smoke test passed!')

        except Exception as e:
            print(f'  FAIL: {model_name} - {e}')
            import traceback
            traceback.print_exc()
            all_results[model_name] = {'error': str(e)}

    # Print summary
    print(f'\n{"="*60}')
    print('SMOKE TEST SUMMARY')
    print(f'{"="*60}')
    print(f'{"Model":<10} {"Train Loss":<12} {"Test EER":<10} {"Test AUC":<10} {"Params"}')
    print('-' * 60)
    for name, r in all_results.items():
        if 'error' in r:
            print(f'{name:<10} ERROR: {r["error"][:40]}')
        else:
            print(f'{name:<10} {r["train_loss"]:<12.4f} '
                  f'{r["test_eer"]:<10.4f} {r["test_auc"]:<10.4f} '
                  f'{r["params"]:,}')

    print('=' * 60)
    return all_results


if __name__ == '__main__':
    smoke_test()
