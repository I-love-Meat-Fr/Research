"""
Unified Finger Vein Dataset for FV-USM + FingerVeinSyn-5M
========================================================

Supports:
- FV-USM (real): 11,808 images, 492 fingers, 2 sessions
- FingerVeinSyn-5M (synthetic): 5M images, 50k identities
- Custom pair generation for verification (genuine/impostor)

Author: Research Team
Date: 2026-08-09
"""
import os
import glob
import random
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from collections import defaultdict

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ============================================================================
# 1. UNIFIED DATASET CLASS
# ============================================================================

class FingerVeinDataset(Dataset):
    """
    Unified dataset for finger vein recognition.

    Supports:
    - FV-USM: real images, structured by session/finger
    - FingerVeinSyn-5M: synthetic images, named like {id}_{seq}_{variant}.png

    Args:
        root: Path to dataset root
        dataset_name: 'fvusm' or 'fingerveinsyn5m'
        session: '1st', '2nd', or 'all' (for FV-USM only)
        transform: torchvision transforms
        mode: 'train', 'val', 'test'
        split_ratio: train/val/test split (e.g., 0.7, 0.15, 0.15)
    """

    def __init__(
        self,
        root: str,
        dataset_name: str = 'fvusm',
        session: str = '1st',
        transform=None,
        mode: str = 'train',
        split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ):
        self.root = Path(root)
        self.dataset_name = dataset_name.lower()
        self.session = session
        self.transform = transform
        self.mode = mode
        self.split_ratio = split_ratio
        self.seed = seed

        # Load samples
        if self.dataset_name == 'fvusm':
            self.samples = self._load_fvusm()
        elif self.dataset_name == 'fingerveinsyn5m':
            self.samples = self._load_fingerveinsyn5m()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")

        # Split into train/val/test
        self.samples = self._split_samples(self.samples)

        # Build class_id -> int label mapping
        self.classes = sorted(set(s['class_id'] for s in self.samples))
        self.class_to_label = {cls: idx for idx, cls in enumerate(self.classes)}

        print(f"[{dataset_name.upper()}] {mode}: {len(self.samples)} samples | "
              f"{self._count_unique_ids()} unique IDs")

    def _load_fvusm(self) -> List[Dict]:
        """
        Load FV-USM structure.
        Path: {session}/raw_data/{user_id}_{finger_id}/0{idx}.jpg
        Example: 1st_session/raw_data/037_4/01.jpg
        """
        samples = []
        sessions = ['1st_session', '2nd_session'] if self.session == 'all' else [f'{self.session}_session']

        for session in sessions:
            session_dir = self.root / session / 'raw_data'
            if not session_dir.exists():
                continue

            # Iterate over all users
            for finger_dir in sorted(session_dir.iterdir()):
                if not finger_dir.is_dir():
                    continue

                # Parse: e.g., "037_4" -> user_id=037, finger_idx=4
                parts = finger_dir.name.split('_')
                if len(parts) != 2:
                    continue

                user_id = parts[0]
                finger_idx = int(parts[1])

                # Unique class ID: user_id * 10 + finger_idx
                class_id = f"{user_id}_{finger_idx}"

                # Load all 6 images
                for img_path in sorted(finger_dir.glob('*.jpg')):
                    samples.append({
                        'path': str(img_path),
                        'class_id': class_id,
                        'user_id': user_id,
                        'finger_idx': finger_idx,
                        'session': session,
                        'image_id': img_path.stem,
                    })

        return samples

    def _load_fingerveinsyn5m(self) -> List[Dict]:
        """
        Load FingerVeinSyn-5M.
        Path: {root}/{identity_id}_{seq}_{variant}.png
        Example: 04551_001_Shift.png
        """
        samples = []
        for img_path in sorted(self.root.glob('*.png')):
            # Parse filename
            parts = img_path.stem.split('_')
            if len(parts) < 2:
                continue

            identity_id = parts[0]
            variant = parts[-1] if len(parts) >= 3 else 'unknown'

            samples.append({
                'path': str(img_path),
                'class_id': identity_id,
                'identity_id': identity_id,
                'variant': variant,
                'image_id': img_path.stem,
            })

        return samples

    def _split_samples(self, samples: List[Dict]) -> List[Dict]:
        """Split samples by class_id into train/val/test."""
        # Group by class_id
        class_groups = defaultdict(list)
        for s in samples:
            class_groups[s['class_id']].append(s)

        # Sort classes for deterministic split
        rng = random.Random(self.seed)
        classes = sorted(class_groups.keys())
        rng.shuffle(classes)

        # Split classes
        n = len(classes)
        if n < 3:
            # Not enough classes for proper split, use all for training
            return samples

        n_train = max(1, int(n * self.split_ratio[0]))
        n_val = max(1, int(n * self.split_ratio[1]))

        # Ensure n_train + n_val + n_test >= n
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)

        if self.mode == 'train':
            selected = classes[:n_train]
        elif self.mode == 'val':
            selected = classes[n_train:n_train + n_val]
        else:  # test
            selected = classes[n_train + n_val:]

        # Collect samples
        result = []
        for cls in selected:
            result.extend(class_groups[cls])

        return result

    def _count_unique_ids(self) -> int:
        return len(set(s['class_id'] for s in self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        sample = self.samples[idx]

        # Load image
        img = cv2.imread(sample['path'], cv2.IMREAD_GRAYSCALE)
        if img is None:
            # Fallback: try PIL
            img = np.array(Image.open(sample['path']).convert('L'))

        # Apply transform
        if self.transform is not None:
            # Convert numpy to PIL for transforms
            img_pil = Image.fromarray(img)
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = torch.from_numpy(img).float().unsqueeze(0) / 255.0

        # Map class_id to integer label
        label = self.class_to_label[sample['class_id']]

        return img_tensor, label


# ============================================================================
# 2. PAIR GENERATION FOR VERIFICATION
# ============================================================================

class FingerVeinPairDataset(Dataset):
    """
    Generate pairs for verification task.
    - Genuine pairs: same class, different images
    - Impostor pairs: different classes, random images

    Args:
        dataset: Base FingerVeinDataset
        num_pairs: Total number of pairs to generate
        genuine_ratio: Ratio of genuine pairs (e.g., 0.5)
        seed: Random seed
    """

    def __init__(
        self,
        dataset: FingerVeinDataset,
        num_pairs: int = 10000,
        genuine_ratio: float = 0.5,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.num_pairs = num_pairs
        self.genuine_ratio = genuine_ratio
        self.seed = seed

        # Group samples by class_id
        self.class_groups = defaultdict(list)
        for idx, sample in enumerate(dataset.samples):
            self.class_groups[sample['class_id']].append(idx)

        self.classes = list(self.class_groups.keys())

        # Generate pairs
        self.pairs = self._generate_pairs()

    def _generate_pairs(self) -> List[Tuple[int, int, int]]:
        """Generate (idx1, idx2, label) tuples."""
        rng = random.Random(self.seed)
        n_genuine = int(self.num_pairs * self.genuine_ratio)
        n_impostor = self.num_pairs - n_genuine

        pairs = []

        # Genuine pairs
        for _ in range(n_genuine):
            cls = rng.choice(self.classes)
            if len(self.class_groups[cls]) < 2:
                continue
            idx1, idx2 = rng.sample(self.class_groups[cls], 2)
            pairs.append((idx1, idx2, 1))

        # Impostor pairs
        for _ in range(n_impostor):
            cls1, cls2 = rng.sample(self.classes, 2)
            idx1 = rng.choice(self.class_groups[cls1])
            idx2 = rng.choice(self.class_groups[cls2])
            pairs.append((idx1, idx2, 0))

        rng.shuffle(pairs)
        return pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        idx1, idx2, label = self.pairs[idx]
        img1, _ = self.dataset[idx1]
        img2, _ = self.dataset[idx2]
        # Ensure both have same shape
        if img1.shape != img2.shape:
            img2 = torch.nn.functional.interpolate(
                img2.unsqueeze(0), size=img1.shape[-2:], mode='bilinear', align_corners=False
            ).squeeze(0)
        return img1, img2, label


# ============================================================================
# 3. EVALUATION METRICS
# ============================================================================

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER) and threshold.

    Args:
        y_true: Ground truth labels (1=genuine, 0=impostor)
        y_scores: Similarity scores (higher = more similar)

    Returns:
        (eer, threshold)
    """
    # Sort by score
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]

    # Compute FPR and FNR at each threshold
    n_genuine = np.sum(y_true == 1)
    n_impostor = np.sum(y_true == 0)

    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)

    fnr = (n_genuine - tp) / n_genuine
    fpr = fp / n_impostor

    # Find EER
    diff = np.abs(fnr - fpr)
    eer_idx = np.argmin(diff)
    eer = (fnr[eer_idx] + fpr[eer_idx]) / 2
    threshold = y_scores[sorted_indices[eer_idx]]

    return float(eer), float(threshold)


def compute_hter(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> float:
    """Compute Half Total Error Rate at given threshold."""
    y_pred = (y_scores >= threshold).astype(int)
    far = np.sum((y_pred == 1) & (y_true == 0)) / np.sum(y_true == 0)
    frr = np.sum((y_pred == 0) & (y_true == 1)) / np.sum(y_true == 1)
    return (far + frr) / 2


def compute_roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute ROC AUC."""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y_true, y_scores)


# ============================================================================
# 4. CUSTOM COLLATE FUNCTION
# ============================================================================

def collate_pairs(batch):
    """Collate function for pair datasets."""
    img1 = torch.stack([b[0] for b in batch])
    img2 = torch.stack([b[1] for b in batch])
    labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return img1, img2, labels


# ============================================================================
# 5. TRANSFORMS
# ============================================================================

def get_transforms(img_size: int = 112, augment: bool = True, strong: bool = False):
    """Get train/val transforms."""
    if augment:
        aug_list = [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
        ]
        if strong:
            aug_list += [
                transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5))], p=0.3),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
            ]
        aug_list += [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ]
        train_transform = transforms.Compose(aug_list)
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

    return train_transform, val_transform


# ============================================================================
# 6. CONVENIENCE FUNCTIONS
# ============================================================================

def get_fvusm_loaders(
    root: str = None,  # Default to project-relative path
    img_size: int = 112,
    batch_size: int = 16,
    num_workers: int = 0,
    session: str = '1st',
    strong_augment: bool = False,
):
    """Get train/val/test DataLoaders for FV-USM."""
    # Auto-detect path: use provided path or try project-relative paths
    if root is None:
        import os
        # Try multiple possible locations
        possible_paths = [
            'd:/WorkSpace/Research/Research/MFFV/data/external/Published_database_FV-USM_Dec2013',
            'D:/WorkSpace/Research/Research/MFFV/data/external/Published_database_FV-USM_Dec2013',
            'd:/WorkSpace/Research/Research/MFFV/data/external',
            'D:/WorkSpace/Research/Research/MFFV/data/external',
            './data/external/Published_database_FV-USM_Dec2013',
            '../data/external/Published_database_FV-USM_Dec2013',
            'data/external/Published_database_FV-USM_Dec2013',
        ]
        found = False
        for path in possible_paths:
            # Check for exact path or parent path containing FV-USM folder
            test_path = os.path.join(path, 'Published_database_FV-USM_Dec2013')
            if os.path.exists(test_path):
                root = test_path
                found = True
                break
            elif os.path.exists(path):
                # Check if this path itself is the FV-USM folder
                for item in os.listdir(path):
                    if 'FV-USM' in item or 'fv_usm' in item.lower():
                        root = os.path.join(path, item)
                        found = True
                        break
                if found:
                    break
        if not found or root is None:
            # Fallback: use the first path as default (will fail with clear error)
            root = possible_paths[0]
            print(f"[WARNING] FV-USM dataset not found at expected locations.")
            print(f"  Expected path: {possible_paths}")
            print(f"  Please download FV-USM dataset and set --data_root or update datasets.py")
    
    train_transform, val_transform = get_transforms(img_size, augment=True)

    train_dataset = FingerVeinDataset(
        root=root, dataset_name='fvusm', session=session,
        transform=train_transform, mode='train',
    )
    val_dataset = FingerVeinDataset(
        root=root, dataset_name='fvusm', session=session,
        transform=val_transform, mode='val',
    )
    test_dataset = FingerVeinDataset(
        root=root, dataset_name='fvusm', session=session,
        transform=val_transform, mode='test',
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=False)

    return train_loader, val_loader, test_loader


def get_fingerveinsyn5m_loader(
    root: str = None,
    img_size: int = 112,
    batch_size: int = 16,
    num_workers: int = 0,
):
    """Get DataLoader for FingerVeinSyn-5M (pretraining)."""
    import os
    # Auto-detect path
    if root is None:
        possible_paths = [
            'd:/WorkSpace/Research/Research/MFFV/data/raw/fingerveinsyn5m_example',
            'D:/WorkSpace/Research/Research/MFFV/data/raw/fingerveinsyn5m_example',
            './data/raw/fingerveinsyn5m_example',
            '../data/raw/fingerveinsyn5m_example',
            'data/raw/fingerveinsyn5m_example',
        ]
        for path in possible_paths:
            if os.path.exists(path):
                root = path
                break
        if root is None:
            print(f"[WARNING] FingerVeinSyn-5M example not found")
            return None
    
    train_transform, _ = get_transforms(img_size, augment=True, strong=strong_augment)

    dataset = FingerVeinDataset(
        root=root, dataset_name='fingerveinsyn5m',
        transform=train_transform, mode='train',
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=num_workers, pin_memory=False)

    return loader


if __name__ == '__main__':
    import os
    # Quick test
    print("=" * 70)
    print("TEST: Unified Finger Vein Dataset")
    print("=" * 70)

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
        print("[WARNING] FV-USM dataset not found. Skipping dataset tests.")
        print(f"  Expected paths: {possible_paths}")
    else:
        print(f"\n[1] Testing FV-USM Dataset at: {fvusm_root}")
        fvusm_train, fvusm_val, fvusm_test = get_fvusm_loaders(
            root=fvusm_root,
            batch_size=8, num_workers=0, session='1st'
        )

        # Sample batch
        for imgs, labels in fvusm_train:
            print(f"  FV-USM train batch: imgs.shape={imgs.shape}, labels={len(labels)}")
            break

        # Test pair dataset
        print("\n[2] Testing Pair Dataset...")
        base_dataset = FingerVeinDataset(
            root=fvusm_root,
            dataset_name='fvusm', session='1st', mode='test',
        )
        pair_dataset = FingerVeinPairDataset(base_dataset, num_pairs=100)
        print(f"  Pair dataset size: {len(pair_dataset)}")
        img1, img2, label = pair_dataset[0]
        print(f"  Pair sample: img1.shape={img1.shape}, img2.shape={img2.shape}, label={label}")

    # Test metrics (no dataset needed)
    print("\n[3] Testing Metrics...")
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_scores = np.array([0.9, 0.8, 0.3, 0.2, 0.7, 0.4])
    eer, thr = compute_eer(y_true, y_scores)
    hter = compute_hter(y_true, y_scores, thr)
    auc = compute_roc_auc(y_true, y_scores)
    print(f"  EER: {eer:.4f}, Threshold: {thr:.4f}")
    print(f"  HTER: {hter:.4f}")
    print(f"  ROC AUC: {auc:.4f}")

    print("\n✓ Dataset tests complete!")
