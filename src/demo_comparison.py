"""
Demo: Test finger vein comparison with synthetic data
=====================================================

This script demonstrates the comparison pipeline using synthetic finger vein images
when the actual data format is unknown.

Usage:
    python demo_comparison.py
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from tqdm import tqdm


# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def create_synthetic_vein_image(
    size: int = 112,
    num_lines: int = 15,
    seed: int = None
) -> np.ndarray:
    """
    Create a synthetic finger vein pattern image.
    
    This mimics the vein structure using line patterns.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Create blank canvas
    img = np.zeros((size, size), dtype=np.uint8)
    
    # Add horizontal vein-like lines
    for _ in range(num_lines):
        y = np.random.randint(10, size - 10)
        thickness = np.random.randint(1, 3)
        brightness = np.random.randint(80, 180)
        
        # Add some variation to the line
        for dy in range(-thickness, thickness + 1):
            for x in range(size):
                if np.random.random() > 0.3:  # Some gaps
                    yy = y + dy
                    if 0 <= yy < size:
                        img[yy, x] = brightness
    
    # Add some noise
    noise = np.random.randint(0, 30, (size, size))
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Add slight blur effect by averaging
    from scipy.ndimage import uniform_filter
    img = uniform_filter(img.astype(float), size=2).astype(np.uint8)
    
    return img


class ModelWrapper:
    """Wrapper for the trained model."""
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        import torchvision.models as models
        
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.embedding_dim = checkpoint.get('embedding_dim', 128)
        self.num_classes = checkpoint.get('num_classes', 344)
        self.img_size = checkpoint.get('img_size', 224)
        self.threshold = checkpoint.get('threshold', 0.5)
        
        # Build model
        self.model = self._build_model()
        self.model.load_state_dict(checkpoint['model_state'], strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded: embedding_dim={self.embedding_dim}, threshold={self.threshold:.4f}")
    
    def _build_model(self):
        """Build model matching checkpoint format."""
        import torchvision.models as models
        
        class WrapperResNet18(torch.nn.Module):
            def __init__(self, embedding_dim=128, num_classes=344):
                super().__init__()
                resnet = models.resnet18(weights=None)
                resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
                resnet.fc = torch.nn.Identity()
                self.backbone = resnet
                self.embedding = torch.nn.Linear(512, embedding_dim)
                self.classifier = torch.nn.Linear(embedding_dim, num_classes)
            
            def forward(self, x, return_embeddings=False):
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
        
        return WrapperResNet18(self.embedding_dim, self.num_classes)
    
    @torch.no_grad()
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """Extract embedding from image."""
        # Resize
        img = Image.fromarray(image)
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        arr = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor (grayscale to 1-channel)
        if arr.ndim == 3:
            arr = arr[:, :, 0]  # Take first channel
        
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, H, W]
        
        # Normalize
        mean = 0.5
        std = 0.5
        tensor = (tensor - mean) / std
        
        # Extract
        emb = self.model(tensor, return_embeddings=True)
        return emb.cpu().numpy().squeeze()
    
    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> Tuple[float, str]:
        """Compare two embeddings."""
        score = np.dot(emb1, emb2)
        verdict = 'GENUINE' if score > self.threshold else 'IMPOSTOR'
        return float(score), verdict


def demo_comparison(
    model_path: str,
    num_persons: int = 10,
    samples_per_person: int = 3,
    num_test_pairs: int = 50,
    device: str = 'auto'
):
    """
    Demo: Test model with synthetic finger vein images.
    """
    print("=" * 70)
    print("FINGER VEIN COMPARISON DEMO (Synthetic Data)")
    print("=" * 70)
    
    # Load model
    print(f"\n[1] Loading model from: {model_path}")
    model = ModelWrapper(model_path, device)
    
    # Create synthetic data
    print(f"\n[2] Creating synthetic finger vein data...")
    print(f"    Persons: {num_persons}")
    print(f"    Samples per person: {samples_per_person}")
    
    persons = {}
    for pid in range(num_persons):
        samples = []
        for s in range(samples_per_person):
            seed = pid * 100 + s
            img = create_synthetic_vein_image(size=112, num_lines=15, seed=seed)
            emb = model.extract_embedding(img)
            samples.append({
                'person_id': pid,
                'sample_id': s,
                'embedding': emb
            })
        persons[pid] = samples
    
    print(f"    Total samples: {len(persons) * samples_per_person}")
    
    # Generate test pairs
    print(f"\n[3] Generating {num_test_pairs} test pairs...")
    import random
    random.seed(42)
    
    pairs = []
    person_ids = list(persons.keys())
    
    # Genuine pairs
    for _ in range(num_test_pairs // 2):
        pid = random.choice(person_ids)
        s1, s2 = random.sample(persons[pid], 2)
        pairs.append((s1, s2, 1))
    
    # Impostor pairs
    for _ in range(num_test_pairs - len(pairs)):
        p1_id, p2_id = random.sample(list(persons.keys()), 2)
        p1 = persons[p1_id]
        p2 = persons[p2_id]
        s1 = random.choice(p1)
        s2 = random.choice(p2)
        pairs.append((s1, s2, 0))
    
    random.shuffle(pairs)
    print(f"    Genuine pairs: {sum(1 for _, _, l in pairs if l == 1)}")
    print(f"    Impostor pairs: {sum(1 for _, _, l in pairs if l == 0)}")
    
    # Run comparison
    print(f"\n[4] Running comparison...")
    y_true = []
    y_scores = []
    
    for s1, s2, label in tqdm(pairs, desc="Comparing"):
        score, verdict = model.compare(s1['embedding'], s2['embedding'])
        y_true.append(label)
        y_scores.append(score)
    
    # Compute metrics
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    
    # EER
    sorted_indices = np.argsort(y_scores)[::-1]
    y_true_sorted = y_true[sorted_indices]
    
    n_genuine = np.sum(y_true == 1)
    n_impostor = np.sum(y_true == 0)
    
    tp = np.cumsum(y_true_sorted == 1)
    fp = np.cumsum(y_true_sorted == 0)
    
    fnr = (n_genuine - tp) / n_genuine
    fpr = fp / n_impostor
    
    diff = np.abs(fnr - fpr)
    eer_idx = np.argmin(diff)
    eer = (fnr[eer_idx] + fpr[eer_idx]) / 2
    threshold = y_scores[sorted_indices[eer_idx]]
    
    # AUC
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y_true, y_scores)
    
    # Results
    print("\n" + "=" * 70)
    print("RESULTS (Synthetic Data)")
    print("=" * 70)
    print(f"Note: Results on synthetic data don't reflect real performance")
    print("-" * 70)
    print(f"EER:  {eer*100:.2f}%")
    print(f"AUC:  {auc:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print("=" * 70)
    
    return {
        'eer': float(eer),
        'auc': float(auc),
        'threshold': float(threshold),
        'num_pairs': len(y_true)
    }


def main():
    parser = argparse.ArgumentParser(description='Demo finger vein comparison')
    parser.add_argument('--checkpoint', type=str,
                       default=r'D:\WorkSpace\Research\Research\MFFV\src\results\baseline_resnet18_arcface_expD\best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--persons', type=int, default=20,
                       help='Number of synthetic persons')
    parser.add_argument('--samples', type=int, default=3,
                       help='Samples per person')
    parser.add_argument('--pairs', type=int, default=100,
                       help='Number of test pairs')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()
    
    results = demo_comparison(
        model_path=args.checkpoint,
        num_persons=args.persons,
        samples_per_person=args.samples,
        num_test_pairs=args.pairs,
        device=args.device
    )
    
    # Save
    output_file = 'demo_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
