"""
NIR Camera Data Decoder & Comparison Script
==========================================

This script handles encoded NIR (Near-Infrared) finger vein data from custom camera formats.
It can:
1. Decode .dat files using various methods
2. Use pre-decoded data in jpg folders
3. Compare decoded images using trained finger vein recognition models

Usage:
    python decode_and_compare.py --mode decode
    python decode_and_compare.py --mode compare
    python decode_and_compare.py --mode full

Author: Research Team
Date: 2026-09-04
"""

import os
import sys
import json
import argparse
import struct
import zlib
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from tqdm import tqdm

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


# ============================================================================
# CONFIGURATION
# ============================================================================

# Data paths
MODEL_DATA_ROOT = Path(r"D:\WorkSpace\Research\Research\model")
TRAINED_MODELS_DIR = Path(r"D:\WorkSpace\Research\Research\MFFV\src\results")

# Common image sizes for NIR cameras
COMMON_SIZES = [
    (64, 64),      # Small
    (80, 80),      # Common
    (96, 96),      # Medium
    (112, 112),    # Standard
    (128, 128),    # Medium-large
    (160, 120),    # 4:3
    (240, 320),    # Portrait
    (320, 240),    # Landscape
]


# ============================================================================
# DECODER
# ============================================================================

@dataclass
class DecodedImage:
    """Container for decoded image data."""
    data: np.ndarray  # Raw pixel data (H, W) or (H, W, C)
    width: int
    height: int
    channels: int
    source_path: str
    decoded_path: Optional[str] = None


class NIRDDataDecoder:
    """
    Decoder for NIR camera .dat files.
    
    The encoding format is detected automatically from the file structure.
    Supports:
    - XOR encoding with various keys
    - Zlib compression (raw deflate or zlib wrapper)
    - Custom header formats
    - Simple byte swapping
    """
    
    def __init__(self):
        self.results = []
    
    def detect_format(self, data: bytes) -> Dict:
        """
        Detect the encoding format from file content.
        
        Returns a dict with detected format info.
        """
        info = {
            'size': len(data),
            'format': 'unknown',
            'confidence': 0.0,
            'params': {}
        }
        
        # Check for zlib signature
        if data[:2] in [b'\x78\x9c', b'\x78\x01', b'\x78\xda']:
            info['format'] = 'zlib'
            info['confidence'] = 0.9
            return info
        
        # Check for gzip signature
        if data[:2] == b'\x1f\x8b':
            info['format'] = 'gzip'
            info['confidence'] = 0.95
            return info
        
        # Check for JPEG magic bytes (at any position)
        for i in range(min(100, len(data) - 3)):
            if data[i:i+3] == b'\xff\xd8\xff':
                info['format'] = 'xor_jpeg'
                info['params']['xor_offset'] = i
                info['confidence'] = 0.85
                return info
        
        # Check for PNG signature
        if data[:4] == b'\x89PNG':
            info['format'] = 'raw_png'
            info['confidence'] = 0.95
            return info
        
        # Check for BMP signature
        if data[:2] == b'BM':
            info['format'] = 'bmp'
            info['confidence'] = 0.95
            return info
        
        # Check if it's simple XOR encoded
        # by testing common image dimensions
        for size in COMMON_SIZES:
            w, h = size
            expected_size = w * h  # Grayscale
            if expected_size == len(data):
                info['format'] = 'xor_grayscale'
                info['params']['width'] = w
                info['params']['height'] = h
                info['confidence'] = 0.7
                return info
            
            expected_rgba = w * h * 4
            if expected_rgba == len(data):
                info['format'] = 'xor_rgba'
                info['params']['width'] = w
                info['params']['height'] = h
                info['confidence'] = 0.7
                return info
        
        # Check for header + data format
        # Many cameras use: [2-byte marker][2-byte width][2-byte height][data...]
        for offset in range(0, 10):
            if offset + 6 > len(data):
                break
            try:
                marker = data[offset:offset+2]
                w = struct.unpack('<H', data[offset+2:offset+4])[0]
                h = struct.unpack('<H', data[offset+4:offset+6])[0]
                
                if 16 <= w <= 2048 and 16 <= h <= 2048:
                    expected = w * h
                    if expected == len(data) - offset - 6:
                        info['format'] = 'header_raw'
                        info['params']['offset'] = offset + 6
                        info['params']['width'] = w
                        info['params']['height'] = h
                        info['confidence'] = 0.8
                        return info
            except:
                pass
        
        # Default: assume XOR encoded with unknown key
        info['format'] = 'xor_unknown'
        info['confidence'] = 0.5
        return info
    
    def try_decode(self, data: bytes, method: str, **kwargs) -> Optional[np.ndarray]:
        """
        Try to decode data using specified method.
        
        Returns decoded image as numpy array or None if failed.
        """
        try:
            if method == 'xor':
                key = kwargs.get('key', 0x42)
                decoded = np.frombuffer(data, dtype=np.uint8) ^ key
                return decoded
            
            elif method == 'xor_auto':
                # Try all possible keys and look for image patterns
                for key in range(1, 256):
                    decoded = np.frombuffer(data, dtype=np.uint8) ^ key
                    # Check for common image patterns
                    if self._looks_like_image(decoded):
                        return decoded
            
            elif method == 'zlib':
                return np.frombuffer(zlib.decompress(data), dtype=np.uint8)
            
            elif method == 'zlib_raw':
                return np.frombuffer(zlib.decompress(data, wbits=-15), dtype=np.uint8)
            
            elif method == 'header_raw':
                offset = kwargs.get('offset', 0)
                w = kwargs.get('width', 64)
                h = kwargs.get('height', 64)
                decoded = data[offset:offset + w * h]
                return np.frombuffer(decoded, dtype=np.uint8).reshape(h, w)
            
            elif method == 'grayscale':
                w = kwargs.get('width', 64)
                h = kwargs.get('height', 64)
                return np.frombuffer(data, dtype=np.uint8).reshape(h, w)
            
            elif method == 'rle':
                # Simple RLE decode
                decoded = bytearray()
                i = 0
                while i < len(data):
                    if i + 1 < len(data) and data[i] == 0xFF:
                        count = data[i+1]
                        i += 2
                    else:
                        count = 1
                        decoded.append(data[i])
                        i += 1
                return np.frombuffer(bytes(decoded), dtype=np.uint8)
            
            return None
            
        except Exception as e:
            return None
    
    def _looks_like_image(self, data: np.ndarray) -> bool:
        """
        Check if decoded data looks like image data.
        """
        if len(data) < 100:
            return False
        
        # Check variance (real images have variance)
        variance = np.var(data)
        if variance < 10:
            return False
        
        # Check range (should use full range)
        min_val = np.min(data)
        max_val = np.max(data)
        if max_val - min_val < 20:
            return False
        
        return True
    
    def decode(self, file_path: str, output_dir: Optional[str] = None) -> Optional[DecodedImage]:
        """
        Decode a .dat file.
        
        Args:
            file_path: Path to .dat file
            output_dir: Optional directory to save decoded image
            
        Returns:
            DecodedImage object or None if decoding failed
        """
        with open(file_path, 'rb') as f:
            data = f.read()
        
        # Detect format
        info = self.detect_format(data)
        
        # Try to decode
        decoded = None
        result_info = {
            'file': file_path,
            'format': info['format'],
            'confidence': info['confidence'],
            'params': info['params']
        }
        
        # Try methods based on detected format
        methods_to_try = []
        
        if info['format'] == 'zlib':
            methods_to_try.append(('zlib', {}))
        elif info['format'] == 'gzip':
            methods_to_try.append(('gzip', {}))
        elif info['format'] == 'xor_jpeg':
            methods_to_try.append(('xor_auto', {'offset': info['params'].get('xor_offset', 0)}))
        elif info['format'] in ['header_raw', 'xor_grayscale', 'xor_rgba']:
            methods_to_try.append(('grayscale', info['params']))
        else:
            # Unknown format - try multiple methods
            methods_to_try.append(('xor_auto', {}))
            for size in COMMON_SIZES:
                methods_to_try.append(('grayscale', {'width': size[0], 'height': size[1]}))
        
        for method, kwargs in methods_to_try:
            decoded = self.try_decode(data, method, **kwargs)
            if decoded is not None and self._looks_like_image(decoded):
                result_info['method'] = method
                result_info['params'] = kwargs
                break
        
        if decoded is None:
            self.results.append(result_info)
            return None
        
        # Ensure proper dimensions
        if decoded.ndim == 1:
            # Try to reshape to 2D
            size = int(np.sqrt(len(decoded)))
            if size * size == len(decoded):
                decoded = decoded.reshape(size, size)
            else:
                # Try other common aspect ratios
                for h in [32, 48, 52, 58, 64, 80, 96]:
                    w = len(decoded) // h
                    if w * h == len(decoded):
                        decoded = decoded.reshape(h, w)
                        break
        
        height, width = decoded.shape[:2] if decoded.ndim >= 2 else (0, 0)
        channels = 1 if decoded.ndim == 2 else (decoded.shape[2] if decoded.ndim >= 3 else 0)
        
        # Save if output directory specified
        decoded_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(file_path))[0]
            output_path = os.path.join(output_dir, f"{base_name}.png")
            try:
                if decoded.ndim == 2:
                    img = Image.fromarray(decoded.astype(np.uint8))
                else:
                    img = Image.fromarray(decoded.astype(np.uint8))
                img.save(output_path)
                decoded_path = output_path
            except Exception as e:
                print(f"Warning: Could not save decoded image: {e}")
        
        return DecodedImage(
            data=decoded,
            width=width,
            height=height,
            channels=channels,
            source_path=file_path,
            decoded_path=decoded_path
        )


# ============================================================================
# MODEL WRAPPER
# ============================================================================

class WrapperResNet18(torch.nn.Module):
    """
    ResNet18 wrapper that matches the saved checkpoint format.
    Uses 'backbone.' prefix for ResNet18 layers.
    """
    
    def __init__(self, embedding_dim: int = 128, num_classes: int = 492):
        super().__init__()
        import torchvision.models as models
        
        # Build ResNet18 backbone with 1-channel input
        resnet = models.resnet18(weights=None)
        resnet.conv1 = torch.nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.fc = torch.nn.Identity()
        
        # Rename layers with 'backbone.' prefix
        self.backbone = resnet
        
        # Embedding and classifier layers
        self.embedding = torch.nn.Linear(512, embedding_dim)
        self.classifier = torch.nn.Linear(embedding_dim, num_classes)
        
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
    
    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> torch.Tensor:
        # Extract features through backbone
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
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        
        if return_embeddings:
            return emb
        
        # Classification
        logits = self.classifier(emb)
        return logits


# ============================================================================
# MODEL LOADER & COMPARISON
# ============================================================================

class FingerVeinModel:
    """
    Wrapper for trained finger vein recognition models.
    """
    
    def __init__(self, checkpoint_path: str, device: str = 'auto'):
        """
        Load a trained model from checkpoint.
        
        Args:
            checkpoint_path: Path to .pth checkpoint file
            device: 'auto', 'cuda', or 'cpu'
        """
        if device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        self._load_model(checkpoint_path)
    
    def _load_model(self, checkpoint_path: str):
        """Load model from checkpoint."""
        import torchvision.models as models
        from torchvision.models import ResNet18_Weights
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Get metadata
        self.embedding_dim = checkpoint.get('embedding_dim', 128)
        self.num_classes = checkpoint.get('num_classes', 344)  # Use from checkpoint
        self.img_size = checkpoint.get('img_size', 224)
        self.threshold = checkpoint.get('threshold', 0.5)
        self.eer = checkpoint.get('eer', None)
        
        # Build model with wrapper class to match saved format
        self.model = WrapperResNet18(embedding_dim=self.embedding_dim, num_classes=self.num_classes).to(self.device)
        
        # Load weights - handle 'backbone.' prefix in state dict
        state_dict = checkpoint['model_state']
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded: embedding_dim={self.embedding_dim}, num_classes={self.num_classes}, threshold={self.threshold:.4f}")
    
    @torch.no_grad()
    def extract_embedding(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding from an image.
        
        Args:
            image: Grayscale image as numpy array (H, W)
            
        Returns:
            128-dim embedding vector
        """
        # Preprocess
        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)  # Convert to RGB
        
        # Resize and normalize
        img = Image.fromarray(image.astype(np.uint8))
        img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        img = np.array(img).astype(np.float32) / 255.0
        
        # Convert to tensor (N, C, H, W) with C=3 for RGB
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        
        # Normalize with ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        tensor = (tensor - mean) / std
        
        # Extract embedding
        emb = self.model(tensor, return_embeddings=True)
        
        return emb.cpu().numpy().squeeze()
    
    def compare(self, emb1: np.ndarray, emb2: np.ndarray) -> Tuple[float, str]:
        """
        Compare two embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            (cosine_similarity, verdict)
        """
        score = np.dot(emb1, emb2)
        
        if score > self.threshold:
            verdict = 'GENUINE'
        else:
            verdict = 'IMPOSTOR'
        
        return score, verdict


# ============================================================================
# DATA LOADER
# ============================================================================

class NIRDataset:
    """
    Dataset loader for NIR finger vein data.
    
    Supports both:
    - .dat files (encoded)
    - .jpg folders (pre-decoded)
    """
    
    def __init__(self, root_dir: str, decoder: Optional[NIRDDataDecoder] = None):
        """
        Initialize dataset loader.
        
        Args:
            root_dir: Root directory containing person folders
            decoder: Optional decoder instance
        """
        self.root = Path(root_dir)
        self.decoder = decoder or NIRDDataDecoder()
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Dict]:
        """Load all samples from the dataset."""
        samples = []
        
        for person_dir in sorted(self.root.iterdir()):
            if not person_dir.is_dir():
                continue
            
            person_id = person_dir.name
            
            # Check for .dat files
            for dat_file in person_dir.glob('*.dat'):
                samples.append({
                    'person_id': person_id,
                    'dat_path': str(dat_file),
                    'jpg_path': str(person_dir / 'jpg' / f'm{dat_file.stem}.jpg'),
                    'type': 'dat'
                })
            
            # Also check for pre-decoded jpg files
            jpg_dir = person_dir / 'jpg'
            if jpg_dir.exists():
                for jpg_file in jpg_dir.glob('*.jpg'):
                    # Skip if already have corresponding dat
                    base_name = jpg_file.stem.replace('m', '')  # m1 -> 1
                    dat_path = person_dir / f'{base_name}.dat'
                    if not dat_path.exists():
                        samples.append({
                            'person_id': person_id,
                            'jpg_path': str(jpg_file),
                            'dat_path': None,
                            'type': 'jpg'
                        })
        
        return samples
    
    def get_pair_generator(self, num_pairs: int = 1000) -> List[Tuple[Dict, Dict, int]]:
        """
        Generate pairs for verification testing.
        
        Returns:
            List of (sample1, sample2, label) tuples
            label: 1 for genuine (same person), 0 for impostor (different people)
        """
        import random
        random.seed(42)
        
        # Group by person
        person_groups = {}
        for sample in self.samples:
            pid = sample['person_id']
            if pid not in person_groups:
                person_groups[pid] = []
            person_groups[pid].append(sample)
        
        persons = list(person_groups.keys())
        pairs = []
        
        # Generate genuine pairs (same person)
        num_genuine = num_pairs // 2
        for _ in range(num_genuine):
            person = random.choice(persons)
            if len(person_groups[person]) >= 2:
                s1, s2 = random.sample(person_groups[person], 2)
                pairs.append((s1, s2, 1))
        
        # Generate impostor pairs (different people)
        num_impostor = num_pairs - len(pairs)
        for _ in range(num_impostor):
            p1, p2 = random.sample(persons, 2)
            s1 = random.choice(person_groups[p1])
            s2 = random.choice(person_groups[p2])
            pairs.append((s1, s2, 0))
        
        random.shuffle(pairs)
        return pairs
    
    def load_image(self, sample: Dict) -> Optional[np.ndarray]:
        """
        Load and decode an image.
        
        Args:
            sample: Sample dict with paths
            
        Returns:
            Decoded image as numpy array or None
        """
        # Try to load as decoded jpg first
        jpg_path = sample.get('jpg_path')
        if jpg_path and os.path.exists(jpg_path):
            try:
                # First try normal JPEG loading
                img = Image.open(jpg_path)
                if img.mode != 'L':
                    img = img.convert('L')
                arr = np.array(img)
                if arr.size > 100:  # Valid image
                    return arr
            except:
                pass
            
            # If JPEG loading failed, try to decode as encoded data
            try:
                with open(jpg_path, 'rb') as f:
                    data = f.read()
                
                # Check for fixed 32-byte header (common in this dataset)
                fixed_header = bytes.fromhex('2d6a00ddd7c5aed7470eff72625291b8ee079478fb1e755e0e4a8d2d87b8b582')
                if data[:32] == fixed_header:
                    data = data[32:]  # Strip header
                
                # Try to interpret as image
                if len(data) == 4096:  # 64x64
                    img = data.reshape(64, 64)
                    return img.astype(np.uint8)
                elif len(data) == 1024:  # 32x32
                    img = data.reshape(32, 32)
                    return img.astype(np.uint8)
                elif len(data) == 16384:  # 128x128
                    img = data.reshape(128, 128)
                    return img.astype(np.uint8)
                elif len(data) == 14400:  # 120x120
                    img = data.reshape(120, 120)
                    return img.astype(np.uint8)
                elif len(data) == 2304:  # 48x48
                    img = data.reshape(48, 48)
                    return img.astype(np.uint8)
                elif len(data) == 5184:  # 72x72
                    img = data.reshape(72, 72)
                    return img.astype(np.uint8)
                elif len(data) == 7744:  # 88x88
                    img = data.reshape(88, 88)
                    return img.astype(np.uint8)
            except:
                pass
        
        # Try to decode .dat file
        dat_path = sample.get('dat_path')
        if dat_path and os.path.exists(dat_path):
            decoded = self.decoder.decode(dat_path)
            if decoded is not None:
                data = decoded.data
                if data.ndim > 1:
                    return data
                # Try to reshape
                size = int(np.sqrt(len(data)))
                if size * size == len(data):
                    return data.reshape(size, size).astype(np.uint8)
        
        return None


# ============================================================================
# MAIN COMPARISON PIPELINE
# ============================================================================

def compare_dataset(
    model_path: str,
    data_root: str,
    num_pairs: int = 1000,
    device: str = 'auto'
) -> Dict:
    """
    Compare samples in the NIR dataset using a trained model.
    
    Args:
        model_path: Path to model checkpoint
        data_root: Root directory of NIR data
        num_pairs: Number of pairs to test
        device: Device to use
        
    Returns:
        Dictionary with comparison results
    """
    print("=" * 70)
    print("NIR FINGER VEIN COMPARISON")
    print("=" * 70)
    
    # Load model
    print(f"\n[1] Loading model from: {model_path}")
    model = FingerVeinModel(model_path, device)
    
    # Load dataset
    print(f"\n[2] Loading NIR dataset from: {data_root}")
    dataset = NIRDataset(data_root)
    print(f"    Total samples: {len(dataset.samples)}")
    
    # Get pairs
    print(f"\n[3] Generating {num_pairs} test pairs...")
    pairs = dataset.get_pair_generator(num_pairs)
    print(f"    Genuine pairs: {sum(1 for _, _, l in pairs if l == 1)}")
    print(f"    Impostor pairs: {sum(1 for _, _, l in pairs if l == 0)}")
    
    # Run comparison
    print(f"\n[4] Running comparison...")
    y_true = []
    y_scores = []
    
    for i, (s1, s2, label) in enumerate(tqdm(pairs, desc="Comparing")):
        # Load images
        img1 = dataset.load_image(s1)
        img2 = dataset.load_image(s2)
        
        if img1 is None or img2 is None:
            continue
        
        # Extract embeddings
        try:
            emb1 = model.extract_embedding(img1)
            emb2 = model.extract_embedding(img2)
            
            # Compare
            score, verdict = model.compare(emb1, emb2)
            
            y_true.append(label)
            y_scores.append(score)
        except Exception as e:
            print(f"Error comparing {s1['person_id']} vs {s2['person_id']}: {e}")
            continue
    
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
    
    # HTER at EER threshold
    y_pred = (y_scores >= threshold).astype(int)
    far = np.sum((y_pred == 1) & (y_true == 0)) / max(1, np.sum(y_true == 0))
    frr = np.sum((y_pred == 0) & (y_true == 1)) / max(1, np.sum(y_true == 1))
    hter = (far + frr) / 2
    
    # Results
    results = {
        'num_pairs': len(y_true),
        'num_genuine': n_genuine,
        'num_impostor': n_impostor,
        'eer': float(eer),
        'hter': float(hter),
        'auc': float(auc),
        'threshold': float(threshold),
        'far': float(far),
        'frr': float(frr),
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Dataset: NIR Finger Vein (custom camera)")
    print(f"Model: {model_path}")
    print(f"Test Pairs: {results['num_pairs']}")
    print(f"  Genuine: {results['num_genuine']}")
    print(f"  Impostor: {results['num_impostor']}")
    print("-" * 70)
    print(f"EER:  {results['eer']*100:.2f}%")
    print(f"HTER: {results['hter']*100:.2f}%")
    print(f"AUC:  {results['auc']:.4f}")
    print(f"Threshold: {results['threshold']:.4f}")
    print("=" * 70)
    
    return results


def decode_all(data_root: str, output_root: str = None, max_samples: int = None):
    """
    Decode all .dat files in the dataset.
    
    Args:
        data_root: Root directory of NIR data
        output_root: Optional output directory for decoded images
        max_samples: Maximum number of samples to decode
    """
    print("=" * 70)
    print("DECODING NIR DATA")
    print("=" * 70)
    
    decoder = NIRDDataDecoder()
    success_count = 0
    fail_count = 0
    
    samples = []
    for person_dir in Path(data_root).iterdir():
        if not person_dir.is_dir():
            continue
        for dat_file in person_dir.glob('*.dat'):
            samples.append(dat_file)
    
    print(f"Found {len(samples)} .dat files")
    
    for i, dat_path in enumerate(tqdm(samples, desc="Decoding")):
        if max_samples and i >= max_samples:
            break
        
        output_dir = None
        if output_root:
            person_dir = dat_path.parent
            output_dir = str(Path(output_root) / person_dir.name / 'decoded')
        
        result = decoder.decode(str(dat_path), output_dir)
        
        if result is not None:
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\nDecoding complete:")
    print(f"  Success: {success_count}")
    print(f"  Failed: {fail_count}")
    
    # Save results
    results_file = Path(output_root or data_root) / 'decode_results.json'
    with open(results_file, 'w') as f:
        json.dump(decoder.results, f, indent=2)
    print(f"  Results saved to: {results_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='NIR Finger Vein Data Decoder & Comparison',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Decode all .dat files
  python decode_and_compare.py --mode decode --data_root D:/WorkSpace/Research/Research/model

  # Compare dataset using trained model
  python decode_and_compare.py --mode compare \\
      --data_root D:/WorkSpace/Research/Research/model \\
      --checkpoint D:/WorkSpace/Research/Research/MFFV/src/results/baseline_resnet18_arcface_expD/best_model.pth

  # Full pipeline (decode + compare)
  python decode_and_compare.py --mode full \\
      --data_root D:/WorkSpace/Research/Research/model \\
      --checkpoint D:/WorkSpace/Research/Research/MFFV/src/results/baseline_resnet18_arcface_expD/best_model.pth
        """
    )
    
    parser.add_argument('--mode', type=str, default='compare',
                       choices=['decode', 'compare', 'full'],
                       help='Mode: decode files, compare dataset, or full pipeline')
    
    parser.add_argument('--data_root', type=str,
                       default=r'D:\WorkSpace\Research\Research\model',
                       help='Root directory of NIR data')
    
    parser.add_argument('--checkpoint', type=str,
                       default=r'D:\WorkSpace\Research\Research\MFFV\src\results\baseline_resnet18_arcface_expD\best_model.pth',
                       help='Path to model checkpoint')
    
    parser.add_argument('--output_root', type=str, default=None,
                       help='Output directory for decoded images')
    
    parser.add_argument('--num_pairs', type=int, default=1000,
                       help='Number of pairs to test in comparison mode')
    
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cuda, or cpu')
    
    parser.add_argument('--max_decode', type=int, default=None,
                       help='Maximum number of samples to decode')
    
    args = parser.parse_args()
    
    if args.mode in ['decode', 'full']:
        decode_all(
            data_root=args.data_root,
            output_root=args.output_root,
            max_samples=args.max_decode
        )
    
    if args.mode in ['compare', 'full']:
        results = compare_dataset(
            model_path=args.checkpoint,
            data_root=args.data_root,
            num_pairs=args.num_pairs,
            device=args.device
        )
        
        # Save results
        output_file = Path(args.output_root or '.') / 'nir_comparison_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_file}")


if __name__ == '__main__':
    main()
