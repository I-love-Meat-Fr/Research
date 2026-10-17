"""
augmentation.py — Strong augmentation for finger vein enrollment (Option A).

User uploads 1 image; this module generates 11 augmented variants with strong
parameters (rotation ±15°, translation ±12%, etc.) to spread the user's
embedding distribution in 128-D ArcFace space.

Augmentation strength is intentionally stronger than training-time augmentation
because:
- We have only 1 real sample, so we MUST artificially diversify.
- Risk: augmented samples may not perfectly match the real distribution.
- Mitigation: the real embedding is included as sample 0 with weight 1.

Designed for finger vein images but works for any grayscale-ish biometric image.
"""

from __future__ import annotations

import json
import random
from typing import Callable

import cv2
import numpy as np
from PIL import Image


# Default config — 11 augmentations
DEFAULT_AUGMENTATIONS = [
    {"name": "rotation",        "params_fn": lambda rng: {"angle_deg": rng.uniform(-15, 15)}},
    {"name": "translation",     "params_fn": lambda rng: {"tx": rng.uniform(-0.12, 0.12), "ty": rng.uniform(-0.12, 0.12)}},
    {"name": "brightness",      "params_fn": lambda rng: {"factor": rng.uniform(0.75, 1.25)}},
    {"name": "contrast",        "params_fn": lambda rng: {"factor": rng.uniform(0.80, 1.20)}},
    {"name": "gamma",           "params_fn": lambda rng: {"gamma": rng.uniform(0.70, 1.40)}},
    {"name": "noise",           "params_fn": lambda rng: {"sigma": rng.uniform(0, 0.015)}},
    {"name": "motion_blur",     "params_fn": lambda rng: {"kernel": rng.choice([3, 5])}},
    {"name": "scale",           "params_fn": lambda rng: {"factor": rng.uniform(0.92, 1.08)}},
    {"name": "elastic",         "params_fn": lambda rng: {"alpha": rng.uniform(2, 4), "sigma": 3}},
    {"name": "combo_geo_photo", "params_fn": lambda rng: {"angle_deg": rng.uniform(-10, 10), "tx": rng.uniform(-0.08, 0.08), "ty": rng.uniform(-0.08, 0.08), "brightness": rng.uniform(0.85, 1.15), "contrast": rng.uniform(0.88, 1.12)}},
    {"name": "combo_gamma_noise", "params_fn": lambda rng: {"gamma": rng.uniform(0.80, 1.25), "sigma": rng.uniform(0, 0.010), "blur_kernel": rng.choice([3, 5])}},
]


def _pil_to_cv(img: Image.Image) -> np.ndarray:
    """PIL RGB -> OpenCV BGR (uint8)."""
    arr = np.array(img.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    """OpenCV BGR (uint8) -> PIL RGB."""
    rgb = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def _apply_rotation(img_bgr: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _apply_translation(img_bgr: np.ndarray, tx: float, ty: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    M = np.float32([[1, 0, tx * w], [0, 1, ty * h]])
    return cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def _apply_brightness_contrast(img_bgr: np.ndarray, brightness: float, contrast: float) -> np.ndarray:
    out = cv2.convertScaleAbs(img_bgr, alpha=contrast, beta=int((brightness - 1.0) * 128))
    return out


def _apply_gamma(img_bgr: np.ndarray, gamma: float) -> np.ndarray:
    inv = 1.0 / max(gamma, 1e-6)
    table = np.array([(i / 255.0) ** inv * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(img_bgr, table)


def _apply_noise(img_bgr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img_bgr
    noise = np.random.normal(0, sigma * 255, img_bgr.shape).astype(np.float32)
    out = np.clip(img_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def _apply_motion_blur(img_bgr: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 1:
        return img_bgr
    kernel = kernel if kernel % 2 == 1 else kernel + 1
    M = np.zeros((kernel, kernel))
    M[kernel // 2, :] = np.ones(kernel)
    M /= kernel
    return cv2.filter2D(img_bgr, -1, M)


def _apply_scale(img_bgr: np.ndarray, factor: float) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    new_w, new_h = int(w * factor), int(h * factor)
    resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    if factor >= 1.0:
        # Crop center to original size
        x0 = (new_w - w) // 2
        y0 = (new_h - h) // 2
        return resized[y0:y0 + h, x0:x0 + w]
    else:
        # Pad to original size
        pad_w = (w - new_w) // 2
        pad_h = (h - new_h) // 2
        return cv2.copyMakeBorder(resized, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT)


def _apply_elastic(img_bgr: np.ndarray, alpha: float, sigma: float) -> np.ndarray:
    """Mild elastic deformation. alpha <= 4 to avoid destroying vein topology."""
    h, w = img_bgr.shape[:2]
    # Random displacement fields
    dx = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv2.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    x_idx, y_idx = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    map_x = (x_idx + dx).astype(np.float32)
    map_y = (y_idx + dy).astype(np.float32)
    return cv2.remap(img_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


_AUG_FUNCTIONS = {
    "rotation":     lambda img, p: _apply_rotation(img, p["angle_deg"]),
    "translation":  lambda img, p: _apply_translation(img, p["tx"], p["ty"]),
    "brightness":   lambda img, p: _apply_brightness_contrast(img, p["factor"], 1.0),
    "contrast":     lambda img, p: _apply_brightness_contrast(img, 1.0, p["factor"]),
    "gamma":        lambda img, p: _apply_gamma(img, p["gamma"]),
    "noise":        lambda img, p: _apply_noise(img, p["sigma"]),
    "motion_blur":  lambda img, p: _apply_motion_blur(img, p["kernel"]),
    "scale":        lambda img, p: _apply_scale(img, p["factor"]),
    "elastic":      lambda img, p: _apply_elastic(img, p["alpha"], p["sigma"]),
    "combo_geo_photo": lambda img, p: _apply_translation(_apply_rotation(_apply_brightness_contrast(img, p["brightness"], p["contrast"]), p["angle_deg"]), p["tx"], p["ty"]),
    "combo_gamma_noise": lambda img, p: _apply_motion_blur(_apply_noise(_apply_gamma(img, p["gamma"]), p["sigma"]), p["blur_kernel"]),
}


def _apply_one_augmentation(img: Image.Image, name: str, params: dict) -> Image.Image:
    """Apply one named augmentation to a PIL image. Returns a new PIL image."""
    if name not in _AUG_FUNCTIONS:
        raise ValueError(f"Unknown augmentation: {name}")
    bgr = _pil_to_cv(img)
    out = _AUG_FUNCTIONS[name](bgr, params)
    return _cv_to_pil(out)


def generate_augmented_embeddings(
    image: Image.Image,
    base_extract_fn: Callable[[Image.Image], np.ndarray],
    n_augmentations: int = 11,
    seed: int | None = None,
) -> list[tuple[np.ndarray, bool, str]]:
    """Generate 1 real + N augmented embeddings for one finger vein image.

    Args:
        image: Source PIL image (the user's single upload).
        base_extract_fn: Callable that takes a PIL image and returns a 128-D
            L2-normalized numpy embedding (from your ArcFace model).
        n_augmentations: How many augmented variants to generate (default 11).
        seed: Optional RNG seed for reproducibility.

    Returns:
        list of (embedding, is_augmented, aug_config_json) tuples.
        The first entry is the real embedding (is_augmented=False).
        Subsequent entries are augmented.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    results: list[tuple[np.ndarray, bool, str]] = []

    # 0. Real embedding
    real_emb = base_extract_fn(image)
    results.append((real_emb, False, ""))

    # 1..N. Augmented embeddings
    rng = random.Random(seed) if seed is not None else random.Random()
    aug_defs = DEFAULT_AUGMENTATIONS[:n_augmentations]
    for aug_def in aug_defs:
        params = aug_def["params_fn"](rng)
        try:
            aug_img = _apply_one_augmentation(image, aug_def["name"], params)
            aug_emb = base_extract_fn(aug_img)
            config_str = json.dumps({"name": aug_def["name"], "params": params})
            results.append((aug_emb, True, config_str))
        except Exception as e:
            # Skip augmentations that fail (e.g., extreme parameters)
            print(f"[augmentation] Skipped {aug_def['name']}: {e}")
            continue

    return results


if __name__ == "__main__":
    # Smoke test
    import sys
    from pathlib import Path
    test_path = Path(__file__).parent.parent / "data" / "raw" / "fingerveinsyn5m_example" / "04551_042_Exposure.png"
    if not test_path.exists():
        print(f"Test image not found: {test_path}")
        sys.exit(0)
    img = Image.open(test_path)
    print(f"Loaded test image: {img.size}")

    # Dummy extractor for testing
    def fake_extract(x):
        rng = np.random.default_rng(hash(str(x.size)) % (2**32))
        v = rng.standard_normal(128).astype(np.float32)
        return v / np.linalg.norm(v)

    results = generate_augmented_embeddings(img, fake_extract, n_augmentations=11, seed=42)
    print(f"Generated {len(results)} embeddings")
    for emb, is_aug, cfg in results:
        print(f"  is_aug={is_aug} shape={emb.shape} norm={np.linalg.norm(emb):.4f}")
