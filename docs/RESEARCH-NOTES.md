# Research Notes & Roadmap

## Project Status

**Current stage:** Setup & data acquisition
**Target:** Cross-domain finger vein recognition benchmark

## Datasets Available

| Dataset | Status | Size | Use Case |
|---------|--------|------|----------|
| FingerVeinSyn-5M (example) | ✅ Available | 100 images | Testing, dev |
| FingerVeinSyn-5M (full) | ⏳ Download needed | 5M images | Pretraining |
| FV-USM | ✅ Available | 11,808 images | Real-world benchmark |
| MFFV-N | ⏳ Pending agreement | Multi-spectral | Main real benchmark |

## Research Direction

Based on literature review (`docs/literature_review_finger_vein_2025.md`), the gap analysis identified 6 research opportunities:

| # | Gap | Our Approach |
|---|-----|--------------|
| 1 | Cross-domain transfer | Define standardized protocol |
| 2 | Synthetic-to-real gap | Systematic measurement |
| 3 | Few-shot benchmarks | 1-shot / 5-shot / 10-shot |
| 4 | Anti-spoofing | Future work |
| 5 | Standardized metrics | EER + FMR100 + FMR1000 + Acc |
| 6 | Real-world deployment | Latency benchmark |

## Roadmap

### Phase 1: Setup (Current)
- [x] Clone MFFV fork
- [x] Add example data from FingerVeinSyn-5M
- [x] Setup FV-USM in external/
- [ ] Download full FingerVeinSyn-5M from Kaggle
- [ ] Request MFFV-N license agreement
- [ ] Setup Python environment (conda + PyTorch + bob-bio)

### Phase 2: Baseline Reproduction
- [ ] Verify MFFV's FVIA + MiuraMatch on FV-USM
- [ ] Reproduce numbers from MFFV paper (Table VII-X)
- [ ] Sanity check vs reported metrics

### Phase 3: Method Implementation
- [ ] Re-implement 4 SOTA baselines:
  - [ ] DenseNet-121 + Channel Attention (Devkota & Kim 2024)
  - [ ] FVTCNet (CNN-Transformer hybrid)
  - [ ] FV-DMHN (MHCN + MHSA)
  - [ ] DiffVein (diffusion-based)
- [ ] Implement unified training pipeline

### Phase 4: Cross-Domain Experiments
- [ ] Train on FingerVeinSyn-5M (subsets)
- [ ] Test on FV-USM, MFFV-N
- [ ] Run few-shot benchmarks
- [ ] Generate learning curves

### Phase 5: Paper Writing
- [ ] Draft methodology
- [ ] Results tables
- [ ] Visualization figures
- [ ] Submit to TIFS / ICB / Pattern Recognition

## Key References

See `docs/literature_review_finger_vein_2025.md` for full bibliography.

## Code Structure

```
MFFV/
├── FE/             # Feature Extraction (from MFFV paper)
├── FVIA/           # Finger Vein Image Analysis (FVIA experiments)
├── FVAB/           # Finger Vein Analysis Benchmark
├── PT/             # Protocols (train/test splits)
├── img_rm/         # Architecture diagrams
├── data/           # Our datasets (gitignored)
├── docs/           # Literature review + analysis
├── src/            # Our new code (custom architectures, training)
├── notebooks/      # Jupyter notebooks for exploration
└── results/        # Experimental outputs
```