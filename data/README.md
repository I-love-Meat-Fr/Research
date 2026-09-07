# Datasets

This directory contains datasets used in our research. **Raw data is NOT committed to git** due to size and licensing constraints.

## Datasets

### 1. FingerVeinSyn-5M (Synthetic Pretraining)
- **Source:** [EvanWang98/FingerVeinSyn-5M](https://github.com/EvanWang98/FingerVeinSyn-5M)
- **License:** CC BY 4.0
- **Citation:** Wang et al., "FingerVeinSyn-5M: A Million-Scale Dataset and Benchmark for Finger Vein Recognition", ACM MM 2025
- **Download:** Kaggle (see `DOWNLOAD.md`)
- **Used as:** Pretraining data (synthetic)

### 2. FV-USM (Real Benchmark)
- **Source:** [FV-USM](http://drfendi.com/fv_usm_database/)
- **License:** Academic use only (TOS)
- **Citation:** Asaari et al., "Fusion of Finger Vein and Finger Print for Identity Verification", IWBF 2014
- **Stats:** 164 subjects × 6 fingers × 2 sessions × 6 images = 11,808 images
- **Used as:** Real-world cross-domain evaluation

### 3. MFFV-N (Real Multi-Spectrum) — Primary dataset for this repo
- **Source:** Signed license agreement (see `MFFV License Agreement.pdf` in repo root)
- **License:** Requires signed agreement with authors
- **Citation:** Huang et al., TIFS 2024 (MFFV paper)
- **Contact:** runrunjun@163.com (Junduan Huang)
- **Used as:** Main real benchmark with multi-spectral channels

## Directory Structure

```
data/
├── README.md                  # This file
├── DOWNLOAD.md                # Download instructions for full datasets
├── raw/                       # Raw images (gitignored)
│   ├── fingerveinsyn5m_example/   # 100 sample images (CC BY 4.0)
│   ├── annotations_example/       # 100 XML annotations
│   ├── vein_masks_example/        # 100 vein pattern masks
│   └── shape_masks_example/       # 100 finger shape masks
├── external/                  # Real datasets (gitignored)
│   └── Published_database_FV-USM_Dec2013/
│       ├── 1st_session/{raw_data,extractedvein}/
│       └── 2nd_session/{raw_data,extractedvein}/
└── processed/                 # Preprocessed data (gitignored)
    ├── ROI/                   # ROI-extracted images
    └── features/              # Extracted features
```

## Important Note on ENG-SDK-C++

The original workspace contained `ENG-SDK-C++/` with vendor SDK binaries (.dll, .msi, .docx) for a commercial finger vein sensor. **These files are NOT included in this fork** to avoid legal issues. See `docs/LICENSE-ANALYSIS.md` for details.

If you need to use the vendor sensor:
1. Contact the vendor directly for licensing
2. Apply for IRB approval if collecting new data
3. Use the SDK API only for research purposes (not redistribution)