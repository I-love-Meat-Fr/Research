# Download Instructions for Full Datasets

## FingerVeinSyn-5M (Synthetic, 5M images)

### Method 1: Kaggle CLI (Recommended)
```bash
# Install Kaggle CLI
pip install kaggle

# Setup credentials (https://www.kaggle.com/settings → API → Create New Token)
mkdir -p ~/.kaggle
mv ~/Downloads/kaggle.json ~/.kaggle/   # or wherever you saved it
chmod 600 ~/.kaggle/kaggle.json

# Download Part 1 (~10GB)
kaggle datasets download -d evanwang98/fingerveinsyn-5m

# Download Part 2 (~10GB)
kaggle datasets download -d evanwang98/fingerveinsyn-5m-part

# Unzip
unzip fingerveinsyn-5m.zip -d raw/fingerveinsyn5m/
unzip fingerveinsyn-5m-part.zip -d raw/fingerveinsyn5m/
```

### Method 2: Manual Download
Visit https://www.kaggle.com/datasets/evanwang98/fingerveinsyn-5m and download manually.

Expected size: ~20GB total, ~5M images.

## FV-USM (Real, 11,808 images)

Already in workspace! Extract:
```bash
unzip -q Published_database_FV-USM_Dec2013.zip -d external/
```

If starting fresh, download from [FV-USM official site](http://drfendi.com/fv_usm_database/) and accept TOS.

## MFFV-N (Real, Multi-Spectrum)

**Requires signed license agreement.**

1. Read `MFFV License Agreement.pdf` in repo root
2. Sign and email to: runrunjun@163.com (Junduan Huang)
3. Once approved, you'll receive access credentials
4. Place data at: `data/external/MFFV-N/`

Path convention from MFFV repo README:
```
/idiap/resource/database/MFFV-N/data
```

For this fork, we use:
```
data/external/MFFV-N/data
```

## SDUMLA-HMT (Optional)

Visit the request page and submit form (typically 1-2 weeks for approval).
Public alternative: search for community mirrors on Kaggle.

## HKPU-IFVF (Optional)

Available from Hong Kong Polytechnic University via request.