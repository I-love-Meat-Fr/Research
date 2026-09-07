# Phân Tích Tình Trạng Dự Án Hiện Tại

> **Ngày:** 2026-08-09
> **Phạm vi:** `/home/qa/Workspace/Research/MFFV/`
> **Trạng thái:** ✅ Pipeline hoạt động — ⚠️ Chưa đạt paper-quality

---

## 1. Tổng Quan

| Aspect | Trạng thái |
|--------|------------|
| **Mục tiêu** | Benchmark 3 baselines cho finger vein recognition trên FV-USM |
| **Stack** | Python 3.12 + PyTorch 2.13 (CPU mode) |
| **Models** | 3 wrappers (FDT, GLA-FD, ILCNN) — 2/3 chưa dùng architecture gốc |
| **Data** | FV-USM (11,808 ảnh, 492 fingers) ✅ đầy đủ / FingerVeinSyn-5M chỉ có 100 ảnh sample |
| **Smoke test** | ✅ Đã chạy được 1 epoch |
| **Paper-quality results** | ❌ Chưa đạt (EER ~28%, paper claim ~0.5–2%) |

---

## 2. Cấu Trúc Code Đã Thêm Vào

```
MFFV/
├── src/
│   ├── datasets.py          ← 17.2 KB, 451 dòng  (Dataset + Metrics)
│   ├── train.py             ← 18.6 KB, 497 dòng  (Training pipeline)
│   ├── quick_start.py       ← 1.1 KB              (Smoke test runner)
│   ├── README.md            ← 4.2 KB              (Docs)
│   └── third_party/
│       ├── FDT/             ← Cloned SCUT-BIP-Lab/FDT (TIFS 2025)
│       ├── GLA-FD/          ← Cloned liangying-Ke/GLA-FD (IEEE Access 2025)
│       └── ILCNN/           ← Cloned liangying-Ke/Finger-vein-recognition (Bioengineering 2023)
└── results/
    └── fdt/
        └── results.json     ← Kết quả smoke test
```

---

## 3. Trạng Thái Từng Thành Phần

### 3.1. Dataset Loader (`datasets.py`) — ✅ Hoàn chỉnh

| Tính năng | Trạng thái | Ghi chú |
|-----------|------------|---------|
| Load FV-USM từ folder structure | ✅ Hoạt động | Parse `{session}/raw_data/{user}_{finger}/0{idx}.jpg` |
| Load FingerVeinSyn-5M | ✅ Hoạt động | Parse `{identity}_{seq}_{variant}.png` |
| Train/Val/Test split (class-disjoint) | ✅ Hoạt động | Tránh data leakage giữa splits |
| Map `class_id` → integer label | ✅ Hoạt động | Fix bug lúc đầu (hash collision) |
| Pair generation cho verification | ✅ Hoạt động | Genuine + impostor pairs |
| Metrics: EER, HTER, ROC AUC | ✅ Hoạt động | Pure NumPy, không cần sklearn |
| Data augmentation | ✅ Hoạt động | Flip + Rotation + ColorJitter |
| Custom collate function cho pairs | ✅ Hoạt động | Handle shape mismatch |

**Kết quả load thực tế:**
```
FV-USM train: 2064 samples | 344 unique IDs
FV-USM val:    438 samples |  73 unique IDs
FV-USM test:   450 samples |  75 unique IDs
Total:        2952 samples / 492 fingers  (sai khác 11,808 vì dùng session='1st')
```

⚠️ **Vấn đề:** Mặc định chỉ dùng `session='1st'` → chỉ train trên 1 session. Paper dùng `session='all'` để cross-session evaluation.

### 3.2. Training Pipeline (`train.py`) — ⚠️ Có chỗ cần fix

| Tính năng | Trạng thái | Ghi chú |
|-----------|------------|---------|
| Model factory (`get_model`) | ✅ Hoạt động | Switch giữa 3 models |
| Classification head wrapper | ✅ Hoạt động | Linear(128 → num_classes) |
| Training loop | ✅ Hoạt động | AdamW + CosineAnnealing + grad clip |
| Verification evaluation | ✅ Hoạt động | Cosine similarity trên embeddings |
| Save results to JSON | ✅ Hoạt động | Per-model folder |
| ArcFace loss | ❌ Chưa dùng | Class có sẵn nhưng đang dùng CrossEntropy |
| Contrastive loss | ❌ Chưa dùng | Class có sẵn |
| Multi-GPU support | ❌ Chưa có | Hiện chỉ single device |

### 3.3. Model Wrappers — ⚠️ Chưa đạt yêu cầu

| Model | Backbone thực tế | Paper backbone | Trạng thái |
|-------|------------------|----------------|------------|
| **FDT** | ResNet18 (11.2M params) | Dynamic Transformer + BatchAttn + LFFN | ⚠️ Fallback |
| **GLA-FD** | Simple version (try-import fails) | ViT + Feature Decoupling | ⚠️ Fallback |
| **ILCNN** | ILCNN gốc (import trực tiếp từ repo) | ILCNN + CoordAttention | ✅ Đúng |

**Code liên quan (`train.py`):**
```python
class FDTWrapper(nn.Module):
    def _build_fdt_simplified(self):
        """Simplified FDT-like backbone using torchvision."""
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, ...)  # 1-channel input
        backbone.fc = nn.Identity()
        return backbone
```

⚠️ **Hệ quả:** Hiện tại 2/3 models không phải architecture gốc của paper → không thể so sánh công bằng với reported numbers.

### 3.4. Third-Party Repos — ✅ Đã Clone

```
src/third_party/
├── FDT/        (TIFS 2025, 8 ⭐)   — Có Dynamic Transformer gốc trong FDT/model/
├── GLA-FD/     (IEEE Access 2025)  — Có ViT gốc trong models/model.py
└── ILCNN/      (Bioengineering 2023, 9 ⭐) — Có ILCNN gốc trong models/model.py
```

⚠️ **Chưa import được vào pipeline** do:
- Import path conflicts (third_party/ILCNN/datasets/ chứa `albumentations` requirement)
- Models gốc cần dependency riêng (`ptflops`, `pytorch-model-summary`)

---

## 4. Kết Quả Smoke Test

**File:** `results/fdt/results.json`

```json
{
  "model": "fdt",
  "epochs": 1,
  "batch_size": 8,
  "num_params": 11280280,
  "history": [{
    "epoch": 1,
    "train_loss": 5.8128,
    "val_eer": 0.348,
    "val_hter": 0.28,
    "val_auc": 0.7708,
    "time_sec": 217.0
  }],
  "final_test": {
    "eer": 0.284,
    "hter": 0.268,
    "auc": 0.8059,
    "threshold": 1.0,
    "num_pairs": 500
  }
}
```

### So sánh với paper claims:

| Metric | Paper claim | Hiện tại | Gap |
|--------|-------------|----------|-----|
| EER (FV-USM) | 0.087% (FV-DMHN), 0.21% (FVCT) | **28.4%** | ~300x worse |
| AUC | ~0.999 | **0.806** | OK as baseline |

### Nguyên nhân EER cao:
1. **Chỉ 1 epoch** — model chưa hội tụ
2. **ResNet18 thay vì Dynamic Transformer** — backbone quá đơn giản
3. **CrossEntropyLoss thay vì ArcFace** — không tối ưu cho verification
4. **Batch size = 8** — quá nhỏ
5. **Image size 112×112** — paper dùng ~300×300
6. **Không pretrain trên FingerVeinSyn-5M** — mất 53% gain từ synthetic

---

## 5. Vấn Đề Môi Trường

| Vấn đề | Nguyên nhân | Ảnh hưởng | Trạng thái |
|---------|-------------|-----------|------------|
| **CUDA không khả dụng** | NVIDIA driver issue (`Can't initialize NVML`) | Train chậm (~3-4 it/s) | ⚠️ Em vừa fix |
| **Network chặn pytorch.org** | Firewall/proxy | PyTorch wheel phải tải từ PyPI mirror | ✅ Đã bypass |
| **PEP 668 (externally-managed)** | Ubuntu 24.04 | Phải dùng `--break-system-packages` | ✅ Đã bypass |
| **Disk usage 88%** | 59G/71G | Còn 8.4G trống | ⚠️ Cần dọn |

### Hardware em đang có:
- **CPU:** Intel i5-13420H (12 threads)
- **GPU:** RTX 3050 Mobile 4GB (sau khi fix driver)
- **RAM:** 15GB

---

## 6. So Sánh Với Mục Tiêu Paper-Quality

| Yếu tố | Paper yêu cầu | Hiện tại | Đạt? |
|---------|---------------|----------|------|
| Models dùng architecture gốc | FDT + GLA-FD + ILCNN đều dùng gốc | 2/3 dùng fallback (ResNet18) | ❌ |
| Pretrain trên synthetic | 5M images FingerVeinSyn-5M | 0 (chỉ 100 sample không pretrain) | ❌ |
| Cross-session evaluation | Session 1 → Session 2 | Mặc định chỉ session 1 | ❌ |
| ArcFace loss | Margin-based metric learning | CrossEntropyLoss | ❌ |
| 20+ epochs training | Paper train ≥ 40 epochs | 1 epoch (smoke test) | ❌ |
| Full FV-USM dataset | 11,808 ảnh | ~2,952 ảnh (1 session) | ⚠️ |
| Multi-spectral channels | NIR + VIS + ... | Chỉ 1-channel | ❌ |

---

## 7. Đánh Giá Mức Độ Hoàn Thành

### ✅ Đã làm được (Foundation)
1. ✅ Environment setup (Python + PyTorch + packages)
2. ✅ Clone 3 public repos thành công
3. ✅ Unified dataset loader (FV-USM + FingerVeinSyn-5M)
4. ✅ Unified training pipeline (model factory + train loop + eval)
5. ✅ Metrics computation (EER, HTER, AUC)
6. ✅ Smoke test chạy được end-to-end

### ⚠️ Cần làm tiếp (Để đạt paper-quality)
1. ⚠️ Import architecture gốc của FDT (Dynamic Transformer)
2. ⚠️ Import architecture gốc của GLA-FD (ViT)
3. ⚠️ Switch loss sang ArcFace (metric learning)
4. ⚠️ Pretrain trên FingerVeinSyn-5M (cần download full dataset)
5. ⚠️ Cross-session evaluation (session='all')
6. ⚠️ Full training (20+ epochs)
7. ⚠️ Re-verify GPU sau khi em fix driver

---

## 8. Roadmap Đề Xuất

### **Giai đoạn 1: GPU Verification + Quick Wins** (1-2 ngày)
```
1. Verify GPU: python3 -c "import torch; print(torch.cuda.is_available())"
   → Nếu True: re-train smoke test với batch_size=32, time sẽ giảm 10x
   → Nếu False: debug driver thêm
2. Run smoke test trên cả 3 models (ilcnn, fdt, gla-fd)
3. Benchmark thời gian train 1 epoch trên GPU vs CPU
```

### **Giai đoạn 2: Import Architecture Gốc** (2-3 ngày)
```
1. Fix import path trong train.py để load FDT gốc (không fallback ResNet18)
2. Install missing dependencies: ptflops, pytorch-model-summary
3. Test forward pass của 3 architectures gốc
4. Compare #params: ResNet18 (11M) vs FDT thật (~?)
```

### **Giai đoạn 3: Loss + Training Improvements** (2-3 ngày)
```
1. Switch CrossEntropyLoss → ArcFaceLoss (đã có code, chỉ cần plug in)
2. Set session='all' để dùng cả 2 sessions
3. Tăng img_size 112 → 224
4. Train 20+ epochs, log metrics mỗi epoch
```

### **Giai đoạn 4: Synthetic Pretraining** (3-5 ngày)
```
1. Download full FingerVeinSyn-5M từ Kaggle (~10-20GB)
2. Implement pretraining loop (train trên synthetic, fine-tune trên real)
3. Compare EER: from-scratch vs pretrained
4. Target: EER giảm 50%+ so với baseline (như paper claim)
```

### **Giai đoạn 5: Cross-Domain Experiments** (5-7 ngày)
```
1. Train trên synthetic → test trên FV-USM, MFFV-N
2. Few-shot benchmarks (1-shot, 5-shot)
3. Ablation studies (architecture components, loss, augmentation)
```

---

## 9. Ưu Tiên Ngay Bây Giờ

| Ưu tiên | Task | Lý do |
|---------|------|-------|
| 🥇 | Verify GPU hoạt động | Mở khóa tăng tốc 10x |
| 🥈 | Import architecture gốc | Đây là "model" thật sự |
| 🥉 | Switch ArcFace loss | Cải thiện EER 30-50% |
| 4 | Pretrain trên synthetic | Điểm mạnh của FingerVeinSyn-5M |
| 5 | Cross-session evaluation | Đánh giá đúng paper protocol |

---

## 10. Tổng Kết

| Aspect | Đánh giá |
|--------|----------|
| **Foundation** | 🟢 Vững chắc — code chạy được, có structure rõ ràng |
| **Performance** | 🔴 Chưa đạt paper (EER ~28% vs paper ~0.2%) |
| **Scalability** | 🟡 Trung bình — pipeline có thể extend nhưng cần refactor |
| **Documentation** | 🟢 Tốt — README + JSON logs đầy đủ |
| **Mức độ hoàn thành ước tính** | **~25%** của paper-quality pipeline |

**Khuyến nghị:** Ưu tiên **Giai đoạn 1 + 2** trước khi train full. Đừng waste thời gian train 20 epochs với ResNet18 fallback — sẽ không publish được.

---

*Tạo bởi Cursor AI Assistant — Phân tích ngày 2026-08-09*
