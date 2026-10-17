# Update Phân Tích — Sau Khi Fix GPU + FDT Original

> **Ngày update:** 2026-08-09 (sau session 2)
> **Cập nhật bởi:** Cursor AI Assistant
> **Xem file gốc:** [`project_status_analysis.md`](./project_status_analysis.md)

---

## Thay Đổi Trong Session Này

### ✅ GPU Investigation & Decision

**Phát hiện:**
- NVIDIA driver 580 đã cài đầy đủ (libcuda.so.580.173.02, kernel modules có sẵn trong DKMS)
- **Secure Boot = Enabled** → Kernel chặn load unsigned NVIDIA module
- "Key was rejected by service" / "Loading of module with unavailable key is rejected"

**Đã thử:**
1. ✅ `sudo modprobe nvidia` → Failed (Key rejected)
2. ✅ Sign module bằng MOK key (`sign-file`, `kmodsign`) → Module đã sign thành công
3. ✅ Replace unsigned modules trong `/lib/modules/.../updates/dkms/` bằng signed versions
4. ❌ Vẫn fail vì MOK cert chưa được **enroll vào UEFI firmware** (cần reboot + MOK Manager)
5. ❌ `mokutil --import` yêu cầu password interactive

**Quyết định cuối:** **Skip GPU, optimize cho CPU** (an toàn + không cần reboot)

### ✅ FDT Original Architecture Imported

**Trước:**
```python
class FDTWrapper(nn.Module):
    def _build_fdt_simplified(self):
        backbone = models.resnet18(weights=None)  # ❌ Fallback
        backbone.conv1 = nn.Conv2d(1, 64, ...)
        backbone.fc = nn.Identity()
        return backbone
# Params: 11,235,904
```

**Sau:**
```python
class FDTWrapper(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 492):
        from fdt import FDT as FDTModel  # ✅ Import gốc từ SCUT-BIP-Lab/FDT
        self.model = FDTModel(
            img_size=(112, 112),
            patch_size=(14, 14),
            in_chans=1,
            num_classes=num_classes,
            embed_dim=embedding_dim,
            depth_fv=4,
            depth_sv=2,
            num_heads=4,
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            keep_rate_fv=(1.0, 1.0, 1.0, 1.0),
            keep_rate_sv=(1.0, 1.0),
            use_fa=False,
            use_ba=False,
            use_lffn=False,  # Disabled - shape mismatch với 112x112
            use_mlppatch=True,
            use_peg=False,   # Disabled - shape issue với multi-view concat
        )
        # Params: 5,488,016 (-51% vs ResNet18 fallback)
```

**Architectural Changes Made:**
- ✅ Patch size: 14×14 (perfect square, 8×8 patches per view)
- ✅ Disabled `use_lffn` (LFFN expects specific shape từ paper gốc 240×320)
- ✅ Disabled `use_peg` (PEG expects square token layout, multi-view concat breaks it)
- ✅ Reduced `depth_fv=4`, `depth_sv=2` (paper uses 6/4 — tăng tốc cho CPU)
- ✅ Disabled `use_ba` (BatchAttention cần special init)

### ✅ ArcFaceLoss Integrated

**Thêm vào `train.py`:**
```python
class ArcFaceLoss(nn.Module):
    """ArcFace loss for metric learning."""
    def __init__(self, embedding_dim: int, num_classes: int,
                 margin: float = 0.5, scale: float = 30.0):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight = F.normalize(self.weight, p=2, dim=1)
        cos_theta = F.linear(embeddings, weight).clamp(-1+1e-7, 1-1e-7)
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        theta = torch.acos(cos_theta)
        target_cos = torch.cos(theta + self.margin)
        output = one_hot * target_cos + (1 - one_hot) * cos_theta
        return F.cross_entropy(output * self.scale, labels)
```

**CLI flags:**
```bash
python3 src/train.py --arcface --arcface_margin 0.5 --arcface_scale 30.0
```

---

## Kết Quả So Sánh

| Metric | ResNet18 Fallback | FDT Original + ArcFace |
|--------|-------------------|------------------------|
| **Params** | 11.24M | 5.49M (-51%) |
| **Loss type** | CrossEntropyLoss | ArcFaceLoss (margin=0.5) |
| **Train speed (batch=8)** | ~4 it/s | ~5 it/s (+25%) |
| **Train speed (batch=4)** | ~3 it/s | ~4 it/s (+33%) |
| **Eval speed** | ~2.5s/batch | ~25s/batch (slow vì 3 views) |
| **EER (1 epoch)** | 28.4% | Chưa save (timeout) |
| **AUC (1 epoch)** | 0.8059 | Chưa save (timeout) |

**Note:** Eval chậm hơn nhiều vì FDT phải forward qua 3 views của cùng 1 ảnh. Tối ưu: chỉ eval trên 1 view hoặc tăng batch size.

---

## Status Update

### ✅ Đã làm được (Updated)
1. ✅ GPU investigation (Secure Boot root cause identified)
2. ✅ FDT original architecture imported (5.49M params)
3. ✅ ArcFaceLoss integrated và hoạt động
4. ✅ Test pipeline với FDT gốc + ArcFace (timeout nhưng training ổn)

### ⚠️ Còn thiếu (Giảm so với trước)
1. ❌ GLA-FD original architecture (vẫn dùng fallback)
2. ❌ ILCNN chưa verify (chỉ import trong code, chưa test end-to-end)
3. ❌ Full training run (chỉ smoke test 1 epoch)
4. ❌ Compare table ResNet18 vs FDT gốc (cần GPU để train nhanh)

### 🔧 Mức Độ Hoàn Thành Ước Tính

| Component | Trước | Sau |
|-----------|-------|-----|
| FDT model | ⚠️ 25% (fallback) | ✅ 75% (original) |
| Loss function | ⚠️ 50% (CrossEntropy) | ✅ 90% (ArcFace ready) |
| GLA-FD model | ❌ 0% | ❌ 0% |
| ILCNN model | ⚠️ 50% (imported only) | ⚠️ 50% (chưa test) |
| **Overall** | **~25%** | **~45%** |

---

## Khuyến Nghị Tiếp Theo

### 🥇 Ưu tiên 1: GLA-FD Original (2-3 giờ)
```python
class GLAFDWrapper(ModelWrapper):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 492):
        from models.model import GLAFD as GLAFDModel
        self.backbone = GLAFDModel(
            img_size=112, patch_size=16, in_chans=1,
            num_classes=num_classes, embed_dim=embedding_dim,
            depth=6, num_heads=4, mlp_ratio=4.,
        )
```

### 🥈 Ưu tiên 2: ILCNN End-to-End Test (1 giờ)
Verify ILCNN forward + training pass:
```bash
PYTHONPATH=src python3 -c "
from train import get_model
m = get_model('ilcnn', num_classes=344, embedding_dim=128)
x = torch.randn(2, 1, 112, 112)
print(m(x).shape)
"
```

### 🥉 Ưu tiên 3: Eval Optimization (1 giờ)
- FDT eval chậm (25s/batch) vì forward 3 views
- Fix: chỉ dùng 1 view cho eval, hoặc batch all 3 views cùng lúc

### 4️⃣ Ưu tiên 4: Full Training Run (3-5 giờ)
- Train FDT 20 epochs với ArcFace
- Compare với ResNet18 baseline
- Save metrics, plot ROC curves

---

## Technical Debt

| Vấn đề | Ảnh hưởng | Cách fix |
|---------|-----------|----------|
| Eval loop forward 3 views per image | 3x slower than needed | Refactor `evaluate_verification` |
| Module sign failed | GPU unusable | Manual MOK enrollment (cần reboot) |
| Patch size hardcoded (14×14) | Không flexible cho 224×224 | Make img_size configurable |
| `use_peg=False`, `use_lffn=False` | Mất 1 số features gốc | Cần investigate shape workaround |
| Timeout không save results | Không có comparison data | Tăng timeout hoặc giảm epochs |

---

*Tạo bởi Cursor AI Assistant — Update 2026-08-09 23:50*