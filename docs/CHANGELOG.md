# Changelog - MFFV Project

> Lich su phat trien du an Finger Vein Recognition tren FV-USM Dataset.
>
> Gop tu 3 file cu: project_status_analysis.md (Aug 2026),
> project_status_update_session2.md, project_status_update_session3.md.
>
> Ban luu tru day du tai archive/2026-09-18_pre_paper_cleanup/old_project_docs/.

---

## Trang thai hien tai (Sep 2026)

| Aspect | Status |
|--------|--------|
| Best model | ResNet-18 + ArcFace (fvusm_resnet18_arcface_v1) |
| Best EER | 0.108% tren FV-USM session 1 |
| Best AUC | 0.9496 |
| Comparison charts | 6 bieu do tai src/results/comparison_charts/ |
| Demo app | Streamlit (src/e2e_app.py) |
| Muc tieu | Viet paper IEEE Conference (Comparative Study) |

---

## Timeline

### 2026-08-09 - Initial Analysis

Context:
- Repository: /home/qa/Workspace/Research/MFFV/
- Stack: Python 3.12 + PyTorch 2.13 (CPU mode)
- Muc tieu: Benchmark 3 baselines cho finger vein recognition tren FV-USM

Foundation da co:
- Environment setup (Python + PyTorch + packages)
- Clone 3 public repos (FDT, GLA-FD, ILCNN)
- Unified dataset loader (FV-USM + FingerVeinSyn-5M)
- Unified training pipeline (model factory + train loop + eval)
- Metrics computation (EER, HTER, AUC)
- Smoke test chay duoc end-to-end (1 epoch)

Van de:
- EER = 28.4% (paper claim ~0.2%, gap ~300x)
- FDT/GLA-FD/ILCNN chua dung architecture goc (2/3 fallback ResNet18)
- Chi smoke test 1 epoch
- Chi session 1 cua FV-USM
- CrossEntropyLoss thay vi ArcFace

Muc hoan thanh uoc tinh: ~25%

---

### 2026-08-09 - Session 2: GPU + FDT Original + ArcFace

Major changes:

1. GPU investigation (Secure Boot root cause)
   - NVIDIA driver 580 OK nhung Secure Boot = Enabled
   - Module signing that bai do MOK cert chua enroll
   - Quyet dinh: Skip GPU, optimize cho CPU

2. FDT Original architecture imported
   - Params: 5.49M (-51% so voi ResNet18 fallback)

3. ArcFaceLoss integrated
   - Margin m = 0.5, Scale s = 30.0
   - CLI flags: --arcface --arcface_margin 0.5 --arcface_scale 30.0

Ket qua so sanh:

| Metric | ResNet18 Fallback | FDT Original + ArcFace |
|--------|-------------------|------------------------|
| Params | 11.24M | 5.49M (-51%) |
| Loss | CrossEntropyLoss | ArcFaceLoss |
| Train speed (batch=8) | ~4 it/s | ~5 it/s (+25%) |
| Eval speed | ~2.5s/batch | ~25s/batch (3 views) |

Muc hoan thanh uoc tinh: ~45%

---

### 2026-08-31 - Session 3: End-to-End Pipeline

Fixes:

1. Model Input Channels - ILCNN/GLA-FD expected RGB nhung input grayscale
   - Them in_channels=1 parameter

2. Windows Path Compatibility - paths hardcoded Linux -> tu dong detect

3. Python Import Order - third_party/ILCNN/datasets/ conflict voi src/datasets.py
   - Fix bang sys.path.insert(0, src) + importlib

4. Model Wrappers Refactored
   - ILCNNWrapper, GLAFDWrapper inherit truc tiep tu nn.Module

Test results:
- ILCNN: [batch, 512] (fallback ResNet18)
- GLA-FD: [batch, 512] (fallback ResNet18)
- FDT: [batch, 128] (original architecture)

Ket qua: Models hoat dong 3/3, Windows compatible.

---

### 2026-09-18 - Phase dot pha: ResNet-18 + ArcFace (Current)

Key insight: Thay vi dung FDT/GLA-FD/ILCNN phuc tap, train ResNet-18 don gian voi ArcFace loss -> ket qua vuot troi.

Final results (FV-USM, session 1):

| Model | Params | Epochs | EER | HTER | AUC |
|-------|--------|--------|-----|------|-----|
| ResNet-18 + ArcFace (v1) | 11.3M | 50 | 0.108% | 0.11% | 0.9496 |
| ResNet-18 + ArcFace (expD) | 11.3M | 50 | 1.04% | 1.00% | 0.9990 |
| ResNet-18 + ArcFace (30ep) | 11.3M | 30 | 2.00% | 2.00% | 0.9978 |
| GLA-FD | 7.6M | 2 | 4.40% | 4.40% | 0.9893 |
| ILCNN | 3.0M | 10 | 10.40% | 11.40% | 0.9544 |
| FDT | 5.5M | 30 | 17.60% | 17.60% | 0.9006 |

Deliverables:
- src/train_fvusm.py - Training pipeline
- src/eval_fvusm.py - Evaluation
- src/e2e_app.py - Streamlit demo app
- src/results/comparison_charts/ - 6 bieu do PNG
- src/results/fvusm_resnet18_arcface_v1/ - Best model checkpoint

Next: Viet paper IEEE Conference.

---

## Cleanup (2026-09-19)

Loai bo:
- 11 file debug rac (_*.py, _*.txt)
- 13 file log cu (giu lai log cua best model)
- 5 thu muc smoke test (fdt_5epochs, fdt_30ep, fdt_30epochs_aug, resnet18_quick_infer, comparison)
- 13 src scripts cu (train.py cu, baseline_resnet.py, dashboard.py, ...)
- 3 file project_status_* (gop vao file nay)

File archive: archive/2026-09-18_pre_paper_cleanup/
- old_debug_scripts/ - 11 files
- old_results_logs/ - 13 files
- old_smoke_results/ - 5 dirs (8 files, ~87 MB)
- old_project_docs/ - 3 files
- old_src_scripts/ - 13 files

Tong: 48 files, ~92.6 MB

---

## Outstanding Tasks (cho paper)

- [ ] Phase 2: Tao paper/ folder voi LaTeX template IEEE Conference
- [ ] Phase 3: Convert 6 PNG -> PDF vector
- [ ] Phase 4: Viet EXPERIMENT_NOTES.md (hyper-params de reproduce)
- [ ] Phase 5: Vietnamese draft + English final
- [ ] Phase 6: Citation tu literature_review_finger_vein_2025.md

---

*File duoc tao ngay 2026-09-19 boi cleanup script. Ban archive tai archive/2026-09-18_pre_paper_cleanup/old_project_docs/.*
