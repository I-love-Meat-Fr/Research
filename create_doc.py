"""
Tạo file Word (.docx) cho dự án MFFV
"""
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# =============================================================================
# TITLE
# =============================================================================
title = doc.add_heading('MFFV: Finger Vein Recognition System', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

subtitle = doc.add_paragraph('Methodology & Pipeline Documentation')
subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
subtitle_run = subtitle.runs[0]
subtitle_run.font.size = Pt(14)
subtitle_run.font.color.rgb = RGBColor(100, 100, 100)

doc.add_paragraph('Author: Research Team | Date: 2026-08-12')
doc.add_paragraph('=' * 80)

# =============================================================================
# 1. METHODOLOGY
# =============================================================================
doc.add_heading('1. Methodology (Phương pháp nghiên cứu)', 1)

doc.add_heading('1.1. Mục tiêu nghiên cứu', 2)
doc.add_paragraph(
    'Xây dựng hệ thống nhận dạng mạch máu ngón tay (finger vein recognition) '
    'tự động sử dụng deep learning, đạt độ chính xác cao với EER < 2%.'
)

doc.add_heading('1.2. Mô hình sử dụng', 2)
table = doc.add_table(rows=8, cols=2)
table.style = 'Light Grid Accent 1'

methodology_data = [
    ('Thành phần', 'Chi tiết'),
    ('Backbone', 'ResNet18 (pretrained trên ImageNet)'),
    ('Input', 'Grayscale 1 channel, 112×112 pixels'),
    ('Embedding dimension', '128-D'),
    ('Loss function', 'ArcFace (margin=0.5, scale=30.0)'),
    ('Optimizer', 'AdamW (LR=1e-3, weight_decay=1e-4)'),
    ('Scheduler', 'CosineAnnealingLR'),
    ('Batch size', '32'),
]
for i, (k, v) in enumerate(methodology_data):
    row = table.rows[i]
    row.cells[0].text = k
    row.cells[1].text = v
    if i == 0:
        for cell in row.cells:
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.bold = True

doc.add_heading('1.3. Dữ liệu', 2)
doc.add_paragraph('• Dataset: FV-USM (Finger Vein USM Database)', style='List Bullet')
doc.add_paragraph('• Số lượng: 492 fingers × 6 ảnh × 2 sessions = 11,808 ảnh', style='List Bullet')
doc.add_paragraph('• Loại ảnh: NIR (Near-Infrared), grayscale', style='List Bullet')
doc.add_paragraph('• Split: 70% train / 15% val / 15% test (theo class)', style='List Bullet')
doc.add_paragraph('• Synthetic data: FingerVeinSyn-5M (5M ảnh, dùng cho pretraining)', style='List Bullet')

doc.add_heading('1.4. Metrics đánh giá', 2)
doc.add_paragraph('• EER (Equal Error Rate) - mục tiêu < 2%', style='List Bullet')
doc.add_paragraph('• HTER (Half Total Error Rate)', style='List Bullet')
doc.add_paragraph('• AUC (Area Under ROC Curve) - mục tiêu > 0.998', style='List Bullet')
doc.add_paragraph('• Classification Accuracy', style='List Bullet')

doc.add_heading('1.5. Phương pháp thực nghiệm', 2)
doc.add_paragraph(
    'Sử dụng ResNet18 pretrained làm baseline, fine-tune trên dataset FV-USM. '
    'So sánh 2 chế độ: (1) Cross-Entropy Loss thông thường, (2) ArcFace Loss. '
    'Đánh giá cross-session: train trên session 1st, test trên session 2nd.'
)

doc.add_paragraph('=' * 80)

# =============================================================================
# 2. PIPELINE
# =============================================================================
doc.add_heading('2. Pipeline Documentation (Luồng xử lý)', 1)

doc.add_paragraph(
    'Pipeline end-to-end từ ảnh gốc đến kết quả verification. '
    'Đường dẫn code: MFFV/src/baseline_resnet.py và MFFV/src/datasets.py'
)

# Step 1
doc.add_heading('Bước 1: Load ảnh gốc', 2)
p = doc.add_paragraph()
p.add_run('Code: ').bold = True
p.add_run('cv2.imread(path, cv2.IMREAD_GRAYSCALE)')
doc.add_paragraph(
    '• Đọc ảnh BMP/JPG từ disk\n'
    '• Chuyển sang grayscale 1 channel\n'
    '• Ảnh NIR chỉ có 1 kênh\n'
    '• Shape output: (H, W) ví dụ (320, 240)'
)

# Step 2
doc.add_heading('Bước 2: Data Augmentation (chỉ khi training)', 2)
p = doc.add_paragraph()
p.add_run('ColorJitter: ').bold = True
p.add_run('brightness=0.2, contrast=0.2')
doc.add_paragraph('Tăng đa dạng ánh sáng, giúp model robust với điều kiện chụp khác nhau.')

p = doc.add_paragraph()
p.add_run('RandomHorizontalFlip: ').bold = True
p.add_run('p=0.5')
doc.add_paragraph('Lật ngang 50% - vein pattern đối xứng, không mất thông tin.')

p = doc.add_paragraph()
p.add_run('RandomRotation: ').bold = True
p.add_run('degrees=10')
doc.add_paragraph('Xoay ±10° - mô phỏng ngón tay đặt lệch khi chụp.')

# Step 3
doc.add_heading('Bước 3: Resize → 112×112', 2)
p = doc.add_paragraph()
p.add_run('Code: ').bold = True
p.add_run('transforms.Resize((112, 112))')
doc.add_paragraph(
    '• Resize ảnh về kích thước chuẩn\n'
    '• ResNet18 yêu cầu input cố định\n'
    '• Dùng bilinear interpolation\n'
    '• Shape output: (112, 112)'
)

# Step 4
doc.add_heading('Bước 4: ToTensor + Normalize', 2)
p = doc.add_paragraph()
p.add_run('Code: ').bold = True
p.add_run('transforms.ToTensor() + Normalize(mean=[0.5], std=[0.5])')
doc.add_paragraph(
    '• ToTensor: PIL Image → PyTorch tensor, scale [0,255] → [0,1]\n'
    '• Shape: (H, W) → (1, H, W)\n'
    '• Normalize: (input - 0.5) / 0.5 → range [-1, 1]\n'
    '• Chuẩn hóa về phân phối chuẩn, giúp model hội tụ nhanh'
)

# Step 5
doc.add_heading('Bước 5: ResNet18 Backbone', 2)
p = doc.add_paragraph()
p.add_run('Code: ').bold = True
p.add_run('self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)')
doc.add_paragraph(
    '• Input: (batch, 1, 112, 112) - ảnh grayscale\n'
    '• Sửa conv1: 3 channels (RGB) → 1 channel (grayscale)\n'
    '• ResNet18 gồm: conv1, bn1, maxpool, layer1-4, avgpool\n'
    '• Output: (batch, 512) - feature vector'
)

# Step 6
doc.add_heading('Bước 6: AvgPool + Flatten', 2)
p = doc.add_paragraph()
p.add_run('Code: ').bold = True
p.add_run('x = self.backbone.avgpool(x); x = torch.flatten(x, 1)')
doc.add_paragraph(
    '• AvgPool: lấy trung bình mỗi feature map → 1 số\n'
    '• Flatten: (B, 512, 1, 1) → (B, 512)\n'
    '• Mỗi ảnh → vector 512 chiều'
)

# Step 7
doc.add_heading('Bước 7: Linear → 128D Embedding', 2)
p = doc.add_paragraph()
p.add_run('Code: ').bold = True
p.add_run('self.embedding = nn.Linear(512, 128)')
doc.add_paragraph(
    '• Project 512D → 128D\n'
    '• Vector đặc trưng compact đại diện cho ngón tay\n'
    '• 128D đủ để phân biệt 492 identities'
)

# Step 8
doc.add_heading('Bước 8: L2 Normalize', 2)
p = doc.add_paragraph()
p.add_run('Code: ').bold = True
p.add_run('emb = F.normalize(emb, p=2, dim=1)')
doc.add_paragraph(
    '• Chuẩn hóa vector về độ dài 1 (unit vector)\n'
    '• Công thức: v / ||v||\n'
    '• Đưa tất cả embeddings về cùng hypersphere\n'
    '• Cosine similarity = dot product khi vectors đã normalize'
)

# Step 9
doc.add_heading('Bước 9a: ArcFace Loss (Training)', 2)
p = doc.add_paragraph()
p.add_run('Code: ').bold = True
p.add_run('ArcFaceLoss(margin=0.5, scale=30.0)')
doc.add_paragraph(
    '• Angular margin: cộng margin vào góc giữa embedding và class center\n'
    '• Cùng identity → gần nhau, khác identity → xa nhau\n'
    '• cos(theta + margin) cho class đúng, cos(theta) cho class khác\n'
    '• scale=30: nhân logits để softmax sắc nét\n'
    '• Output: loss để backpropagation'
)

# Step 10
doc.add_heading('Bước 9b: Cosine Similarity (Testing)', 2)
p = doc.add_paragraph()
p.add_run('Code: ').bold = True
p.add_run('F.cosine_similarity(emb1, emb2, dim=1)')
doc.add_paragraph(
    '• So sánh 2 embeddings (2 ảnh ngón tay)\n'
    '• Công thức: (a · b) / (||a|| ||b||)\n'
    '• Range: [0, 1] do L2 normalize\n'
    '• 1 = giống nhau, 0 = khác nhau\n'
    '• Threshold (VD: 0.42):\n'
    '   - score >= threshold → Same person\n'
    '   - score < threshold → Different person'
)

doc.add_paragraph('=' * 80)

# =============================================================================
# 3. DIAGRAM
# =============================================================================
doc.add_heading('3. Pipeline Diagram', 1)

diagram_text = (
    "Ảnh BMP gốc (320×240)\n"
    "        ↓\n"
    "[cv2.imread → grayscale]\n"
    "        ↓\n"
    "[ColorJitter: brightness ±0.2, contrast ±0.2]\n"
    "        ↓\n"
    "[RandomHorizontalFlip p=0.5]\n"
    "        ↓\n"
    "[RandomRotation ±10°]\n"
    "        ↓\n"
    "[Resize → 112×112]\n"
    "        ↓\n"
    "[ToTensor → [0,1]]\n"
    "        ↓\n"
    "[Normalize(0.5, 0.5) → [-1, 1]]\n"
    "        ↓\n"
    "[ResNet18 backbone] (conv1 modified cho 1 channel)\n"
    "        ↓\n"
    "[AvgPool + Flatten] (512-dim)\n"
    "        ↓\n"
    "[Linear → 128D embedding]\n"
    "        ↓\n"
    "[L2 normalize]\n"
    "        ↓\n"
    "[ArcFace loss] (training)\n"
    "    HOẶC\n"
    "[Cosine similarity] (testing)"
)
p = doc.add_paragraph(diagram_text)
p.runs[0].font.name = 'Courier New'

doc.add_paragraph('=' * 80)

# =============================================================================
# 4. TRAINING VS TESTING
# =============================================================================
doc.add_heading('4. So sánh Training vs Testing', 1)

table = doc.add_table(rows=9, cols=3)
table.style = 'Light Grid Accent 1'

train_test_data = [
    ('Bước', 'Training', 'Testing'),
    ('Augmentation', 'Có (Flip, Rotate, Jitter)', 'Không'),
    ('Resize, Normalize', 'Có', 'Có'),
    ('ResNet18', 'Có', 'Có'),
    ('Embedding 128D', 'Có', 'Có'),
    ('L2 normalize', 'Có', 'Có'),
    ('Loss/Metric', 'ArcFace Loss', 'Cosine Similarity'),
    ('Output', 'Loss (backprop)', 'Score (so sánh)'),
    ('Backward pass', 'Có', 'Không (@torch.no_grad)'),
]
for i, (k, v1, v2) in enumerate(train_test_data):
    row = table.rows[i]
    row.cells[0].text = k
    row.cells[1].text = v1
    row.cells[2].text = v2
    if i == 0:
        for cell in row.cells:
            for para in cell.paragraphs:
                for run in para.runs:
                    run.font.bold = True

doc.add_paragraph('=' * 80)

# =============================================================================
# 5. FUTURE WORK
# =============================================================================
doc.add_heading('5. Hướng phát triển (Future Work)', 1)

doc.add_paragraph('• Pretrain trên FingerVeinSyn-5M rồi fine-tune trên FV-USM', style='List Bullet')
doc.add_paragraph('• So sánh ResNet18 vs ResNet50 vs EfficientNet-B0', style='List Bullet')
doc.add_paragraph('• Thử các loss khác: Triplet Loss, Circle Loss, SubCenter ArcFace', style='List Bullet')
doc.add_paragraph('• Attention mechanism: CBAM, SE-Net để focus vào vein region', style='List Bullet')
doc.add_paragraph('• ROI extraction tự động (loại bỏ background)', style='List Bullet')
doc.add_paragraph('• Data augmentation nâng cao: Mixup, CutMix, RandAugment', style='List Bullet')
doc.add_paragraph('• Knowledge Distillation: model nhỏ hơn cho mobile deployment', style='List Bullet')
doc.add_paragraph('• Test trên các dataset khác: SDUMLA, HKPU, UTFVP', style='List Bullet')

doc.add_paragraph('=' * 80)

# Footer
footer = doc.add_paragraph('© 2026 Research Team. All rights reserved.')
footer.alignment = WD_ALIGN_PARAGRAPH.CENTER

# Save
output_path = '/home/qa/Workspace/Research/MFFV_Methodology_and_Pipeline.docx'
doc.save(output_path)
print(f'✅ Đã tạo file: {output_path}')
