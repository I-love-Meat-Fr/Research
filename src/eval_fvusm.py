"""Evaluate saved best.pth on FV-USM test set (session 2)."""
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image


class FVUSMDataset(Dataset):
    def __init__(self, root, session="2nd_session", target_size=(128, 128)):
        self.root = os.path.join(root, session, "extractedvein")
        self.target_size = target_size
        self.samples = []
        label_map = {}
        for folder in sorted(os.listdir(self.root)):
            folder_path = os.path.join(self.root, folder)
            if not os.path.isdir(folder_path):
                continue
            if folder not in label_map:
                label_map[folder] = len(label_map)
            label = label_map[folder]
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".bmp", ".png", ".jpg", ".tif", ".tiff", ".jpeg")):
                    self.samples.append((os.path.join(folder_path, fname), label))
        self.num_classes = len(label_map)
        print(f"[{session}] {len(self.samples)} images, {self.num_classes} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L").resize(self.target_size)
        x = np.array(img, dtype=np.float32) / 255.0
        x = (x - 0.5) / 0.5
        return torch.from_numpy(x).unsqueeze(0).float(), label


class ResNet18Backbone(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        import torchvision.models as tvm

        m = tvm.resnet18(weights=None)
        m.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        m.fc = nn.Identity()
        self.backbone = nn.Sequential(*list(m.children())[:-1])
        self.embed = nn.Linear(512, embedding_dim)
        self.bn = nn.BatchNorm1d(embedding_dim)

    def forward(self, x):
        f = self.backbone(x).flatten(1)
        return self.bn(self.embed(f))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = r"D:\WorkSpace\Research\Research\MFFV\src\results\fvusm_resnet18_arcface_v1\best.pth"
    data_root = r"D:\WorkSpace\Research\Research\MFFV\data\external\Published_database_FV-USM_Dec2013"

    print(f"[INFO] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    print(f"[INFO] Checkpoint epoch: {ckpt.get('epoch', '?')}, EER: {ckpt.get('eer', '?')}")

    backbone = ResNet18Backbone(embedding_dim=ckpt.get("embedding_dim", 128)).to(device)
    backbone.load_state_dict(ckpt["backbone"])
    backbone.eval()

    test_ds = FVUSMDataset(data_root, "2nd_session")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

    embs_per_class = {}
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            e = nn.functional.normalize(backbone(x), dim=1)
            for i, label in enumerate(y.tolist()):
                embs_per_class.setdefault(label, []).append(e[i].cpu())

    classes = list(embs_per_class.keys())
    print(f"[INFO] Test classes: {len(classes)}")
    gallery = {c: torch.stack(embs_per_class[c]).mean(0) for c in classes}

    pos, neg = [], []
    for c in classes:
        for e in embs_per_class[c]:
            pos.append(torch.nn.functional.cosine_similarity(e.view(1, -1), gallery[c].view(1, -1)).item())
    for i, c1 in enumerate(classes):
        for c2 in classes[i + 1 :]:
            neg.append(torch.nn.functional.cosine_similarity(gallery[c1].view(1, -1), gallery[c2].view(1, -1)).item())

    pos, neg = np.array(pos), np.array(neg)
    thresholds = np.linspace(-1, 1, 200)
    far = np.array([(neg >= t).mean() for t in thresholds])
    frr = np.array([(pos < t).mean() for t in thresholds])
    idx = np.argmin(np.abs(far - frr))
    eer = (far[idx] + frr[idx]) / 2

    results = {
        "eer_percent": float(eer * 100),
        "pos_mean": float(pos.mean()),
        "neg_mean": float(neg.mean()),
        "margin": float(pos.mean() - neg.mean()),
        "pos_std": float(pos.std()),
        "neg_std": float(neg.std()),
        "n_classes": len(classes),
        "n_test_images": int(len(pos)),
        "checkpoint_epoch": int(ckpt.get("epoch", 0)) if isinstance(ckpt.get("epoch"), int) else ckpt.get("epoch"),
    }
    out_json = r"D:\WorkSpace\Research\Research\MFFV\src\results\fvusm_resnet18_arcface_v1\eval_results.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 50)
    print(f"EER: {eer*100:.2f}%")
    print(f"Intra-class cosine (pos): {pos.mean():.3f} +/- {pos.std():.3f}")
    print(f"Inter-class cosine (neg): {neg.mean():.3f} +/- {neg.std():.3f}")
    print(f"Margin (pos - neg): {pos.mean() - neg.mean():.3f}")
    print(f"Classes: {len(classes)} | Test images: {len(pos)}")
    print("=" * 50)
    print(f"Results saved to: {out_json}")


if __name__ == "__main__":
    main()
