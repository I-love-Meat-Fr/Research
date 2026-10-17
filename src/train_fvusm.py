"""Train ResNet18+ArcFace on FV-USM (real finger-vein)."""
import os, argparse, time, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
from scipy.ndimage import rotate, shift

class FVUSMDataset(Dataset):
    def __init__(self, root, session="1st_session", augment=False, target_size=(128, 128)):
        self.root = os.path.join(root, session, "extractedvein")
        self.augment = augment
        self.target_size = target_size
        self.samples = []
        label_map = {}
        for folder in sorted(os.listdir(self.root)):
            folder_path = os.path.join(self.root, folder)
            if not os.path.isdir(folder_path): continue
            if folder not in label_map:
                label_map[folder] = len(label_map)
            label = label_map[folder]
            for fname in sorted(os.listdir(folder_path)):
                if fname.lower().endswith((".bmp",".png",".jpg",".tif",".tiff",".jpeg")):
                    self.samples.append((os.path.join(folder_path, fname), label))
        self.num_classes = len(label_map)
        print(f"[{session}] {len(self.samples)} images, {self.num_classes} classes")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L").resize(self.target_size)
        x = np.array(img, dtype=np.float32) / 255.0
        x = (x - 0.5) / 0.5
        if self.augment:
            if np.random.rand() < 0.5: x = x[:, ::-1].copy()
            x = rotate(x, np.random.uniform(-15,15), reshape=False, order=1, mode="constant", cval=0)
            x = shift(x, (np.random.randint(-10,10), np.random.randint(-10,10)), order=1, mode="constant", cval=0)
            x = np.clip(x * np.random.uniform(0.85,1.15), -1, 1)
            x = x + np.random.normal(0, 0.02, x.shape).astype(np.float32)
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

class ArcFaceHead(nn.Module):
    def __init__(self, in_features, num_classes, s=30.0, m=0.5):
        super().__init__()
        self.s, self.m = s, m
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.xavier_uniform_(self.weight)
    def forward(self, emb, labels=None):
        cosine = nn.functional.linear(nn.functional.normalize(emb), nn.functional.normalize(self.weight))
        cosine = torch.clamp(cosine, -1+1e-7, 1-1e-7)
        if labels is not None:
            theta = torch.acos(cosine)
            target = torch.cos(theta + self.m)
            one_hot = torch.zeros_like(cosine)
            one_hot.scatter_(1, labels.view(-1,1).long(), 1)
            return self.s * (one_hot * target + (1-one_hot) * cosine)
        return self.s * cosine

@torch.no_grad()
def compute_eer(model, loader, device):
    model.eval()
    embs_per_class = {}
    for x, y in loader:
        x = x.to(device)
        e = nn.functional.normalize(model(x), dim=1)
        for i, label in enumerate(y.tolist()):
            embs_per_class.setdefault(label, []).append(e[i].cpu())
    classes = list(embs_per_class.keys())
    gallery = {c: torch.stack(embs_per_class[c]).mean(0) for c in classes}
    pos, neg = [], []
    for c in classes:
        for e in embs_per_class[c]:
            pos.append(torch.nn.functional.cosine_similarity(e.view(1,-1), gallery[c].view(1,-1)).item())
    for i, c1 in enumerate(classes):
        for c2 in classes[i+1:]:
            neg.append(torch.nn.functional.cosine_similarity(gallery[c1].view(1,-1), gallery[c2].view(1,-1)).item())
    pos, neg = np.array(pos), np.array(neg)
    thresholds = np.linspace(-1, 1, 200)
    far = np.array([(neg >= t).mean() for t in thresholds])
    frr = np.array([(pos <  t).mean() for t in thresholds])
    idx = np.argmin(np.abs(far - frr))
    eer = (far[idx] + frr[idx]) / 2
    return {"eer": float(eer), "pos_mean": float(pos.mean()), "neg_mean": float(neg.mean()),
            "pos_std": float(pos.std()), "neg_std": float(neg.std()), "n_classes": len(classes)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=r"D:\WorkSpace\Research\Research\MFFV\data\external\Published_database_FV-USM_Dec2013")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--output-dir", default=r"D:\WorkSpace\Research\Research\MFFV\src\results\fvusm_resnet18_arcface_v1")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")
    if device == "cuda":
        print(f"[INFO] GPU={torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    train_ds = FVUSMDataset(args.data_root, "1st_session", augment=True)
    test_ds = FVUSMDataset(args.data_root, "2nd_session", augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    num_classes = train_ds.num_classes
    print(f"[INFO] num_classes={num_classes}")

    backbone = ResNet18Backbone(args.embedding_dim).to(device)
    head = ArcFaceHead(args.embedding_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(list(backbone.parameters()) + list(head.parameters()),
                                 lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler(enabled=(device == "cuda"))

    best_eer = 1.0
    log = []

    for epoch in range(args.epochs):
        backbone.train(); head.train()
        t0 = time.time(); total=0; correct=0; loss_sum=0.0
        if epoch < args.warmup_epochs:
            for g in optimizer.param_groups: g["lr"] = args.lr * (epoch+1) / args.warmup_epochs

        for x, y in train_loader:
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast(enabled=(device=="cuda")):
                emb = backbone(x); logits = head(emb, y)
                loss = nn.functional.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            loss_sum += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            total += x.size(0)
        if epoch >= args.warmup_epochs: scheduler.step()

        train_acc = correct/total; train_loss = loss_sum/total
        if (epoch+1) % 5 == 0 or epoch == args.epochs-1:
            metrics = compute_eer(backbone, test_loader, device)
            elapsed = time.time()-t0
            print(f"ep {epoch+1:3d}/{args.epochs} | loss {train_loss:.4f} | acc {train_acc:.3f} | EER {metrics['eer']*100:.2f}% | pos {metrics['pos_mean']:.3f} neg {metrics['neg_mean']:.3f} | {elapsed:.1f}s")
            log.append({"epoch":epoch+1,"loss":train_loss,"acc":train_acc,**metrics,"elapsed_s":elapsed})
            if metrics["eer"] < best_eer:
                best_eer = metrics["eer"]
                torch.save({"backbone":backbone.state_dict(),"head":head.state_dict(),
                            "num_classes":num_classes,"embedding_dim":args.embedding_dim,
                            "eer":best_eer,"pos_mean":metrics["pos_mean"],"neg_mean":metrics["neg_mean"],
                            "epoch":epoch+1}, out/"best.pth")
                print(f"   -> saved best.pth (EER={best_eer*100:.2f}%)")
        else:
            elapsed = time.time()-t0
            print(f"ep {epoch+1:3d}/{args.epochs} | loss {train_loss:.4f} | acc {train_acc:.3f} | {elapsed:.1f}s")

    with open(out/"training_log.json","w") as f: json.dump(log, f, indent=2)
    print(f"=== Training complete. Best EER: {best_eer*100:.2f}% | Checkpoint: {out/'best.pth'} ===")

if __name__ == "__main__":
    main()
