#!/usr/bin/env python3
import os
import csv
import json
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T
import torchvision.models as models

# Model (must match your SimCLR training)
class SimCLRModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x):
        h = self.encoder(x)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        return z


# Dataset for probing
class ProbeDataset(Dataset):
    def __init__(self, root_dir: str, csv_path: str, transform=None, use_column="img_a"):
        self.root = Path(root_dir)
        self.transform = transform
        self.items = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            if "mode" not in reader.fieldnames:
                raise ValueError("pairs.csv must include a 'mode' column for linear probing.")
            if use_column not in reader.fieldnames:
                raise ValueError(f"pairs.csv missing column '{use_column}'.")

            for row in reader:
                img_rel = row[use_column]
                mode = row["mode"]
                mode = mode.strip().lower()
                allowed = {"stacked", "side_by_side", "near_miss"}
                if mode not in allowed:
                    continue  # or raise error to catch it early
                img_path = self.root / img_rel
                if img_path.exists():
                    self.items.append((img_path, mode))

        if not self.items:
            raise RuntimeError("No samples found. Check root_dir/csv_path and images exist.")

        # Build label mapping
        modes = [m for _, m in self.items]
        self.mode_counts = Counter(modes)
        self.classes = sorted(list(self.mode_counts.keys()))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        print("[probe dataset] samples:", len(self.items))
        print("[probe dataset] class counts:", dict(self.mode_counts))
        print("[probe dataset] classes:", self.classes)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        img_path, mode = self.items[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        y = self.class_to_idx[mode]
        return img, y


# Confusion matrix and accuracy
def confusion_matrix(num_classes, y_true, y_pred):
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def accuracy(y_true, y_pred):
    return (y_true == y_pred).float().mean().item()


# Main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, required=True, help="Dataset root (contains images/ and pairs.csv)")
    parser.add_argument("--csv", type=str, required=True, help="Path to labeled pairs.csv (must have mode column)")
    parser.add_argument("--ckpt", type=str, required=True, help="SimCLR checkpoint .pt (e.g., simclr_sanity_e19.pt)")
    parser.add_argument("--out", type=str, default="probe_out", help="Output folder to save probe + results")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Transform: keep it simple and consistent
    tfm = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
    ])

    ds = ProbeDataset(args.root, args.csv, transform=tfm, use_column="img_a")
    num_classes = len(ds.classes)

    # Split train/val
    n_total = len(ds)
    n_val = max(1, int(round(args.val_frac * n_total)))
    n_train = n_total - n_val
    if n_train < 1:
        raise RuntimeError(f"Not enough data after split: n_total={n_total}, n_val={n_val}")
    
    print(f"[split] total={n_total} train={n_train} val={n_val}")

    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    # Load SimCLR model and freeze encoder
    simclr = SimCLRModel(proj_dim=128).to(device)
    simclr.load_state_dict(torch.load(args.ckpt, map_location=device))
    simclr.eval()

    for p in simclr.parameters():
        p.requires_grad = False

    # Linear probe on top of frozen embeddings z (dim=128)
    probe = nn.Linear(512, num_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr)

    best_val = float("-inf")
    best_path = out_dir / "probe_best.pt"

    for epoch in range(args.epochs):
        # train
        probe.train()
        train_true = []
        train_pred = []

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.no_grad():
                h = simclr.encoder(x)
                h = F.normalize(h, dim=1)

            logits = probe(h)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            pred = torch.argmax(logits, dim=1)
            train_true.append(y.detach().cpu())
            train_pred.append(pred.detach().cpu())
            counts = torch.bincount(pred, minlength=num_classes).cpu().numpy()
        print("pred_counts:", counts)

        train_true = torch.cat(train_true)
        train_pred = torch.cat(train_pred)
        train_acc = accuracy(train_true, train_pred)

        # val
        probe.eval()
        val_true = []
        val_pred = []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                h = simclr.encoder(x)
                h = F.normalize(h, dim=1)
                logits = probe(h)
                pred = torch.argmax(logits, dim=1)

                val_true.append(y.detach().cpu())
                val_pred.append(pred.detach().cpu())

        val_true = torch.cat(val_true)
        val_pred = torch.cat(val_pred)
        val_acc = accuracy(val_true, val_pred)

        print(f"epoch {epoch:02d}  train_acc {train_acc:.3f}  val_acc {val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save(probe.state_dict(), best_path)

    # Load best and compute confusion matrix
    if not best_path.exists():
        print("[warning] probe_best.pt not found, saving last probe instead")
        torch.save(probe.state_dict(), best_path)

    probe.load_state_dict(torch.load(best_path, map_location=device))
    probe.eval()

    all_true = []
    all_pred = []

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            h = simclr.encoder(x)
            h = F.normalize(h, dim=1)
            logits = probe(h)
            pred = torch.argmax(logits, dim=1)
            all_true.append(y.detach().cpu())
            all_pred.append(pred.detach().cpu())

    all_true = torch.cat(all_true)
    all_pred = torch.cat(all_pred)

    cm = confusion_matrix(num_classes, all_true, all_pred)
    cm_np = cm.numpy()

    print("\n[best val acc]", best_val)
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(cm_np)

    # Save results
    results = {
        "best_val_acc": float(best_val),
        "classes": ds.classes,
        "class_counts": dict(ds.mode_counts),
        "confusion_matrix": cm_np.tolist(),
        "ckpt": str(args.ckpt),
        "root": str(args.root),
        "csv": str(args.csv),
    }
    with open(out_dir / "probe_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save class mapping
    with open(out_dir / "class_to_idx.json", "w") as f:
        json.dump(ds.class_to_idx, f, indent=2)

    print(f"\nSaved: {best_path}")
    print(f"Saved: {out_dir / 'probe_results.json'}")
    print(f"Saved: {out_dir / 'class_to_idx.json'}")


if __name__ == "__main__":
    main()
