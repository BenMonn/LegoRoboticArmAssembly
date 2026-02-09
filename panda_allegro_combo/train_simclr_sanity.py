import os
import csv
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import torchvision.transforms as T
import torchvision.models as models

class PairDataset(Dataset):
    def __init__(self, root_dir, csv_path, transform=None):
        self.root = Path(root_dir)
        self.transform = transform
        self.items = []

        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                a = self.root / row["img_a"]
                b = self.root / row["img_b"]
                self.items.append((a, b))

        print("[dataset] pairs:", len(self.items))
        for k in range(min(5, len(self.items))):
            assert self.items[k][0].exists(), f"Missing {self.items[k][0]}"
            assert self.items[k][1].exists(), f"Missing {self.items[k][1]}"

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        a_path, b_path = self.items[idx]
        img_a = Image.open(a_path).convert("RGB")
        img_b = Image.open(b_path).convert("RGB")

        if self.transform is not None:
            img_a = self.transform(img_a)
            img_b = self.transform(img_b)
        return img_a, img_b


class SimCLRModel(nn.Module):
    def __init__(self, proj_dim=128):
        super().__init__()
        if models is None:
            raise RuntimeError("torchvision not available; install torchvision for this script.")

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

def nt_xent_loss(z1, z2, tau=0.2):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)  # [2B, D]

    # cosine similarity (since z normalized, dot = cosine)
    sim = torch.mm(z, z.t()) / tau  # [2B, 2B]

    # mask out self-similarity
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -1e9)

    # positive pairs: i <-> i+B
    pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)  # [2B]

    # denominator: logsumexp over all negatives + the positive is included naturally
    loss = -pos + torch.logsumexp(sim, dim=1)
    return loss.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="simclr_pairs", help="Root folder containing pairs.csv and images/")
    parser.add_argument("--csv", type=str, default="simclr_pairs/pairs.csv", help="Path to pairs.csv")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.2)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--overfit", action="store_true", help="Overfit a tiny subset (sanity check)")
    parser.add_argument("--subset", type=int, default=128, help="Number of pairs for overfit subset")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[device]", device)

    # Simple train-time transform (you already applied augmentation during collection)
    # For a real SimCLR run, you'd apply stronger transforms here, not during collection.
    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
    ])

    ds = PairDataset(args.root, args.csv, transform=transform)

    if args.overfit:
        ds.items = ds.items[:args.subset]
        print(f"[overfit] using {len(ds)} pairs")

    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    model = SimCLRModel(proj_dim=128).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    step = 0
    model.train()
    for epoch in range(args.epochs):
        for x1, x2 in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            # Basic finite checks on inputs
            assert torch.isfinite(x1).all(), "x1 has NaN/Inf"
            assert torch.isfinite(x2).all(), "x2 has NaN/Inf"

            z1 = model(x1)
            z2 = model(x2)

            # Embedding finite checks
            assert torch.isfinite(z1).all(), "z1 has NaN/Inf"
            assert torch.isfinite(z2).all(), "z2 has NaN/Inf"

            loss = nt_xent_loss(z1, z2, tau=args.tau)
            assert torch.isfinite(loss), "loss is NaN/Inf"

            opt.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient finite check
            for name, p in model.named_parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise RuntimeError(f"Non-finite grad in {name}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            if step % 10 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
            step += 1

        # End-of-epoch checkpoint (optional)
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/simclr_sanity_e{epoch}.pt")

    print("Done.")


if __name__ == "__main__":
    main()
