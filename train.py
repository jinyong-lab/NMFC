# Last updated: 2026-04-20 (v4)
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Subset

from dataset import get_train_datasets_both_transforms, DATASET_CLASSES
from backbone import get_backbone
from nmfc import (
    linear_mfa_loss, fisher_ratio, auto_sigma,
    compute_energies, compute_logits, APTController,
)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────
# v4: Fixed temperature as hyperparameter.
# Rationale: Ref 11 SphereFace §4.1 treats margin m as hyperparameter;
# Ref 11 follow-ups ArcFace/CosFace §3.3 use fixed scale s=30~64;
# Ref 12 Prototypical §3 uses implicit fixed scaling.
# No reference paper uses learnable temperature → revert v3's nn.Parameter.
# ─────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, out_dim)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# v4: Val/test use CLEAN transforms (Ref 5 SimCLR §3)
# Split train by index, assign augmented transform to train subset
# and clean transform to val subset.
# ─────────────────────────────────────────────────────────────

def make_loaders(dataset_name, batch_size, num_workers, val_frac=0.2, seed=42):
    train_aug, train_clean, test_ds = get_train_datasets_both_transforms(dataset_name)

    n_total = len(train_aug)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val

    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    train_sub = Subset(train_aug, train_idx)        # augmented
    val_sub = Subset(train_clean, val_idx)          # clean (v4 fix)

    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def evaluate(head, backbone, loader, device, sigma, lam, num_classes, temperature):
    head.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        feats = backbone(images)
        embeddings = head(feats)
        E_pos, E_neg = compute_energies(embeddings, labels, sigma, num_classes)
        logits = compute_logits(E_pos, E_neg, lam, temperature=temperature)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    head.train()
    return correct / total


# ─────────────────────────────────────────────────────────────
# v4: MFA weight schedule (Ref 2 MFA §3 — ratio form has equal weight)
# Start at high alpha (near 1.0) and decay gradually during Phase 2.
# This preserves Phase 1 structure early and lets NMFC dominate late.
# ─────────────────────────────────────────────────────────────

def mfa_weight_schedule(phase2_epoch, phase2_total, alpha_start=1.0, alpha_end=0.1):
    """Linear decay from alpha_start to alpha_end across Phase 2 epochs."""
    if phase2_total <= 1:
        return alpha_end
    t = min(phase2_epoch / (phase2_total - 1), 1.0)
    return alpha_start + t * (alpha_end - alpha_start)


def phase2_loss(embeddings, labels, sigma, lam, num_classes, temperature, mfa_weight):
    E_pos, E_neg = compute_energies(embeddings, labels, sigma, num_classes)
    logits = compute_logits(E_pos, E_neg, lam, temperature=temperature)
    loss_nmfc = F.cross_entropy(logits, labels)
    loss_mfa = linear_mfa_loss(embeddings, labels, num_classes)
    return loss_nmfc + mfa_weight * loss_mfa, logits


def train(config, dataset_name):
    set_seed(config["seed"])

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}  |  Device: {device}  |  v4")
    print(f"{'='*60}")

    train_loader, val_loader, test_loader = make_loaders(
        dataset_name, batch_size=config["batch_size"],
        num_workers=config["num_workers"], seed=config["seed"]
    )

    backbone = get_backbone(device, pretrained=True, freeze=True)
    num_classes = len(DATASET_CLASSES[dataset_name])
    head = ProjectionHead(in_dim=512, out_dim=config["embedding_dim"]).to(device)

    optimizer = optim.Adam(head.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    raw = config["max_phase1_epochs"]
    max_p1 = None if raw == -1 else min(int(raw), config["epochs"])
    apt = APTController(delta=config["apt_delta"], patience=config["apt_patience"],
                        max_phase1_epochs=max_p1)

    sigma = config["sigma"]
    lam = config["lam"]
    tau = config["temperature"]               # v4: fixed
    alpha_start = config["mfa_weight_start"]  # v4: schedule
    alpha_end = config["mfa_weight_end"]
    phase2_total = config["epochs"] - (max_p1 or 0)

    fisher_log = []
    phase2_triggered = False
    phase2_epoch = 0
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, config["epochs"] + 1):
        head.train()
        total_loss, fr_accum, steps = 0.0, 0.0, 0

        # v4: compute current mfa_weight if in Phase 2
        if apt.phase == 2:
            mfa_w = mfa_weight_schedule(phase2_epoch, phase2_total,
                                        alpha_start=alpha_start, alpha_end=alpha_end)
        else:
            mfa_w = None

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            with torch.no_grad():
                feats = backbone(images)
            embeddings = head(feats)

            if config["sigma"] == -1:
                sigma = auto_sigma(embeddings.detach())

            if apt.phase == 1:
                loss = linear_mfa_loss(embeddings, labels, num_classes)
            else:
                loss, _ = phase2_loss(embeddings, labels, sigma, lam, num_classes,
                                      temperature=tau, mfa_weight=mfa_w)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            with torch.no_grad():
                fr_accum += fisher_ratio(embeddings.detach(), labels, num_classes)
            steps += 1

        if apt.phase == 1:
            scheduler.step()

        avg_loss = total_loss / steps
        avg_fr = fr_accum / steps
        fisher_log.append(avg_fr)
        phase = apt.update(avg_fr)

        if apt.phase == 2 and not phase2_triggered:
            phase2_triggered = True
            for g in optimizer.param_groups:
                g["lr"] = config["lr_phase2"]
            print(f"[APT] Learning rate reduced to {config['lr_phase2']}")

        mfa_str = f"mfa_w {mfa_w:.2f}" if mfa_w is not None else "mfa_w -"
        print(f"Epoch {epoch:3d} | Phase {phase} | Loss {avg_loss:.4f} | "
              f"FR {avg_fr:.4f} | tau {tau:.1f} | {mfa_str}")

        if apt.phase == 2:
            phase2_epoch += 1
            val_acc = evaluate(head, backbone, val_loader, device, sigma, lam,
                               num_classes, tau)
            print(f"           >> Val Accuracy: {val_acc*100:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(head.state_dict(), f"best_head_{dataset_name}.pth")
                print(f"           >> New best val! Saved ({val_acc*100:.2f}%)")

    print(f"\nLoading best checkpoint from epoch {best_epoch} "
          f"(val acc {best_val_acc*100:.2f}%)...")
    head.load_state_dict(torch.load(f"best_head_{dataset_name}.pth"))
    test_acc = evaluate(head, backbone, test_loader, device, sigma, lam, num_classes, tau)
    print(f"*** FINAL TEST Accuracy: {test_acc*100:.2f}% ***")

    torch.save(head.state_dict(), f"head_{dataset_name}.pth")
    np.save(f"fisher_log_{dataset_name}.npy", np.array(fisher_log))
    print(f"Training complete. Best val: {best_val_acc*100:.2f}% @ epoch {best_epoch}  |  "
          f"Test: {test_acc*100:.2f}%")
    return test_acc


if __name__ == "__main__":
    config = {
        "seed"              : 42,
        "epochs"            : 50,
        "batch_size"        : 128,
        "num_workers"       : 4,
        "embedding_dim"     : 128,
        "lr"                : 1e-3,
        "lr_phase2"         : 2e-4,
        "sigma"             : -1,
        "lam"               : 0.5,
        "temperature"       : 20.0,     # v4: FIXED (SphereFace/ArcFace convention)
        "mfa_weight_start"  : 1.0,      # v4: MFA schedule (MFA §3 ratio form)
        "mfa_weight_end"    : 0.1,
        "apt_delta"         : 0.05,
        "apt_patience"      : 5,
        "max_phase1_epochs" : 15,
    }

    print("\nSelect dataset to train on:")
    print("  1. CIFAR-10")
    print("  2. Fashion-MNIST")
    print("  3. STL-10")
    print("  4. All three sequentially")

    MENU = {"1": ["cifar10"], "2": ["fashionmnist"],
            "3": ["stl10"], "4": ["cifar10", "fashionmnist", "stl10"]}

    while True:
        choice = input("\nEnter option (1 / 2 / 3 / 4): ").strip()
        if choice in MENU:
            break
        print("  Invalid option.")

    results = {}
    for ds in MENU[choice]:
        results[ds] = train(config, ds)

    print("\n" + "="*60)
    print("v4 SUMMARY (honest test accuracy)")
    print("="*60)
    for ds, acc in results.items():
        print(f"  {ds:15s}: {acc*100:.2f}%")
