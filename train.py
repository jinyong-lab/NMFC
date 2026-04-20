# Last updated: 2026-04-20 (v3)
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split

from dataset import get_loaders, DATASET_CLASSES
from backbone import get_backbone
from nmfc import (
    nmfc_loss, linear_mfa_loss, fisher_ratio,
    auto_sigma, compute_energies, compute_logits, APTController,
)


# ─────────────────────────────────────────────────────────────
# v3 (2026-04-20): Random seed fixing (Issue #11)
# ─────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────────────────────
# Projection head with learnable temperature (Ref 11 SphereFace §3.2,
# Ref 12 Prototypical §3) — Issue #4
# ─────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    def __init__(self, in_dim=512, out_dim=128, init_temp=20.0):
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
        # v3: learnable temperature parameter
        self.log_temperature = nn.Parameter(torch.tensor(np.log(init_temp)))

    @property
    def temperature(self):
        return self.log_temperature.exp()

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(head, backbone, loader, device, sigma, lam, num_classes):
    head.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        feats = backbone(images)
        embeddings = head(feats)
        E_pos, E_neg = compute_energies(embeddings, labels, sigma, num_classes)
        logits = compute_logits(E_pos, E_neg, lam, temperature=head.temperature)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    head.train()
    return correct / total


# ─────────────────────────────────────────────────────────────
# v3 loss: Phase 2 = NMFC + α * MFA (Ref 2 MFA §3) — Issue #5
# ─────────────────────────────────────────────────────────────

def phase2_loss(embeddings, labels, sigma, lam, num_classes, temperature,
                mfa_weight=0.1):
    """
    NMFC cross-entropy + alpha * MFA scatter ratio.

    Rationale (Issue #5): Phase 1 Linear MFA builds good S_w/S_b structure
    but switching to pure NMFC destroys it (FR drops >90%). Keeping a small
    MFA term preserves that structure — the multi-task principle from
    Ref 2 MFA §3 (simultaneous S_w min + S_b max).
    """
    E_pos, E_neg = compute_energies(embeddings, labels, sigma, num_classes)
    logits = compute_logits(E_pos, E_neg, lam, temperature=temperature)
    loss_nmfc = F.cross_entropy(logits, labels)
    loss_mfa = linear_mfa_loss(embeddings, labels, num_classes)
    return loss_nmfc + mfa_weight * loss_mfa, logits


# ─────────────────────────────────────────────────────────────
# v3 train/val split (Issue #7) — NO test leak
# ─────────────────────────────────────────────────────────────

def make_train_val_loaders(dataset_name, batch_size, num_workers, val_frac=0.2, seed=42):
    """
    v3: split train set into 80% train / 20% val.
    Test set remains held out — evaluated only once at the end.
    """
    train_full, test_loader = get_loaders(
        dataset_name, batch_size=batch_size, num_workers=num_workers
    )
    train_ds = train_full.dataset
    n_total = len(train_ds)
    n_val = int(n_total * val_frac)
    n_train = n_total - n_val
    g = torch.Generator().manual_seed(seed)
    train_sub, val_sub = random_split(train_ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_sub, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_sub, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────
# Training loop (v3)
# ─────────────────────────────────────────────────────────────

def train(config, dataset_name):
    set_seed(config["seed"])

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}  |  Device: {device}  |  v3")
    print(f"{'='*60}")

    # v3: train/val/test split
    train_loader, val_loader, test_loader = make_train_val_loaders(
        dataset_name, batch_size=config["batch_size"],
        num_workers=config["num_workers"], seed=config["seed"]
    )

    backbone = get_backbone(device, pretrained=True, freeze=True)

    # v3: num_classes from dataset (Issue #10)
    num_classes = len(DATASET_CLASSES[dataset_name])

    head = ProjectionHead(
        in_dim=512, out_dim=config["embedding_dim"],
        init_temp=config["temperature"]
    ).to(device)

    optimizer = optim.Adam(head.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    raw = config["max_phase1_epochs"]
    max_p1 = None if raw == -1 else min(int(raw), config["epochs"])

    apt = APTController(
        delta=config["apt_delta"], patience=config["apt_patience"],
        max_phase1_epochs=max_p1
    )

    sigma = config["sigma"]
    lam = config["lam"]
    mfa_weight = config["mfa_weight"]
    fisher_log = []
    phase2_triggered = False
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, config["epochs"] + 1):
        head.train()
        total_loss, fr_accum, steps = 0.0, 0.0, 0

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
                # v3: NMFC + alpha*MFA (Issue #5)
                loss, _ = phase2_loss(
                    embeddings, labels, sigma, lam, num_classes,
                    temperature=head.temperature, mfa_weight=mfa_weight
                )

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

        tau = head.temperature.item()
        print(f"Epoch {epoch:3d} | Phase {phase} | Loss {avg_loss:.4f} | "
              f"FR {avg_fr:.4f} | tau {tau:.2f}")

        # v3: evaluate every epoch on VALIDATION (not test!) during Phase 2
        if apt.phase == 2:
            val_acc = evaluate(head, backbone, val_loader, device, sigma, lam, num_classes)
            print(f"           >> Val Accuracy: {val_acc*100:.2f}%")
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch
                torch.save(head.state_dict(), f"best_head_{dataset_name}.pth")
                print(f"           >> New best val! Saved ({val_acc*100:.2f}%)")

    # v3: test evaluation ONCE at the end, using best val checkpoint
    print(f"\nLoading best checkpoint from epoch {best_epoch} "
          f"(val acc {best_val_acc*100:.2f}%)...")
    head.load_state_dict(torch.load(f"best_head_{dataset_name}.pth"))
    test_acc = evaluate(head, backbone, test_loader, device, sigma, lam, num_classes)
    print(f"*** FINAL TEST Accuracy: {test_acc*100:.2f}% ***")

    torch.save(head.state_dict(), f"head_{dataset_name}.pth")
    np.save(f"fisher_log_{dataset_name}.npy", np.array(fisher_log))
    print(f"Training complete.  Best val: {best_val_acc*100:.2f}% @ epoch {best_epoch}  |  "
          f"Test: {test_acc*100:.2f}%  |  final tau: {head.temperature.item():.2f}")
    return test_acc


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = {
        "seed"             : 42,         # v3: reproducibility (Issue #11)
        "epochs"           : 50,
        "batch_size"       : 128,
        "num_workers"      : 4,
        "embedding_dim"    : 128,
        "lr"               : 1e-3,
        "lr_phase2"        : 2e-4,
        "sigma"            : -1,
        "lam"              : 0.5,
        "temperature"      : 20.0,       # v3: initial temperature, learnable
        "mfa_weight"       : 0.1,        # v3: alpha for Phase 2 multi-task
        "apt_delta"        : 0.05,
        "apt_patience"     : 5,
        "max_phase1_epochs": 15,
    }

    print("\nSelect dataset to train on:")
    print("  1. CIFAR-10")
    print("  2. Fashion-MNIST")
    print("  3. STL-10")
    print("  4. All three sequentially")

    MENU = {
        "1": ["cifar10"],
        "2": ["fashionmnist"],
        "3": ["stl10"],
        "4": ["cifar10", "fashionmnist", "stl10"],
    }

    while True:
        choice = input("\nEnter option (1 / 2 / 3 / 4): ").strip()
        if choice in MENU:
            break
        print("  Invalid option.")

    results = {}
    for ds in MENU[choice]:
        results[ds] = train(config, ds)

    print("\n" + "="*60)
    print("v3 SUMMARY (test accuracy on held-out test set)")
    print("="*60)
    for ds, acc in results.items():
        print(f"  {ds:15s}: {acc*100:.2f}%")
