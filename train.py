# Last updated: 2026-04-15 18:30
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from dataset import get_loaders
from backbone import get_backbone
from nmfc import nmfc_loss, linear_mfa_loss, fisher_ratio, auto_sigma, APTController


# ─────────────────────────────────────────────────────────────
# Projection head: 512D → embedding_dim → L2-normalized hypersphere
# ─────────────────────────────────────────────────────────────

class ProjectionHead(nn.Module):
    # v2 (2026-04-19): Deeper head with BatchNorm.
    # Frozen backbone means this is the ONLY learnable component.
    # Ref 5 Contrastive/SimCLR: deeper heads + BN improve representations.
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
# Extract all embeddings from backbone (no grad needed)
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(backbone, loader, device):
    all_emb, all_lbl = [], []
    for images, labels in loader:
        images = images.to(device)
        emb = backbone(images)
        all_emb.append(emb.cpu())
        all_lbl.append(labels)
    return torch.cat(all_emb), torch.cat(all_lbl)


# ─────────────────────────────────────────────────────────────
# Evaluate top-1 accuracy
# ─────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(head, backbone, loader, device, sigma, lam, num_classes):
    head.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            feats = backbone(images)
        embeddings = head(feats)
        _, logits = nmfc_loss(embeddings, labels, sigma, lam, num_classes)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    head.train()
    return correct / total


# ─────────────────────────────────────────────────────────────
# Training loop (single dataset)
# ─────────────────────────────────────────────────────────────

def train(config, dataset_name):
    """
    Train NMFC on a single dataset.

    EN: Output files are named with the dataset so results from different
        datasets never overwrite each other:
          head_<dataset>.pth       — final weights
          best_head_<dataset>.pth  — best Phase 2 weights
          fisher_log_<dataset>.npy — per-epoch Fisher Ratio
    KR: 출력 파일에 데이터셋 이름을 붙여 서로 덮어쓰지 않도록 함:
          head_<dataset>.pth       — 최종 가중치
          best_head_<dataset>.pth  — Phase 2 최적 가중치
          fisher_log_<dataset>.npy — 에포크별 Fisher Ratio
    """
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()}  |  Device: {device}")
    print(f"{'='*60}")

    # Data
    train_loader, test_loader = get_loaders(
        dataset_name,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"]
    )

    # Backbone (frozen ResNet-18)
    backbone = get_backbone(device, pretrained=True, freeze=True)

    # Projection head (trainable)
    head = ProjectionHead(in_dim=512, out_dim=config["embedding_dim"]).to(device)

    # Optimizer
    optimizer = optim.Adam(head.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    # EN: max_phase1_epochs is a general APT parameter, not dataset-specific.
    #     -1 = rely only on FR stabilization (pure APT, original paper behavior)
    #      N = force Phase 2 transition at epoch N regardless of dataset.
    #     Useful when FR never stabilizes (e.g. Fashion-MNIST, STL-10 with
    #     frozen ResNet-18), but applies equally to any dataset.
    #     Clamped to config["epochs"] so it can never exceed total training.
    #     Last updated: 2026-04-15
    # KR: max_phase1_epochs는 데이터셋별이 아닌 일반 APT 파라미터.
    #     -1 = FR 안정화에만 의존 (순수 APT, 원본 논문 동작)
    #      N = 데이터셋에 관계없이 epoch N에서 Phase 2 강제 전환.
    #     FR이 안정화되지 않는 경우(예: Fashion-MNIST, STL-10)에 유용하나
    #     모든 데이터셋에 동일하게 적용됨.
    #     총 훈련 epochs를 초과하지 않도록 클램핑.
    #     최종 수정: 2026-04-15
    raw = config["max_phase1_epochs"]
    if raw == -1:
        max_p1 = None
    else:
        max_p1 = min(int(raw), config["epochs"])
        if max_p1 != raw:
            print(f"[APT] max_phase1_epochs clamped to {max_p1} (cannot exceed epochs={config['epochs']})")

    apt = APTController(
        delta=config["apt_delta"],
        patience=config["apt_patience"],
        max_phase1_epochs=max_p1
    )

    num_classes = 10
    sigma = config["sigma"]
    lam   = config["lam"]
    fisher_log = []
    phase2_triggered = False
    best_acc = 0.0          # EN: track best Phase 2 accuracy for checkpointing
                            # KR: 체크포인트를 위한 Phase 2 최고 정확도 추적

    for epoch in range(1, config["epochs"] + 1):
        head.train()
        total_loss = 0.0
        fr_accum   = 0.0
        steps      = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            with torch.no_grad():
                feats = backbone(images)

            embeddings = head(feats)

            # Auto sigma: median of pairwise distances in mini-batch
            if config["sigma"] == -1:
                sigma = auto_sigma(embeddings.detach())

            # ── Phase selection ──────────────────────────────
            if apt.phase == 1:
                loss = linear_mfa_loss(embeddings, labels, num_classes)
            else:
                loss, _ = nmfc_loss(embeddings, labels, sigma, lam, num_classes)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # Accumulate Fisher Ratio for APT (cheap, no extra backward)
            with torch.no_grad():
                fr_accum += fisher_ratio(embeddings.detach(), labels, num_classes)

            steps += 1

        # EN: Only step the cosine scheduler during Phase 1.
        #     In Phase 2 the lr is fixed to lr_phase2; continuing to call
        #     scheduler.step() lets the cosine formula (anchored at lr=1e-3)
        #     override the manually-set lr_phase2 and causes loss divergence.
        #     Observed: loss rising from 1.797 to 1.806 after epoch ~30 with
        #     emb=128. Peak accuracy was epoch 30 (75.97%), then degraded.
        #     Fix: freeze the scheduler once Phase 2 is active.
        #     Last updated: 2026-04-14
        # KR: 코사인 스케줄러는 Phase 1에서만 step.
        #     Phase 2에서는 lr을 lr_phase2로 고정해야 하는데,
        #     계속 step하면 코사인 공식(lr=1e-3 기준)이 수동 설정값을
        #     덮어써 손실 발산을 유발함.
        #     관찰: emb=128에서 epoch ~30 이후 손실 1.797 → 1.806 상승,
        #     정확도 75.97%(epoch 30) 이후 하락.
        #     수정: Phase 2 진입 후 스케줄러 동결.
        #     최종 수정: 2026-04-14
        if apt.phase == 1:
            scheduler.step()

        avg_loss = total_loss / steps
        avg_fr   = fr_accum / steps
        fisher_log.append(avg_fr)
        phase    = apt.update(avg_fr)

        # Reduce lr when Phase 2 is triggered for the first time
        if apt.phase == 2 and not phase2_triggered:
            phase2_triggered = True
            for g in optimizer.param_groups:
                g["lr"] = config["lr_phase2"]
            print(f"[APT] Learning rate reduced to {config['lr_phase2']}")

        print(f"Epoch {epoch:3d} | Phase {phase} | Loss {avg_loss:.4f} | Fisher Ratio {avg_fr:.4f}")

        # Evaluate every 5 epochs and save best checkpoint
        # EN: Phase 2 accuracy peaks early (epoch ~25-30) then degrades as the
        #     NMFC geometric loss reshapes embeddings away from Phase 1 structure.
        #     Save best_head_<dataset>.pth whenever a new accuracy peak is found.
        #     Last updated: 2026-04-14
        # KR: Phase 2 정확도는 초기(epoch ~25-30)에 정점을 찍고 하락함.
        #     NMFC 기하학적 손실이 Phase 1 구조를 점진적으로 해체하기 때문.
        #     새로운 정확도 최고점 발견 시 best_head_<dataset>.pth 저장.
        #     최종 수정: 2026-04-14
        if epoch % 5 == 0 and apt.phase == 2:
            acc = evaluate(head, backbone, test_loader, device, sigma, lam, num_classes)
            print(f"           >> Test Accuracy: {acc*100:.2f}%")
            if acc > best_acc:
                best_acc = acc
                torch.save(head.state_dict(), f"best_head_{dataset_name}.pth")
                print(f"           >> New best! Saved to best_head_{dataset_name}.pth ({acc*100:.2f}%)")

    # Save final weights and Fisher Ratio log
    torch.save(head.state_dict(), f"head_{dataset_name}.pth")
    np.save(f"fisher_log_{dataset_name}.npy", np.array(fisher_log))
    print(f"Training complete. Final model  → head_{dataset_name}.pth")
    print(f"                   Best model   → best_head_{dataset_name}.pth ({best_acc*100:.2f}%)")
    print(f"                   Fisher log   → fisher_log_{dataset_name}.npy")


# ─────────────────────────────────────────────────────────────
# Entry point — dataset selection
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    config = {
        "epochs"       : 50,      # EN: 50 epochs for Phase 2 to train sufficiently (2026-04-19)
                                  # KR: 빠른 반복을 위해 35로 감소 (2026-04-15)
        "batch_size"   : 128,
        "num_workers"  : 4,
        "embedding_dim": 128,     # EN: 128 outperforms 64 (+1.4% accuracy, 2026-04-14)
                                  # KR: 128이 64보다 +1.4% 정확도 향상 (2026-04-14)
        "lr"           : 1e-3,
        "lr_phase2"    : 2e-4,    # EN: fixed lr for Phase 2 (scheduler frozen)
                                  # KR: Phase 2 고정 lr (스케줄러 동결)
        "sigma"        : -1,      # -1 = auto (median heuristic, scale=0.5)
        "lam"          : 0.5,
        "apt_delta"    : 0.05,
        "apt_patience" : 5,
        # EN: Max epochs to stay in Phase 1 before forcing transition to Phase 2.
        #     -1 = automatic (APT triggers only when Fisher Ratio stabilizes).
        #      N = force Phase 2 at epoch N, works for any dataset.
        #     Cannot exceed "epochs" — clamped automatically if so.
        #     Recommended: -1 for CIFAR-10 (FR stabilizes naturally),
        #                  15 for Fashion-MNIST or STL-10 (FR may not stabilize).
        #     Last updated: 2026-04-15
        # KR: Phase 2로 강제 전환하기 전 Phase 1의 최대 에포크 수.
        #     -1 = 자동 (Fisher Ratio가 안정화될 때만 APT 트리거).
        #      N = 데이터셋에 관계없이 epoch N에서 Phase 2 강제 전환.
        #     "epochs"를 초과할 수 없음 — 초과 시 자동 클램핑.
        #     권장: CIFAR-10은 -1 (FR이 자연적으로 안정화),
        #           Fashion-MNIST 또는 STL-10은 15 (FR이 안정화 안될 수 있음).
        #     최종 수정: 2026-04-15
        "max_phase1_epochs": 15,
    }

    # EN: Interactive dataset selection menu at runtime.
    # KR: 실행 시 대화형 데이터셋 선택 메뉴.
    print("\nSelect dataset to train on:")
    print("  1. CIFAR-10        (60,000 images, 32x32 RGB,  10 classes)")
    print("  2. Fashion-MNIST   (70,000 images, 28x28 gray, 10 classes)")
    print("  3. STL-10          ( 5,000 train,  96x96 RGB,  10 classes)")
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
        print("  Invalid option. Please enter 1, 2, 3, or 4.")

    for ds in MENU[choice]:
        train(config, ds)
