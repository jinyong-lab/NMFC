# Last updated: 2026-04-15 18:30
# =============================================================
# diagnostics.py
# -------------------------------------------------------------
# EN: Diagnostic script to inspect E_pos, E_neg, sigma and
#     logit values before training. Helps identify scale issues
#     in the NMFC energy computation.
# KR: 훈련 전 E_pos, E_neg, sigma 및 로짓 값을 검사하는
#     진단 스크립트. NMFC 에너지 계산의 스케일 문제를 파악하는 데 사용.
# Last updated: 2026-04-14
# =============================================================

import torch
from dataset import get_cifar10_loaders
from backbone import get_backbone
from train import ProjectionHead
from nmfc import compute_energies, auto_sigma


def run_diagnostics():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load one batch ────────────────────────────────────────────
    # EN: num_workers=0 required on Windows to avoid multiprocessing error.
    # KR: Windows에서 멀티프로세싱 오류 방지를 위해 num_workers=0 사용.
    train_loader, _ = get_cifar10_loaders(batch_size=128, num_workers=0)
    backbone = get_backbone(device)
    head = ProjectionHead(512, 64).to(device)

    images, labels = next(iter(train_loader))
    images, labels = images.to(device), labels.to(device)

    # ── Compute embeddings and energies ──────────────────────────
    # EN: Pass images through backbone + head, then compute energies.
    # KR: 이미지를 백본 + 헤드에 통과시킨 후 에너지 계산.
    with torch.no_grad():
        feats  = backbone(images)
        emb    = head(feats)
        sigma  = auto_sigma(emb)
        E_pos, E_neg = compute_energies(emb, labels, sigma, num_classes=10)
        logits = E_pos - 0.5 * E_neg

    # ── Print results ─────────────────────────────────────────────
    # EN: Print statistics to understand energy and logit scales.
    # KR: 에너지 및 로짓 스케일을 파악하기 위한 통계 출력.
    print(f"\n--- Energy Diagnostics ---")
    print(f"sigma        : {sigma:.4f}")
    print(f"E_pos  mean  : {E_pos.mean():.4f}  |  std: {E_pos.std():.4f}  |  min: {E_pos.min():.4f}  |  max: {E_pos.max():.4f}")
    print(f"E_neg  mean  : {E_neg.mean():.4f}  |  std: {E_neg.std():.4f}  |  min: {E_neg.min():.4f}  |  max: {E_neg.max():.4f}")
    print(f"logits mean  : {logits.mean():.4f}  |  std: {logits.std():.4f}  |  min: {logits.min():.4f}  |  max: {logits.max():.4f}")

    # ── Per-class logit stats ─────────────────────────────────────
    # EN: Show mean logit per class to check if classes are separable.
    # KR: 클래스별 평균 로짓을 확인하여 클래스 분리 가능성 파악.
    print(f"\n--- Per-class logit mean ---")
    for c in range(10):
        mask = labels == c
        if mask.sum() > 0:
            print(f"  Class {c}: logit mean = {logits[mask].mean():.4f}")


if __name__ == "__main__":
    run_diagnostics()
