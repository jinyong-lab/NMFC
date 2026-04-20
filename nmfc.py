# Last updated: 2026-04-15 18:30
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Section 3.1 — Local Affinity Matrix
# ─────────────────────────────────────────────────────────────

def compute_affinities(embeddings, labels, sigma):
    """
    Compute positive (intra-class) and negative (inter-class) affinity
    matrices for a mini-batch.

    Args:
        embeddings : (N, D) tensor of L2-normalized embeddings
        labels     : (N,)   tensor of class labels
        sigma      : kernel bandwidth (scalar)

    Returns:
        A_pos : (N, N) positive affinity matrix (same class, excluding self)
        A_neg : (N, N) negative affinity matrix (different class)
    """
    N = embeddings.size(0)

    # Pairwise squared Euclidean distances: (N, N)
    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # (N, N, D)
    dist_sq = (diff ** 2).sum(dim=-1)                         # (N, N)

    # RBF kernel
    K = torch.exp(-dist_sq / (2 * sigma ** 2))               # (N, N)

    # Class masks
    same_class = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (N, N)
    diff_class = 1.0 - same_class

    # Exclude self-pairs from positive affinity
    eye = torch.eye(N, device=embeddings.device)
    A_pos = K * same_class * (1 - eye)
    A_neg = K * diff_class

    return A_pos, A_neg


def compute_soft_weights(A_pos, A_neg):
    """
    Normalize affinities into soft neighborhood weights (Section 3.1).

    W_pos[i,j] = A_pos[i,j] / sum_k A_pos[i,k]
    W_neg[i,j] = A_neg[i,j] / sum_k A_neg[i,k]
    """
    W_pos = A_pos / (A_pos.sum(dim=1, keepdim=True) + 1e-8)
    W_neg = A_neg / (A_neg.sum(dim=1, keepdim=True) + 1e-8)
    return W_pos, W_neg


# ─────────────────────────────────────────────────────────────
# Section 3.2 — Local Geometric Energy and Logit
# ─────────────────────────────────────────────────────────────

def auto_sigma(embeddings, scale=0.5):
    """
    Median heuristic: sigma = scale * median of pairwise distances (Section 4.3).

    EN: scale=0.5 narrows the kernel so only close neighbors get high K values,
        increasing contrast between E_pos and E_neg.
        scale=2.0 (previous) made sigma too large — all K values converged
        to ~0.88, making logits indistinguishable (range ~0.009).
        scale=1.0 (original) also too flat (range ~0.029).
        Last updated: 2026-04-14 12:04
    KR: scale=0.5로 커널을 좁혀 가까운 이웃만 높은 K값을 가지게 하여
        E_pos와 E_neg의 대비를 높임.
        scale=2.0(이전)에서는 sigma가 너무 커서 모든 K값이 ~0.88로
        수렴하여 로짓을 구분할 수 없었음(범위 ~0.009).
        scale=1.0(원본)도 너무 평탄했음(범위 ~0.029).
        최종 수정: 2026-04-14 12:04
    """
    with torch.no_grad():
        diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)
        dist = (diff ** 2).sum(dim=-1).sqrt()
        return scale * dist[dist > 0].median().item()


def compute_energies(embeddings, labels, sigma, num_classes):
    """
    Compute E_pos and E_neg for each sample × class pair.

    Returns:
        E_pos : (N, C)
        E_neg : (N, C)
    """
    N = embeddings.size(0)
    C = num_classes
    device = embeddings.device

    diff = embeddings.unsqueeze(0) - embeddings.unsqueeze(1)  # (N, N, D)
    dist_sq = (diff ** 2).sum(dim=-1)                         # (N, N)
    K = torch.exp(-dist_sq / (2 * sigma ** 2))                # (N, N)

    eye = torch.eye(N, device=device)

    E_pos = torch.zeros(N, C, device=device)
    E_neg = torch.zeros(N, C, device=device)

    for c in range(C):
        same_c = (labels == c).float()                        # (N,)
        diff_c = 1.0 - same_c

        # Positive: average of K[i,j] where j is in class c, j != i
        mask_pos = same_c.unsqueeze(0) * (1 - eye)           # (N, N)
        pos_count = mask_pos.sum(dim=1).clamp(min=1)          # (N,) avoid div by 0
        E_pos[:, c] = (K * mask_pos).sum(dim=1) / pos_count

        # Negative: average of K[i,j] where j is NOT in class c
        mask_neg = diff_c.unsqueeze(0)                        # (N, N)
        neg_count = mask_neg.sum(dim=1).clamp(min=1)          # (N,) avoid div by 0
        E_neg[:, c] = (K * mask_neg).sum(dim=1) / neg_count

    return E_pos, E_neg


def empirical_mean_scaling(E_pos, E_neg):
    """
    Scale energies by their empirical batch mean (Section 3.2).

    E_pos_scaled[i,c] = E_pos[i,c] / mean(E_pos)
    E_neg_scaled[i,c] = E_neg[i,c] / mean(E_neg)
    """
    E_pos_scaled = E_pos / (E_pos.mean() + 1e-8)
    E_neg_scaled = E_neg / (E_neg.mean() + 1e-8)
    return E_pos_scaled, E_neg_scaled


def compute_logits(E_pos, E_neg, lam, temperature=20.0):
    """
    Local Geometric Logit: g(z, c) = tau * (E_pos(z,c) - lambda * E_neg(z,c))

    Temperature scaling amplifies the narrow logit range (~0.27) so that
    softmax produces peaked probabilities instead of near-uniform ~10%.
    Ref 11 SphereFace, Ref 12 Prototypical both use temperature/scale.

    temperature can be a scalar or a tensor (for learnable temperature).
    """
    return temperature * (E_pos - lam * E_neg)


def ood_score(logits):
    """
    OOD detection via Free Energy (Ref 10 JEM §3):
        E(x) = -LogSumExp_k logits[k]

    High E(x) -> low p(x) -> OOD candidate.
    Returns per-sample OOD score (higher = more OOD-like).
    """
    return -torch.logsumexp(logits, dim=1)


# ─────────────────────────────────────────────────────────────
# Section 3.3 — Linear MFA Loss (Phase I warm-up)
# ─────────────────────────────────────────────────────────────

def compute_scatter_matrices(embeddings, labels, num_classes):
    """
    Compute within-class (S_w) and between-class (S_b) scatter matrices
    for the Linear MFA warm-up phase.
    """
    D = embeddings.size(1)
    device = embeddings.device

    global_mean = embeddings.mean(dim=0)                      # (D,)

    S_w = torch.zeros(D, D, device=device)
    S_b = torch.zeros(D, D, device=device)

    for c in range(num_classes):
        mask = labels == c
        if mask.sum() < 2:
            continue
        X_c = embeddings[mask]                                # (N_c, D)
        mu_c = X_c.mean(dim=0)                               # (D,)

        # Within-class scatter
        diff_w = X_c - mu_c                                   # (N_c, D)
        S_w += diff_w.T @ diff_w

        # Between-class scatter
        diff_b = (mu_c - global_mean).unsqueeze(1)            # (D, 1)
        S_b += mask.sum().float() * (diff_b @ diff_b.T)

    return S_w, S_b


def linear_mfa_loss(embeddings, labels, num_classes, eps=1e-6):
    """
    Linear MFA loss: minimize trace(S_w) / (trace(S_b) + eps)

    EN: The gradient anchor (embeddings.sum() * 0) ensures the loss always
        has a grad_fn even when a mini-batch has fewer than 2 samples per
        class, which leaves S_w and S_b as zero tensors with no grad_fn.
        This can happen with small datasets (e.g. STL-10, 500/class) when
        max_phase1_epochs=-1 and Phase 1 runs for many epochs.
        The anchor adds 0 to the loss value so numerics are unaffected.
        Last updated: 2026-04-15
    KR: 그래디언트 앵커(embeddings.sum() * 0)는 미니배치에 클래스당
        샘플이 2개 미만일 때 S_w와 S_b가 grad_fn 없는 영행렬로 남아도
        loss가 항상 grad_fn을 갖도록 보장함.
        소규모 데이터셋(예: STL-10, 클래스당 500개)에서 max_phase1_epochs=-1
        설정으로 Phase 1이 오래 실행될 때 발생할 수 있음.
        앵커는 loss 값에 0을 더하므로 수치에 영향 없음.
        최종 수정: 2026-04-15
    """
    S_w, S_b = compute_scatter_matrices(embeddings, labels, num_classes)
    loss = torch.trace(S_w) / (torch.trace(S_b) + eps)
    # Gradient anchor: ensures grad_fn exists even when scatter matrices are zero
    loss = loss + embeddings.sum() * 0.0
    return loss


def fisher_ratio(embeddings, labels, num_classes, eps=1e-6):
    """
    Fisher Ratio: trace(S_b) / (trace(S_w) + eps)
    Used by APT to monitor convergence.
    """
    S_w, S_b = compute_scatter_matrices(embeddings, labels, num_classes)
    return (torch.trace(S_b) / (torch.trace(S_w) + eps)).item()


# ─────────────────────────────────────────────────────────────
# NMFC Loss (Phase II)
# ─────────────────────────────────────────────────────────────

def nmfc_loss(embeddings, labels, sigma, lam, num_classes):
    """
    Full NMFC loss: NLL over Local Geometric Logits.

    Steps:
      1. Compute E_pos, E_neg per (sample, class)
      2. [2026-04-14] empirical_mean_scaling removed: energies are already
         in [0,1] after averaging by neighbor count in compute_energies.
         Scaling by batch mean was flattening all values to ~1.0,
         removing discriminative signal from the logits.
         -- 2026-04-14: compute_energies에서 이웃 수로 평균을 낸 후
         에너지가 이미 [0,1] 범위에 있으므로 empirical_mean_scaling 제거.
         배치 평균으로 나누면 모든 값이 ~1.0으로 평탄화되어
         로짓의 판별 신호가 사라짐.
      3. Build logits g = E_pos - lambda * E_neg
      4. Softmax + NLL
    """
    E_pos, E_neg = compute_energies(embeddings, labels, sigma, num_classes)

    # [2026-04-14] empirical_mean_scaling removed (see docstring above)
    # EN: was: E_pos_s, E_neg_s = empirical_mean_scaling(E_pos, E_neg)
    # KR: 이전: E_pos_s, E_neg_s = empirical_mean_scaling(E_pos, E_neg)

    logits = compute_logits(E_pos, E_neg, lam)
    loss = F.cross_entropy(logits, labels)
    return loss, logits


# ─────────────────────────────────────────────────────────────
# APT Controller (Section 3.3)
# ─────────────────────────────────────────────────────────────

class APTController:
    """
    Automatic Phase Transition controller.

    Monitors the moving average of the Fisher Ratio and triggers
    the transition from Phase I (linear MFA) to Phase II (NMFC)
    when its derivative falls below threshold delta.

    EN: max_phase1_epochs is a safety valve for datasets where the Fisher
        Ratio never stabilizes (e.g. Fashion-MNIST with frozen ResNet-18,
        where FR grows exponentially to ~20,000 because the pretrained
        features separate the classes too easily). If Phase 1 has not
        triggered naturally by max_phase1_epochs, the transition is forced.
        Set to None to disable (original behavior, CIFAR-10 default).
        Last updated: 2026-04-15
    KR: max_phase1_epochs는 Fisher Ratio가 안정화되지 않는 데이터셋
        (예: 고정된 ResNet-18의 Fashion-MNIST — 사전학습 특징이 클래스를
        너무 쉽게 분리해 FR이 ~20,000까지 지수적으로 증가)을 위한
        안전 밸브. max_phase1_epochs까지 자연 전환이 없으면 강제 전환.
        None으로 설정 시 비활성화 (원래 동작, CIFAR-10 기본값).
        최종 수정: 2026-04-15
    """
    def __init__(self, delta=0.01, patience=5, window=5, max_phase1_epochs=None):
        self.delta = delta
        self.patience = patience
        self.window = window
        self.max_phase1_epochs = max_phase1_epochs
        self.history = []          # Fisher ratio history
        self.counter = 0
        self.phase = 1             # 1 = linear warm-up, 2 = NMFC

    def update(self, ratio):
        """Call once per epoch with the current Fisher Ratio."""
        if self.phase == 2:
            return 2

        self.history.append(ratio)
        epoch = len(self.history)

        # Safety valve: force transition if Phase 1 runs too long
        if self.max_phase1_epochs is not None and epoch >= self.max_phase1_epochs:
            self.phase = 2
            print(f"[APT] Phase transition forced at epoch {epoch} "
                  f"(max_phase1_epochs={self.max_phase1_epochs}). "
                  f"Fisher Ratio: {ratio:.4f}")
            return self.phase

        if len(self.history) < self.window + 1:
            return self.phase

        # Moving average of last `window` values
        ma_now  = sum(self.history[-self.window:]) / self.window
        ma_prev = sum(self.history[-self.window - 1:-1]) / self.window

        # Original: absolute delta (scale-dependent, fails at large FR values)
        # delta_fr = abs(ma_now - ma_prev)

        # Relative delta: change normalized by current MA (scale-invariant)
        delta_fr = abs(ma_now - ma_prev) / (abs(ma_prev) + 1e-8)

        if delta_fr < self.delta:
            self.counter += 1
        else:
            self.counter = 0

        if self.counter >= self.patience:
            self.phase = 2
            print(f"[APT] Phase transition triggered. Fisher Ratio stabilized at {ratio:.4f}")

        return self.phase
