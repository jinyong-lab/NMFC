# NMFC Changelog

모든 변경 사항은 이 파일에 기록합니다.

---

## [v2] - 2026-04-19

### 핵심 개선 (74.53% → 89.09%, +14.56%p on CIFAR-10)

#### Added
- `nmfc.py::compute_logits()` — **temperature scaling** (τ=20) 파라미터 추가
  - `return temperature * (E_pos - lam * E_neg)`
  - 근거: Ref SphereFace, Prototypical Networks
- `dataset.py` — CIFAR-10/Fashion-MNIST/STL-10 train transform에 **data augmentation** 적용
  - `RandomResizedCrop(224, scale=(0.8, 1.0))`
  - `RandomHorizontalFlip()`
  - test transform은 augmentation 없이 `Resize(224)`만
  - 근거: Ref Contrastive (SimCLR), FaceNet
- `train.py::ProjectionHead` — **3층 + BatchNorm** 구조로 확장
  - 512 → 512 (BN+ReLU) → 256 (BN+ReLU) → 128
  - 근거: Ref Contrastive (SimCLR)

#### Changed
- `train.py::config`
  - `epochs`: 35 → **50** (Phase 2 학습 시간 확보)
  - `max_phase1_epochs`: -1 → **15** (APT 자연 전환 실패 시 강제)

### 결과
| Dataset | v0 | v2 |
|---------|:---:|:---:|
| CIFAR-10 | 74.53% | **89.09%** |
| Fashion-MNIST | — | **89.65%** |
| STL-10 | — | **94.34%** |

---

## [v1.1] - 2026-04-18 (⚠️ 실패, 원복됨)

### Attempted
- `train.py::ProjectionHead` — `F.normalize(z, p=2, dim=1)` 추가 (L2 정규화)
- `nmfc.py::auto_sigma` — scale 0.5 → 1.0
- `train.py::config` — `max_phase1_epochs` -1 → 15, `epochs` 35 → 50

### 결과
- **53.94%** (v0 대비 -20.6%p 하락) — 실패
- 원인: L2 정규화로 pairwise dist가 [0,2]로 제한되어 logit range가 ~0.1로 좁아짐.
  Phase 2 진입 시 loss 0.03 → 2.0 급등, Phase 1 구조 파괴.

### Reverted
- 모든 변경을 v0로 원복 후 v2에서 다른 접근 (temperature scaling) 채택.

---

## [v0] - 2026-04-16 (기준선)

원본: https://github.com/JASorianoHernandez/nmfc-implementation

- CIFAR-10 74.53% (원본 재현)
- Phase 1 MFA → APT → Phase 2 NMFC 구조
- ResNet-18 frozen backbone + ProjectionHead (2층, BN 없음)
- 데이터 증강 없음
- 원본의 한계: logit range ~0.27로 softmax가 거의 균등 확률
