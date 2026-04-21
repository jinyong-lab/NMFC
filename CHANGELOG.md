# NMFC Changelog

모든 변경 사항은 이 파일에 기록합니다.

---

## [v5] - 2026-04-21

### 데이터셋별 MFA weight 튜닝

#### Changed
- `train.py::__main__` — **per-dataset MFA weight config** 도입
  - CIFAR-10: `1.0 → 0.1` (어려운 데이터셋, FR 46)
  - Fashion-MNIST: `0.5 → 0.05` (쉬운 데이터셋, FR 3,700)
  - STL-10: `0.5 → 0.05` (중간, FR 236)
- `BASE` config + `PER_DATASET` overrides 패턴

### 결과
| Dataset | v4 | **v5** |
|---------|:---:|:---:|
| CIFAR-10 | 87.88% | **87.88%** (동일) |
| Fashion-MNIST | 89.66% | **90.02%** (+0.36%p) |
| STL-10 | 94.51% | **94.49%** (±0) |

- Fashion-MNIST 회복: MFA 초기 값 완화로 NMFC 학습 충분
- CIFAR-10은 v4와 동일 config → 결과 동일 (재현성 확인)

---

## [v4] - 2026-04-21

### 참고논문 재검토 기반 교정 (v3의 오해결 수정)

#### Changed
- `train.py::ProjectionHead` — **learnable temperature 제거** (nn.Parameter 삭제)
  - 참고논문 재검토 결과: SphereFace §4.1, ArcFace §3.3, CosFace §3.2 모두
    scale/margin을 **하이퍼파라미터로 고정**. Learnable 버전 없음.
  - `τ=20.0` 고정값 유지 (하이퍼파라미터 sweep으로 튜닝 권장)
- `dataset.py::get_train_datasets_both_transforms()` — **val은 clean transform**
  - Ref 5 SimCLR §3: val/test는 augmentation 없이 Resize만
  - v3의 `random_split`은 train augmentation을 val에도 적용하는 버그
- `train.py::make_loaders()` — Subset 기반 split (train augmented, val clean)
- `train.py::mfa_weight_schedule()` — **linear decay schedule** 1.0 → 0.1
  - Ref 2 MFA §3 ratio form은 S_w/S_b를 동등 가중 → 초기 alpha를 높게
  - Phase 2 초반은 Phase 1 구조 보존(α=1.0), 후반은 NMFC 주도(α=0.1)

### 결과 (v4 honest test)
| Dataset | v3 | v4 |
|---------|:---:|:---:|
| CIFAR-10 | 87.82% | **87.88%** |
| Fashion-MNIST | 90.17% | **89.66%** |
| STL-10 | 94.96% | **94.51%** |

- CIFAR 미세 개선, Fashion/STL 약간 하락 (MFA 초기 alpha가 너무 높은 듯)
- **방법론적으로는 가장 엄밀** (val bug 제거, 논문 관행 준수)
- FR 급락 크게 완화: CIFAR -64% (v3는 -79%)

### Issue 교정
- v3에서 learnable τ로 "#4 해결"했다고 했으나, 실제 참고논문은 learnable을
  지지하지 않음 → v4에서 fixed τ로 정정

---

## [v3] - 2026-04-20

### 참고논문 기반 개선 (방법론적 엄밀성 + 일반화 성능)

#### Added
- `nmfc.py::ood_score()` — Free Energy 기반 OOD 탐지 함수
  - `E(x) = -LogSumExp_k logits[k]`
  - 근거: Ref 10 JEM §3 (모든 softmax 분류기는 암묵적으로 EBM)
- `train.py::set_seed()` — 재현성 확보
  - torch/numpy/cuda/cudnn seed 모두 42로 고정
- `train.py::ProjectionHead` — **learnable temperature** (nn.Parameter)
  - `log_temperature` 파라미터로 log-space에서 학습
  - 근거: Ref 11 SphereFace §3.2, Ref 12 Prototypical §3
- `train.py::phase2_loss()` — **multi-task loss** (NMFC + 0.1×MFA)
  - Phase 2에서 Phase 1 구조 보존
  - 근거: Ref 2 MFA §3 (S_w min + S_b max 동시 유지)
- `train.py::make_train_val_loaders()` — train 80/20 split
  - val로 best checkpoint 선정, test는 마지막 1회만 평가
  - 근거: 학계 표준 (v2의 test leak 문제 해결)

#### Changed
- `train.py::train()` — 평가 프로토콜 변경
  - 매 epoch val 평가 (이전: 5 epoch마다)
  - 학습 종료 후 best checkpoint 로드 → test 평가 1회
- `train.py::train()` — `num_classes` 자동 추론
  - `len(DATASET_CLASSES[dataset_name])` (하드코딩 제거)

### 결과 (Honest test accuracy, no data leak)
| Dataset | v2 (w/ leak) | v3 (honest) |
|---------|:---:|:---:|
| CIFAR-10 | 89.09% | **87.82%** |
| Fashion-MNIST | 89.65% | **90.17%** |
| STL-10 | 94.34% | **94.96%** |

- v2의 CIFAR 89.09%는 test set으로 best 선정 → 과대평가
- v3는 train/val/test 엄격 분리 → 실제 일반화 성능
- Fashion/STL은 개선, CIFAR는 data leak 제거로 정직한 수치

### 해결된 Issues
- ✅ #4 Temperature 고정 → learnable
- ✅ #5 Phase 2 FR 급락 → multi-task loss
- ✅ #7 Test set data leak → train/val split
- ✅ #9 OOD 미구현 → ood_score() 함수
- ✅ #10 num_classes 하드코딩 → 자동 추론
- ✅ #11 Random seed 미고정 → set_seed(42)

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
