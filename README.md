# NMFC — Nonlinear Marginal Fisher Classification

논문 **"Nonlinear Marginal Fisher Classification via Local Geometric Logit Modeling and Automatic Phase Transition"** 의 PyTorch 구현.

12편의 참고 논문을 분석해 원 구현의 한계를 찾아 개선한 v2 버전입니다.

---

## 📊 결과 (CIFAR-10 / Fashion-MNIST / STL-10)

| 데이터셋 | v0 (원본) | **v2 (개선)** | 향상 |
|---------|:---:|:---:|:---:|
| CIFAR-10 | 74.53% | **89.09%** | +14.56%p |
| Fashion-MNIST | — | **89.65%** | — |
| STL-10 | — | **94.34%** | — |

---

## 🔑 v2 핵심 개선 사항

| # | 변경 | 파일 | 참고논문 |
|---|------|------|---------|
| 1 | **Temperature scaling** (logit × τ=20) | `nmfc.py` | Ref SphereFace, Prototypical |
| 2 | **Data augmentation** (RandomResizedCrop + HorizontalFlip) | `dataset.py` | Ref Contrastive, FaceNet |
| 3 | **Deeper ProjectionHead + BatchNorm** (3층) | `train.py` | Ref SimCLR |
| 4 | **max_phase1_epochs=15** (APT 강제 전환) | `train.py` | 원 논문 Sec 3.3 |

### 근본 원인
원본 구현의 logit 범위가 ~0.27로 너무 좁아 softmax가 거의 균등 확률(~10%)을 생성, CE gradient가 사실상 사라지는 문제가 있었다. Temperature scaling으로 logit을 증폭하여 근본 해결.

---

## 📁 파일 구성

```
NMFC/
├── dataset.py        # CIFAR-10 / Fashion-MNIST / STL-10 로더 (augmentation 적용)
├── backbone.py       # ResNet-18 frozen feature extractor
├── nmfc.py           # 핵심 알고리즘 (affinity, energy, logit, APT)
├── train.py          # 학습 루프 (Phase 1 MFA → APT → Phase 2 NMFC)
├── diagnostics.py    # 시각화 유틸
└── requirements.txt  # 의존성
```

---

## 🚀 사용법

### 환경 설정
```bash
pip install -r requirements.txt
```

### 학습
```bash
python train.py
```
실행 시 데이터셋 선택 메뉴가 나타남:
1. CIFAR-10
2. Fashion-MNIST
3. STL-10
4. All three sequentially

### 출력 파일
- `best_head_{dataset}.pth` — Phase 2 최고 정확도 체크포인트
- `head_{dataset}.pth` — 최종 가중치
- `fisher_log_{dataset}.npy` — 에포크별 Fisher Ratio

---

## 🧪 하이퍼파라미터 (v2)

| 파라미터 | 값 | 비고 |
|---------|:---:|------|
| epochs | 50 | Phase 2 학습 시간 확보 |
| batch_size | 128 | — |
| embedding_dim | 128 | — |
| lr (Phase 1) | 1e-3 | cosine schedule |
| lr (Phase 2) | 2e-4 | 고정 |
| λ | 0.5 | negative energy 가중치 |
| σ | auto | median heuristic × 0.5 |
| temperature τ | 20.0 | **v2 핵심** |
| max_phase1_epochs | 15 | Phase 2 강제 전환 |

---

## 📚 12편의 참고 논문

1. Fisherfaces / LDA — scatter matrix 이론 기반
2. MFA (Marginal Fisher Analysis) — 그래프 기반 S_w/S_b
3. Isomap — manifold preserving embedding
4. LLE — 로컬 선형 재구성
5. **Contrastive Loss / SimCLR** — augmentation + deeper head ✅ 적용
6. **FaceNet / Triplet** — 거리 기반 학습의 다양성 ✅ 적용
7. Proxy-NCA — proxy 기반 NCA
8. Magnet Loss — 클러스터 기반 손실
9. EBM Tutorial — 에너지 기반 모델
10. JEM — 분류기 = EBM
11. **SphereFace** — angular margin, temperature scaling ✅ 적용
12. **Prototypical Networks** — 프로토타입 기반 logit, temperature ✅ 적용

---

## 📜 버전 관리

- **v0** (원본, JASorianoHernandez): 74.53%
- **v1.1** (L2 normalize): 53.94% — 실패, 원복
- **v2** (현재): 89.09% — **채택**

각 버전의 상세 변경 내역은 `CHANGELOG.md` 참조.

---

## 📝 라이선스

연구용 참조 구현.
