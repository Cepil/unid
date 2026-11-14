# Hybrid v2 + 베이스라인 시계열 피처 (hybrid_bs) — 배터리 열폭주 6단계 분류

> 현재 팀 최종 후보. 대회 서버에서 Train만으로 학습했고, 공식 `check_submission.py`를 모두 통과했다. Public **0.9384348938**.

## 모델 요약

이미지와 센서를 모두 쓰는 두 모델의 확률을 **50:50으로 평균**한다.

1. **Fusion MLP** (seed 42/43/44 평균)
   - 입력: 고정 ResNet-18(ImageNet) 512차원 이미지 임베딩 + 현재 행 센서 9개
   - 구성: image 512→128, sensor 9→32, 160→64→6. ReLU, dropout .2
   - 학습: AdamW lr 1e-3 / weight decay 1e-3, batch 256, 16 epoch 고정, 가중치 없는 cross entropy
   - 센서 9개: V, I, t1, t2, p, t2−t1, 온도 평균, 온도 최댓값, V×I
2. **HistGradientBoosting 트리**
   - 입력: 현재 행 센서 9개 + **과거 30초 시계열 피처 15개** + 이미지 임베딩 PCA 32
   - 하이퍼파라미터: learning_rate .06, max_iter 250, max_leaf_nodes 15, min_samples_leaf 32, l2 5, early_stopping 끔, random_state 42

- 결측 대체, 표준화, 이미지 표준화, PCA는 모두 Train에서만 fit해서 저장한다. 추론에서는 저장된 값을 적용만 한다.

## 시계열 피처 구성

- 주최 측 제공 베이스라인 v7의 `add_causal_features`를 그대로 사용한다 (`baseline_v7.py`, 원본 `model.py`와 동일한 파일).
- 같은 `experiment_id` 안에서 `relative_time_s` 기준으로 계산한다. 센서 5개(V, I, t1, t2, p) 각각에 대해 세 가지를 만든다.
  - (t−30초, t] 구간의 평균
  - 같은 구간의 최댓값
  - t−30초 이하 마지막 관측과의 차이
- 현재 이후 시점의 관측, 다른 실험의 관측, 실험 전체 통계는 쓰지 않는다.
- 행 순서 변경, 실험 단위 분리, 미래 행 제거에서는 같은 예측을 내도록 설계했다. 과거 행이 빠지면 달라질 수 있다.
- 같은 실험 안에 같은 시각의 행이 여러 개 있으면 window 계산이 행 순서에 영향을 받을 수 있다. 제공 Train·Validation·Test에는 그런 행이 없음을 `check_ties.py`로 확인했다.
- 공식 점검 결과: 형식, 반복 실행, 최종 파일 재현, 행 순서 섞기, 실험 나눠 넣기, 미래 행 제거 모두 0.00% 차이. 모든 이미지를 검은 이미지로 바꾸면 45.20%의 예측이 바뀐다.

## 학습·검증 구성

- 최종 학습: 제공 **Train만** 사용. 제공 Validation은 학습이 끝난 뒤 평가하는 데만 썼다.
- 모델 선택: Train 안에서 `experiment_id` 단위로 나눈 교차검증.
  - 사전에 정한 기준으로, 처음 쓰는 무작위 실험 분할 2회 × 5-fold에서 기존 Hybrid v2와 비교했다.

| 모델 | 10-fold 평균 | 최저 fold | 제공 Validation |
| --- | --- | --- | --- |
| Hybrid v2 | 0.8553 | 0.7799 | 0.7307 |
| hybrid_bs (본 모델) | 0.8726 | 0.8008 | 0.7500 |

- 최종 학습 (대회 서버, Train만): 제공 Validation **0.751682**. 클래스 F1: .9984 / .6977 / .8560 / .5366 / .5937 / .8277.
- Public: Hybrid v2 서버 재현본 0.8984965037 → **hybrid_bs 0.9384348938**.

- 10개 fold 모두 개선됐다 (평균 +0.0173).
- 이미지 밝기 통계와 그 시계열 피처를 추가한 변형도 시험했다.
  - 주 평가(새 분할)에서는 본 모델보다 낮았다 (0.8714). 보조 평가(기본 GroupKFold)에서는 약간 높았다 (0.8778 vs 0.8761).
  - 사전에 정한 선택 규칙에 따라 더 단순한 본 모델을 선택했다.

## 실행 방법

```bash
# torch/torchvision은 운영진 설치본을 유지하고, scikit-learn은 1.7.2를 쓴다
python train.py --data_dir ~/data/battery_data_light --output_dir weights
python test.py --input ~/data/battery_data_light/test/test.csv --image_root ~/data/battery_data_light/test --output submission.csv
```

- 가중치
  - `weights/model.pt`: encoder, MLP 3개, 전처리 통계, 이미지 표준화 값
  - `weights/tree.joblib`: PCA, 트리
  - `weights/manifest.json`: 파일 해시
  - 가중치는 레포에 넣지 않고 팀에 따로 전달한다 (`hybrid_bs_weights.tar.gz`, 설치 방법은 TEAM_HANDOFF.md). 최종 공개 레포에서는 다운로드 링크 또는 가중치 파일을 추가할 예정 (TODO).
- 점검
  - `python test_synthetic.py`: 가짜 데이터로 학습·추론과 일관성 검사
  - `python check_ties.py --input <csv>`: 같은 시각 행 집계
  - 공식 점검: `python check_submission.py --data_dir ~/data/battery_data_light --final <제출 CSV>`

## 개발 환경

- 대회 제공 서버: Linux, Python 3.10, NVIDIA RTX 5080 16GB
- torch 2.11.0+cu128, scikit-learn 1.7.2, numpy 2.2.6 (서버에서 확인한 값)
- 정확한 전체 버전은 `weights/manifest.json`의 `environment_versions`에 기록된다

## 사전학습 모델과 출처

- torchvision `ResNet18_Weights.IMAGENET1K_V1`, frozen
  - 가중치: https://download.pytorch.org/models/resnet18-f37072fd.pth
  - 문서: https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.resnet18.html
  - 코드 라이선스: torchvision BSD-3-Clause
- 시계열 피처 코드: 주최 측 제공 베이스라인 v7 (`baseline_v7.py`)

## 데이터

- 대회 데이터는 이 레포에 포함하지 않는다.
