# hybrid_pair — hybrid_cum + 센서 전용 인접 단계 보조 분류기

> **팀 최종 선택 모델.** hybrid_cum(Public 0.9447)에 센서 전용 인접 단계 보조 분류기 2개를 더했다. Public **0.9454656711**.

## 추가된 부분

- hybrid_cum 확률을 만든 뒤, 단계 쌍 (3,4)와 (5,6)에 보조 분류기를 하나씩 적용한다.
  - 보조기: 해당 두 단계의 Train 행만으로 학습한 이진 HistGradientBoosting. 하이퍼파라미터는 본 트리와 같다.
  - 입력: 본 트리의 **센서 피처 31개만** 사용한다 (이미지 PCA 제외).
  - 적용 조건: hybrid_cum의 상위 2개 클래스가 정확히 그 쌍인 행만. 동률이면 번호가 작은 클래스가 먼저 선택된다.
  - 조정: 두 단계 확률의 합은 그대로 두고, 내부 비율만 `0.5 × 기존 비율 + 0.5 × 보조기 확률`로 바꾼다. 나머지 단계 확률은 바뀌지 않는다.
- 근거: 처음 쓰는 실험 분할 10-fold (exp_t20)에서 hybrid_cum 대비 **+0.0098, 9/10 fold 개선** (사전 선언 기준 통과). 제공 Validation 0.7617 → 0.7630. 학습에 없던 해상도 그룹으로 나눈 진단 CV에서도 +0.023.
- 남은 오류가 가장 많던 4→3, 6→5 혼동을 직접 겨냥한다 (OOF 4→3 0.125→0.089, 6→5 0.109→0.071).

## 결과

| 모델 | 처음 쓰는 분할 10-fold (hybrid_cum 대비) | 제공 Validation | Public |
| --- | --- | --- | --- |
| hybrid_cum | 기준 | 0.761662 | 0.9447186465 |
| **hybrid_pair** | +0.0098 (9/10, exp_t20), +0.0141 (10/10, exp_t21), +0.0099 (9/10, exp_t22) | **0.761793** | **0.9454656711** |

- 최종 학습: 대회 서버에서 Train만 사용. 클래스 F1: .9997 / .7034 / .8650 / .5279 / .6167 / .8582.
- 공식 `check_submission.py` 전 항목 통과
  - 반복 실행, 최종 파일 재현, 행 순서 섞기, 실험 나눠 넣기, 미래 행 제거 모두 0.00% 차이
  - 모든 이미지를 검은 이미지로 바꾸면 39.06%의 예측이 바뀜
- Test 예측은 hybrid_cum과 19행만 다르며, 모두 3↔4, 5↔6 안에서만 바뀐다.

## 선택 경위 (사실 기록)

- 팀의 사전 교체 규칙은 "Public이 현재 후보보다 +0.005 이상일 때만 교체"였다. hybrid_pair(+0.0007)는 이 규칙을 통과하지 못했다.
- 최종 선택에서는 반복된 실험 분할 결과, 구조의 단순함, 외부 평가(Public·Validation에서 손해 없음)를 종합해 **사후에 정책을 바꿔** hybrid_pair를 골랐다. 선택 편향과 Private 불확실성이 있음을 인정한다.
- 이미지 분기를 바꾼 후보(비율 유지 보기, layer4 미세조정)는 실험 분할 CV에서는 이겼지만 Public에서 하락해 채택하지 않았다.

---

(이하 hybrid_cum README)

# hybrid_cum — Hybrid v2 + 베이스라인 30초 피처 + 누적 극값 피처

> 이전 후보. hybrid_bs(Public 0.9384)에서 트리 입력에 누적 극값 피처 7개를 더했다. Public **0.9447186465**.

## 구성

- MLP: hybrid_bs와 같다. 고정 ResNet-18 512차원 + 현재 센서 9개, seed 3개 평균.
- 트리 입력
  - 현재 센서 9개
  - 주최 측 베이스라인 v7의 30초 window 피처 15개
  - **누적 극값 피처 7개**
  - 이미지 PCA 32
- 최종 확률 = MLP 0.5 + 트리 0.5

## 누적 극값 피처 (같은 실험, 첫 관측부터 현재 시각까지)

- t1·t2·압력의 누적 최댓값, 전압의 누적 최솟값
- t1 − 누적최대 t1, t2 − 누적최대 t2, 전압 − 누적최소 전압
- 같은 시각의 행은 묶어서 집계한다. 미래 행이나 다른 실험은 쓰지 않는다. 행 순서 변경, 실험 분리, 미래 행 제거에서 값이 같다.

## 설계 근거와 경위 (사실 기록)

- 근거
  - hybrid_bs의 OOF 혼동행렬: 5→6 14%, 6→5 14%, 2→3 15%, 4→3 12%
  - 팀원이 코드로 분석한 Train·Validation 인사이트: 단계가 되돌아가지 않음, 누적 최댓값·최솟값이 단계 전환의 안정적인 신호
- 검증: 처음 쓰는 실험 분할 10-fold에서 hybrid_bs 대비 +0.0113, 10/10 개선 (사전 선언 기준 통과).
- 경위
  - 온도 차이 피처 2개는, 이전에 사용 보류한 팀 피처(과거 최고 온도 대비 하락량)와 부호만 반대인 같은 정의다.
  - 이 코드는 새로 작성했다. 보류 코드를 import하거나 복사하지 않았고, 데이터는 보지 않았다.
  - 결과 확인 전에는 참고용으로 분류했으나, 결과 확인 후 팀 판단으로 후보에 포함했다.

## 결과

| 모델 | 새 분할 10-fold 개선 | 제공 Validation | Public |
| --- | --- | --- | --- |
| Hybrid v2 (서버 재현) | 기준 | 0.7317 | 0.8985 |
| hybrid_bs | +0.0173 (10/10) | 0.7517 | 0.9384 |
| **hybrid_cum** | +0.0113 (10/10, hybrid_bs 대비) | **0.761662** | **0.9447186465** |

- 최종 학습: 대회 서버에서 Train만 사용. 클래스 F1: .9997 / .7034 / .8581 / .5450 / .6156 / .8483.
- 공식 `check_submission.py` 전 항목 통과
  - 반복 실행, 최종 파일 재현, 행 순서 섞기, 실험 나눠 넣기, 미래 행 제거 모두 0.00% 차이
  - 모든 이미지를 검은 이미지로 바꾸면 39.15%의 예측이 바뀜
- 누적값 4개만 쓴 변형(차이 피처 제외)은 +0.0033 (7/10)으로 사전 기준에 미달했다. 이득은 주로 "현재값이 누적 극값에서 얼마나 떨어졌는가"에서 나온다.

## 실행

```bash
# torch/torchvision은 운영진 설치본을 유지하고, scikit-learn은 1.7.2를 쓴다
python train.py --data_dir ~/data/battery_data_light --output_dir weights
python test.py --input ~/data/battery_data_light/test/test.csv --image_root ~/data/battery_data_light/test --output submission.csv
```

- 가중치(`weights/model.pt`, `weights/tree.joblib`, `weights/manifest.json`)는 레포에 넣지 않고 팀에 따로 전달한다. 설치 방법은 TEAM_HANDOFF.md 참고.
- 점검: `python test_synthetic.py`, `python check_ties.py --input <csv>`, 공식 `python check_submission.py --data_dir ~/data/battery_data_light --final <제출 CSV>`

## 개발 환경

- 대회 제공 서버: Linux, Python 3.10, NVIDIA RTX 5080 16GB
- torch 2.11.0+cu128, scikit-learn 1.7.2, numpy 2.2.6
- 정확한 전체 버전은 `weights/manifest.json`에 기록된다

## 사전학습 모델과 출처

- torchvision `ResNet18_Weights.IMAGENET1K_V1`, frozen
  - 가중치: https://download.pytorch.org/models/resnet18-f37072fd.pth
  - 코드 라이선스: torchvision BSD-3-Clause
- 시계열 window 피처: 주최 측 베이스라인 v7 (`baseline_v7.py`, 원본 `model.py`와 동일)
- 대회 데이터는 이 레포에 포함하지 않는다.
