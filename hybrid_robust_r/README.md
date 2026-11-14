# hybrid_robust_r — hybrid_epair + MLP 센서 입력의 중앙값·IQR 정규화

> 이전 최종 선택 모델 (현재 최종은 `hybrid_rewm`의 변형 REW). hybrid_epair(Public 0.9476)에서 MLP의 센서 입력 정규화만 바꿨다. Public **0.9526954937**.

## 모델 구성

이미지와 센서를 모두 쓰는 두 모델의 확률을 50:50으로 섞고, 헷갈리기 쉬운 인접 단계 쌍만 센서 전용 보조 분류기로 다듬는다.

1. **Fusion MLP** (seed 42/43/44 평균, 각 16 epoch): 고정 ResNet-18 이미지 임베딩 512차원 + 현재 행의 센서 9개
2. **HistGradientBoosting 트리**: 센서 피처 46개 + 이미지 임베딩의 PCA 32
3. **쌍 보조기**: 단계 (3,4), (5,6) 각각에 대해 센서 46개만 읽는 이진 트리. 1·2를 섞은 확률의 상위 2개가 그 쌍일 때만, 두 단계 확률의 합은 그대로 두고 내부 비율을 `0.5 × 기존 + 0.5 × 보조기`로 바꾼다.

## 추가된 부분 (hybrid_epair 대비)

- MLP의 현재 센서 9개 입력을 **Train 중앙값과 사분위 범위(IQR)**로 정규화하고 ±20으로 자른다. 이전에는 평균·표준편차였다.
  - 결측과 비유한값은 Train 중앙값으로 채우고, IQR이 1e−8보다 작은 열은 1로 나눈다.
- 근거 (Train 분석): 전압에서 표준편차의 20배를 넘는 관측 6개가 전압 전체 제곱편차의 15%를 차지했다. 평균·표준편차 정규화는 이런 값에 끌려가 나머지 대부분의 값을 좁은 범위로 눌러 버린다. 극단값을 측정 오류로 보고 지우지는 않았다.
- 이미지 정규화, 트리 입력과 전처리, PCA, 쌍 보조기, 혼합비는 hybrid_epair와 같다.

## 시계열 피처 (트리와 쌍 보조기의 센서 46개)

모두 **같은 실험의 현재 시각 이하 행만** 사용한다. 미래 행·다른 실험·Test 전체 통계는 쓰지 않는다. 행 순서를 섞거나, 실험을 나눠 넣거나, 미래 행을 빼도 값이 같다.

| 묶음 | 개수 | 내용 |
| --- | --- | --- |
| 현재 값 | 9 | 전압, 전류, t1, t2, 압력, t2−t1, 온도 평균, 온도 최댓값, 전압×전류 |
| 30초 창 | 15 | 센서 5개의 최근 30초 평균·최댓값·30초 전 대비 변화 (주최 측 베이스라인 v7) |
| 누적 극값 | 7 | t1·t2·압력의 누적 최댓값, 전압의 누적 최솟값, 현재 값과 그 극값의 차이 |
| 누적 사건 | 15 | 전압·t1·t2·압력·thermal_t의 최고값 대비 최대 하락폭·하락률, 최저값 대비 최대 상승폭 |

## 학습·검증 구성

- 학습: 제공 Train (78 실험, 33,522행)만 사용. 제공 Validation (19 실험)은 학습에 넣지 않았다.
- 모델 비교: Train을 **실험 단위**로 나눈 5-fold를 서로 다른 seed로 2번 (10 fold). 같은 실험이 학습과 검증에 함께 들어가지 않는다. 실험마다 처음 쓰는 분할 seed를 썼다.
- 전처리·PCA·모델은 fold의 학습 부분에서만 fit했다.
- 제공 Validation은 여러 번 참고했으므로 독립적인 검증 결과로 보지 않는다.

## 결과

| 모델 | 실험 단위 10-fold 평균 (같은 분할) | 제공 Validation | Public |
| --- | --- | --- | --- |
| hybrid_epair | 0.897248 | 0.764126 | 0.9476357394 |
| **hybrid_robust_r** | 0.898484 (+0.0012, 6/10) | **0.769399** | **0.9526954937** |

- 다른 분할 두 번에서는 hybrid_epair와 차이가 −0.0007, −0.0004였다. **교차검증으로는 두 모델을 구분할 수 없다.** 차이는 제공 Validation(+0.0053)과 Public(+0.0051)에서 나왔다.
- 제공 Validation 클래스 F1 (1~6): .9990 / .6880 / .8627 / .5447 / .6513 / .8707
- 공식 `check_submission.py` 전 항목 통과
  - 반복 실행, 최종 파일 재현, 행 순서 섞기, 실험 나눠 넣기, 미래 행 제거 모두 0.00% 차이
  - 모든 이미지를 검은 이미지로 바꾸면 23.18%의 예측이 바뀜
- 제출 파일 `submission_robust_r.csv`, SHA256 `937fe4746739eeddd97e97ee282fe9b4517c1f10046d9239064da6b9185aa313`

## 선택 경위 (사실 기록)

- 사전에 정한 교체 기준은 "실험 단위 10-fold에서 +0.003 이상, 7/10 fold 이상 개선"이었고, 이 모델은 **통과하지 못했다** (+0.0012, 6/10).
- 작지만 일관된 신호(두 번의 OOF 상승, 제공 Validation 상승)를 근거로 Public에 한 번 제출해 확인했고, 그 결과를 함께 보고 최종으로 골랐다. 사전 기준 통과가 아니라 사후에 종합한 선택이며, 선택 편향과 Private 불확실성이 있다.
- 같은 방향의 후속 후보는 모두 채택하지 않았다.
  - 전압만 중앙값·IQR로 바꾼 변형, 두 MLP를 1:1로 섞은 변형: 교차검증 +0.0007, 제공 Validation 하락
  - 60초 창 피처와 넓은 센서 MLP를 더한 변형: 교차검증 +0.0008 (8/10), 제공 Validation 0.7740으로 상승, **Public 0.9479로 하락**

## 실행

```bash
# torch/torchvision은 운영진 설치본을 유지하고, scikit-learn은 1.7.2를 쓴다
python train.py --data_dir ~/data/battery_data_light --output_dir weights
python test.py --input ~/data/battery_data_light/test/test.csv --image_root ~/data/battery_data_light/test --output submission.csv
```

- `test.py`는 `weights/model.pt`, `weights/tree.joblib`, `weights/manifest.json`을 읽어 추론만 한다. 재학습이나 다운로드를 하지 않는다. 세 파일은 한 세트다.
- 가중치는 이 폴더에 넣지 않는다. 받는 곳은 레포 최상위 README를 본다.
- 제출한 가중치는 `train.py`와 같은 레시피로 학습한 실험 산출물을 `export_r.py`로 변환한 것이다 (재학습 없음, Validation 확률 최대 차이 0). `train.py`로 새로 학습하면 GPU 연산 순서 차이로 값이 아주 조금 달라질 수 있으므로, 제출 결과의 재현은 제공한 가중치로 한다.
- 점검: `python test_synthetic.py` (지어낸 데이터로 전 과정), `python check_encoding.py`, 공식 `python check_submission.py --data_dir ~/data/battery_data_light --final <제출 CSV>`

## 파일

| 파일 | 역할 |
| --- | --- |
| `model.py` | 이 모델의 학습·추론 진입점 (중앙값·IQR 정규화, 형식 검증) |
| `base_epair.py` | hybrid_epair 본체 (피처 생성, 트리, 쌍 보조기, 추론). hybrid_epair의 `model.py`와 같은 파일 |
| `base_model.py` | 이미지 인코더, Fusion MLP, CSV 읽기 |
| `baseline_v7.py` | 주최 측 베이스라인 v7 (30초 창 피처) |
| `mlp_training.py`, `robust_sensor_scaling.py` | MLP 학습 함수, 정규화 함수 |
| `train.py`, `test.py` | 학습, 추론 |
| `export_r.py`, `embedding_cache.py`, `run_prepare_submission.sh` | 실험 산출물을 제출용 가중치로 변환할 때 쓴 스크립트 |
| `test_synthetic.py`, `check_encoding.py`, `check_ties.py` | 자체 검사 |

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
