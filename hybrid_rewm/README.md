# hybrid_rewm (변형 REW) — hybrid_robust_r + 60초 창 피처

> **팀 최종 선택 모델 (변형 REW).** hybrid_robust_r(Public 0.9527)의 트리와 쌍 보조기에 60초 창 피처 15개를 더했다. Public **0.9548552368**.
>
> 이 폴더의 코드는 두 변형을 지원한다. **최종은 `--variant REW`**다. 기본값인 REWM은 채택하지 않았다 (아래 참고).

## 모델 구성

이미지와 센서를 모두 쓰는 두 모델의 확률을 50:50으로 섞고, 헷갈리기 쉬운 인접 단계 쌍만 센서 전용 보조 분류기로 다듬는다.

1. **Fusion MLP** (seed 42/43/44 평균, 각 16 epoch): 고정 ResNet-18 이미지 임베딩 512차원 + 현재 행의 센서 9개. 센서 입력은 Train 중앙값·사분위 범위(IQR)로 정규화하고 ±20으로 자른다.
2. **HistGradientBoosting 트리**: 센서 피처 61개 + 이미지 임베딩의 PCA 32
3. **쌍 보조기**: 단계 (3,4), (5,6) 각각에 대해 센서 61개만 읽는 이진 트리. 1·2를 섞은 확률의 상위 2개가 그 쌍일 때만, 두 단계 확률의 합은 그대로 두고 내부 비율을 `0.5 × 기존 + 0.5 × 보조기`로 바꾼다.

## 추가된 부분 (hybrid_robust_r 대비)

- 트리와 쌍 보조기 입력에 **60초 창 피처 15개**를 더한다 (센서 5개의 최근 60초 평균·최댓값·60초 전 대비 변화, 주최 측 베이스라인 v7의 함수에 창 길이만 60초로 지정). 센서 입력 46 → 61.
- 근거: 30초 창만으로는 느리게 진행하는 단계(가열·팽창)의 변화가 작게 보인다. 더 긴 창은 이전 실험에서도 누적 극값 모델 위에서 교차검증 +0.0028 (8/10)이었다.
- MLP, 이미지 정규화, PCA, 혼합비는 hybrid_robust_r와 같다.

## 시계열 피처 (트리와 쌍 보조기의 센서 61개)

모두 **같은 실험의 현재 시각 이하 행만** 사용한다. 미래 행·다른 실험·Test 전체 통계는 쓰지 않는다. 행 순서를 섞거나, 실험을 나눠 넣거나, 미래 행을 빼도 값이 같다.

| 묶음 | 개수 | 내용 |
| --- | --- | --- |
| 현재 값 | 9 | 전압, 전류, t1, t2, 압력, t2−t1, 온도 평균, 온도 최댓값, 전압×전류 |
| 30초 창 | 15 | 센서 5개의 최근 30초 평균·최댓값·30초 전 대비 변화 (주최 측 베이스라인 v7) |
| 누적 극값 | 7 | t1·t2·압력의 누적 최댓값, 전압의 누적 최솟값, 현재 값과 그 극값의 차이 |
| 누적 사건 | 15 | 전압·t1·t2·압력·thermal_t의 최고값 대비 최대 하락폭·하락률, 최저값 대비 최대 상승폭 |
| 60초 창 | 15 | 센서 5개의 최근 60초 평균·최댓값·60초 전 대비 변화 |

## 학습·검증 구성

- 학습: 제공 Train (78 실험, 33,522행)만 사용. 제공 Validation (19 실험)은 학습에 넣지 않았다.
- 모델 비교: Train을 **실험 단위**로 나눈 5-fold를 서로 다른 seed로 2번 (10 fold). 같은 실험이 학습과 검증에 함께 들어가지 않는다.
- 전처리·PCA·모델은 fold의 학습 부분에서만 fit했다.
- 제공 Validation은 여러 번 참고했으므로 독립적인 검증 결과로 보지 않는다.

## 결과

| 모델 | 실험 단위 10-fold 평균 (같은 분할, exp_t39) | 제공 Validation | Public |
| --- | --- | --- | --- |
| hybrid_epair | 0.899326 | 0.764126 | 0.9476357394 |
| hybrid_robust_r | 0.898908 | 0.769399 | 0.9526954937 |
| **REW (최종)** | 0.899310 (hybrid_robust_r 대비 +0.0004, 6/10) | **0.769316** | **0.9548552368** |
| REWM | 0.899728 (+0.0008, 8/10) | 0.773971 | 0.9478585685 |

- **교차검증으로는 네 모델을 구분할 수 없다** (차이 0.001 이내). 사전에 정한 교체 기준(+0.003, 7/10 fold)은 REW도 REWM도 통과하지 못했다.
- 제공 Validation 클래스 F1 (1~6, REW): .9996 / .6916 / .8646 / .5444 / .6461 / .8696
- 공식 `check_submission.py` 전 항목 통과 (REW)
  - 반복 실행, 최종 파일 재현, 행 순서 섞기, 실험 나눠 넣기, 미래 행 제거 모두 0.00% 차이
  - 모든 이미지를 검은 이미지로 바꾸면 23.53%의 예측이 바뀜
- 제출 파일 `submission_rew.csv`, SHA256 앞 16자 `a0dafe3f05c118f3`. Test 예측은 hybrid_robust_r와 14행이 다르다.

## 선택 경위 (사실 기록)

- 실험 전에 "정식 기준에 못 미쳐도 hybrid_robust_r 대비 평균이 양수이고 6/10 fold 이상 이기며 Validation이 0.002 넘게 떨어지지 않으면 Public에 한 번 내 본다"는 탐색 기준을 정했고, REW와 REWM 모두 여기에 들었다. 두 후보의 차이가 0.001 이내면 단순한 REW를 고른다는 규칙도 미리 정했다.
- REWM은 교차검증과 Validation이 더 좋았지만 Public이 0.9479로 내려갔다. REW는 Public이 0.9549로 올랐다. 이 결과를 함께 보고 REW를 최종으로 골랐다.
- 사전 기준 통과가 아니라 사후에 종합한 선택이다. Test 14행의 차이에서 나온 결과이므로 선택 편향과 Private 불확실성이 있다.

## REWM 변형 (채택하지 않음)

REW에 넓은 Fusion MLP(이미지 임베딩 + 트리 센서 입력 61개)를 소량 섞는다: `mixed = 0.75 × (0.5·MLP9 + 0.5·트리) + 0.25 × (0.5·MLP61 + 0.5·트리)`. 코드는 비교 실험의 재현을 위해 남겨 두었다.

## 실행

```bash
# torch/torchvision은 운영진 설치본을 유지하고, scikit-learn은 1.7.2를 쓴다
python train.py --data_dir ~/data/battery_data_light --output_dir weights --variant REW
python test.py --input ~/data/battery_data_light/test/test.csv --image_root ~/data/battery_data_light/test --output submission.csv
```

- **학습할 때 `--variant REW`를 꼭 지정한다.** 생략하면 REWM이 학습된다. 변형은 가중치에 저장되므로 `test.py`에는 지정하지 않는다.
- `test.py`는 `weights/model.pt`, `weights/tree.joblib`, `weights/manifest.json`을 읽어 추론만 한다. 재학습이나 다운로드를 하지 않는다. 세 파일은 한 세트다.
- 가중치는 이 폴더에 넣지 않는다. 받는 곳은 레포 최상위 README를 본다.
- 점검: `python test_synthetic.py REW` (지어낸 데이터로 전 과정), `python check_ties.py --input <csv>`, 공식 `python check_submission.py --data_dir ~/data/battery_data_light --final <제출 CSV>`

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
