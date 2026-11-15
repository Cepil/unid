# Uni-DTHON — 배터리 열폭주 단계 분류

열화상 이미지와 센서 시계열로 배터리 열폭주의 6단계를 분류한다 (평가 지표 Macro F1).

## 최종 모델

**레포 최상위의 `model.py` · `train.py` · `test.py`** (모델 이름 hybrid_rewm, 변형 REW) — Public **0.9548552368**, 제공 Validation 0.769316.

- 제출 파일: `submission_rew.csv` (SHA256 앞 16자 `a0dafe3f05c118f3`)
- 가중치: 레포 최상위 [`weights/`](weights/)에 들어 있다 — `model.pt` (45.7MB), `tree.joblib` (3.7MB), `manifest.json` 한 세트. `manifest.json`에 두 파일의 SHA256과 학습 환경 버전이 적혀 있고, `test.py`는 실행할 때 이 해시를 확인한다.
- 대회 데이터는 이 레포에 포함하지 않는다.

### 추론 (재학습 없이 제출 결과 재현)

```bash
pip install -r requirements.txt        # torch / torchvision은 설치된 GPU 빌드를 그대로 쓴다
python test.py --input <test.csv> --image_root <이미지 폴더> --output submission.csv
```

`test.py`는 `weights/`의 가중치만 읽는다. 입력 CSV의 행만 사용하며, 각 행의 예측은 같은 실험의 그 시각 이하 행에만 의존한다. 행 순서를 섞거나, 실험을 나눠 넣거나, 미래 행을 빼도 같은 샘플의 예측은 같다 (공식 `check_submission.py` 전 항목 통과).

### 학습

```bash
python train.py --data_dir <battery_data_light> --output_dir weights
```

제공 Train만 사용하고, 제공 Validation은 학습 뒤 한 번 평가만 한다. 기본 변형이 최종 모델인 REW다 (`--variant REWM`은 채택하지 않은 비교용 변형).

## 모델 한눈에 보기

1. **이미지**: 고정(frozen) ResNet-18 (ImageNet 사전학습)의 512차원 임베딩
2. **Fusion MLP**: 이미지 임베딩 + 현재 센서 9개 (Train 중앙값·IQR 정규화), seed 3개 평균
3. **트리 (HistGradientBoosting)**: 센서 피처 61개 + 이미지 임베딩 PCA 32
   - 현재 값 9, 30초 창 15, 60초 창 15, 누적 극값 7, 누적 사건(최고값 대비 하락폭 등) 15
   - 모두 같은 실험의 현재 시각 이하 행만 사용
4. 2와 3의 확률을 50:50으로 섞은 뒤, 헷갈리기 쉬운 단계 쌍 (3,4)·(5,6)만 센서 전용 보조 분류기로 다듬는다

자세한 구성, 시계열 피처 정의, 검증 방법, 선택 경위는 [`hybrid_rewm/README.md`](hybrid_rewm/README.md)에 있다.

## 폴더

모델이 발전한 순서대로 남겨 두었다. 각 폴더는 혼자서 학습·추론할 수 있고, README에 그 단계에서 더한 것과 근거, 결과를 적었다.

| 폴더 | 더한 것 | Public |
| --- | --- | --- |
| `hybrid_bs/` | 이미지·센서 하이브리드 + 주최 측 베이스라인의 30초 창 피처 | 0.9384 |
| `hybrid_cum/` | 누적 극값 피처 | 0.9447 |
| `hybrid_pair/` | 인접 단계 쌍 보조 분류기 | 0.9455 |
| `hybrid_epair/` | 누적 사건 피처 | 0.9476 |
| `hybrid_robust_r/` | MLP 센서 입력의 중앙값·IQR 정규화 | 0.9527 |
| **`hybrid_rewm/`** | **60초 창 피처 (최종, 변형 REW)** | **0.9549** |

최상위의 `model.py` · `base_model.py` · `baseline_v7.py` · `test.py`는 `hybrid_rewm/`의 파일과 같다. `train.py`는 기본 변형만 REW로 바꿨다 (`hybrid_rewm/train.py`는 기본값이 REWM이라 `--variant REW`를 붙여야 한다).

## 환경과 출처

- 대회 제공 서버: Linux, Python 3.10, NVIDIA RTX 5080 16GB, torch 2.11.0+cu128, scikit-learn 1.7.2
- 사전학습: torchvision `ResNet18_Weights.IMAGENET1K_V1` (https://download.pytorch.org/models/resnet18-f37072fd.pth, 코드 라이선스 BSD-3-Clause)
- 시계열 창 피처: 주최 측 베이스라인 v7의 함수 (`baseline_v7.py`)
