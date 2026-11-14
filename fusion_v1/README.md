# Fusion v1 — Public 0.8504558014

열화상 이미지와 현재 센서값을 결합한 Uni-DTHON 6단계 분류 모델입니다. 이 폴더는 해당 Public 제출에 사용한 코드 버전입니다. 원본 데이터, 모델 체크포인트, 제출 CSV, 데이터가 포함된 로그 및 fold 매핑은 공개 저장소에 포함하지 않습니다. 실제 체크포인트와 제출 원본은 팀 내부 전달 ZIP으로 공유합니다.

## 결과

| 평가 | Fusion v1 | 베이스라인 |
| --- | --- | --- |
| Public | 0.8504558014 | 0.8010471797 |
| 제공 Validation Macro F1 | 0.778443 | 팀원 보고 약 0.677 |
| Train 5-fold GroupKFold 평균 | 0.815681 | none 0.805236 / balanced 0.807152 |
| 전체 OOF Macro F1 | 0.817944 | none 0.808266 / balanced 0.809984 |

Public, 제공 Validation, CV는 서로 다른 평가 데이터입니다. Private 결과는 아직 없습니다. 실제 제출 모델은 Train으로 학습하고 제공 Validation으로 선택한 epoch 16 체크포인트이며, Train+Validation 재학습이나 CV 앙상블을 사용하지 않았습니다.

## 파일

- `model.py`: 이미지·센서 전처리와 모델 구조.
- `train.py`: Train 학습, 제공 Validation 기반 체크포인트 선택.
- `test.py`: 저장된 체크포인트로 추론만 수행.
- `cv.py`: Train만 사용한 실험 단위 GroupKFold 비교 평가.
- `baseline_reference.py`: 배포 baseline.zip의 원본 model.py. CV 비교에만 사용하며 최종 추론에는 필요 없습니다.
- `verify_submission.py`: 제출 형식·반복·부분 입력·행 순서 변경 검사.
- `pack_handoff.py`: 기존 코드·가중치·제출 원본·검증 자료를 팀 내부 전달용 ZIP으로 묶는 도구. 새로운 학습/예측 자료를 만들지 않습니다.

사용한 노트북·별도 설정 파일·피처 캐시는 없습니다. 이미지 특징은 실행 중 메모리에서 한 번 추출합니다.

## 모델·전처리

고정된 ResNet-18 ImageNet 특징 512차원 → Linear(512,128) + ReLU + Dropout(0.2). 센서 9차원 → Linear(9,32) + ReLU. 두 출력을 연결한 160차원 → Linear(160,64) + ReLU + Dropout(0.2) + Linear(64,6).

센서 피처 순서:

1. meas_v
2. meas_a
3. meas_t1
4. meas_t2
5. meas_p
6. meas_t2 − meas_t1
7. (meas_t1 + meas_t2) / 2
8. max(meas_t1, meas_t2)
9. meas_v × meas_a

결측·무한값은 Train 중앙값으로 대체하고 Train 평균/표준편차로 정규화한 뒤 ±20으로 제한합니다. 이미지 임베딩도 Train 평균/표준편차로 정규화합니다. 이미지 전체를 RGB·224×224로 변환하고 ImageNet mean/std를 사용하며 crop·색상 증강은 없습니다. experiment_id, relative_time_s, 과거/미래 행이나 전체 실험 통계는 추론 입력이 아닙니다.

학습 내부 클래스는 0~5, 제출 클래스는 argmax+1인 1~6입니다. 앙상블·임계값·후처리는 없습니다.

`weights/fusion.pt` 하나에 다음이 들어 있습니다:

- 고정된 ResNet-18 가중치 (`encoder`)
- 결합 분류기 가중치 (`fusion`)
- 센서 median/mean/std (`preprocessing`)
- 이미지 임베딩 mean/std (`image_mean`, `image_std`)
- seed, 선택 epoch, 검증 점수, class_weight 설정

별도 `.joblib`, 스케일러, PCA, 인코더 파일이나 사전학습 가중치 원본 파일은 추론에 필요하지 않습니다. `test.py`는 인터넷 다운로드와 전처리 재학습을 수행하지 않습니다.

## 환경

실행 환경: Linux, Python 3.10.22, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, NVIDIA RTX A2000 6GB. 실제 설치 패키지 목록은 팀 전달 ZIP에서 실행 환경 기준으로 수집합니다.

`requirements.txt`는 이 학습 환경용입니다. **대회 제공 RTX 5080 서버에서는 이 파일을 무조건 설치해 기존 GPU용 PyTorch를 바꾸지 마세요.** 운영진이 설정한 torch/torchvision 조합을 유지하고 필요한 나머지 라이브러리만 준비한 뒤 해당 환경에서 재현 검사를 수행해야 합니다. 서버 간 실행은 아직 검증하지 않았습니다.

## 학습·추론 명령

아래 명령은 GPU가 할당된 컴퓨트 노드에서 사용합니다. Seraph 마스터/로그인 노드에서는 Python 실행·설치·압축 작업을 수행하지 않습니다. 데이터는 컴퓨트 노드 로컬 저장소를 사용하고 사용자가 자원 사용을 감독합니다.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_HOME=/data/$USER/unid/torch-cache
UNID_PY=/data/$USER/venvs/unid/bin/python
UNID_DATA=/local_datasets/$USER/unid/battery_data_light

# 실제 제출 모델의 학습 명령: Train만 학습, Validation으로 선택
"$UNID_PY" -u train.py --data_dir "$UNID_DATA" \
  --batch_size 32 --workers 2 --epochs 60 --output weights/fusion.pt

# 실제 제출 CSV 생성 명령
"$UNID_PY" -u test.py --input "$UNID_DATA/test/test.csv" \
  --image_root "$UNID_DATA/test" --model weights/fusion.pt \
  --output fusion_v1_submission.csv

# GroupKFold: 각 fold에서 16 epoch 고정, 기존 체크포인트를 변경하지 않음
"$UNID_PY" -u cv.py --data_dir "$UNID_DATA" \
  --folds 5 --epochs 16 --batch_size 32 --workers 2

# 제출 형식과 일부 샘플의 추론 일관성 검사
"$UNID_PY" verify_submission.py --input "$UNID_DATA/test/test.csv" \
  --image_root "$UNID_DATA/test" --submission fusion_v1_submission.csv \
  --sample_submission "$UNID_DATA/sample_submission.csv" \
  --temp_dir /local_datasets/$USER/unid/verify
```

이미지 추출 batch size는 32, 분류기 학습 batch size는 256입니다. AdamW(lr=0.001, weight_decay=0.001), 기본 class_weight 없음, seed 42. Validation Macro F1으로 best checkpoint를 선택하며 patience 12입니다. 실제 실행은 epoch 28에 종료되어 best epoch 16을 저장했습니다.

최종 모델은 원본 `.pt`를 그대로 사용해야 합니다. 재학습한 모델·CV fold 모델·다른 epoch를 같은 제출 모델로 대체하지 마세요. 원본 제출 파일명은 `fusion_v1_submission.csv`입니다. 선택된 최대 64개 샘플의 부분 입력·반복·역순 및 배치 크기 변경 검사와 전체 제출 형식 검사를 통과했습니다. 모든 부분집합 또는 다른 하드웨어의 동일 결과를 보장하는 검사는 아닙니다.

## 사전학습 출처

torchvision 0.20.1 `ResNet18_Weights.IMAGENET1K_V1`.

- 문서: https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.resnet18.html
- 원본 가중치: https://download.pytorch.org/models/resnet18-f37072fd.pth
- torchvision 코드 라이선스: BSD-3-Clause, https://github.com/pytorch/vision/blob/v0.20.1/LICENSE

코드 라이선스와 ImageNet/가중치의 이용 조건은 구분해야 합니다. torchvision 코드 라이선스만으로 데이터·가중치의 별도 라이선스를 확정하지 않으며 최종 제출 전 해당 이용 조건을 확인해야 합니다.

## 팀 전달 ZIP

최종 체크포인트와 실제 제출 CSV는 필수, 기존 로그·CV metrics·fold 매핑은 존재하는 경우만 포함합니다. 별도로 저장하지 않은 Validation 예측/확률·혼동행렬·CV 모델은 새로 생성하지 않습니다. 각 파일 SHA256과 크기를 manifest에 기록합니다. 데이터·비밀번호·토큰·가상환경·피처 캐시는 포함하지 않으며 ZIP은 공개 GitHub에 올리지 않습니다.
