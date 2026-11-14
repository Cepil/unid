# temporal_tree 모델 팀 전달 — Public 0.9401

현재 리더보드에서 확인한 점수는 **Public Macro F1 0.9401**이고, 확인 당시 13팀은 1위였습니다. 표시값은 소수 넷째 자리로 반올림된 값이며, 원본 제출 점수의 전체 자릿수나 Private 성능은 아직 기록하지 않았습니다.

모델 이름: `temporal_tree`, 코드 폴더: `hybrid_v2_temporal`. 이전 팀원 Hybrid와 구분해 보관하세요. `format_version=3` 체크포인트를 이 폴더의 코드와 함께 사용합니다.

## 결과와 선택 이유

같은 실행에서 Train 실험 단위 5-fold GroupKFold로 비교한 사용자 제공 집계 결과:

| 후보 | fold 평균 | 전체 OOF Macro F1 | 최저 fold | 기존 Hybrid 대비 개선 |
|---|---:|---:|---:|---:|
| hybrid_control | 0.857105 | 0.859539 | 0.780038 | 기준 |
| temporal_tree | **0.878267** | **0.880200** | 0.803915 | **5/5 fold** |
| temporal_both | 0.868185 | 0.865354 | **0.810256** | 4/5 fold |

`temporal_tree`를 선택해 제공 Train 전체로 16 epoch 고정 학습했습니다. 제공 Validation은 학습에 쓰지 않았고 학습 뒤 평가했습니다. Validation Macro F1 **0.751640**, 단계 1~6 F1은 **0.9974, 0.6934, 0.8561, 0.5203, 0.5875, 0.8551**입니다. 최종 CSV는 해당 가중치를 사용한 `test.py` 결과이며 사용자가 입력 순서 변경·실험 분리·미래 행 제거 등을 포함한 로컬 검사 통과 후 제출했습니다. 운영진 공식 검사 통과는 아직 별도 확인되지 않았습니다.

위 CV·Validation은 사용자가 제공한 집계 결과이고 Public은 실제 리더보드에서 확인했습니다. CV 후보 선택에 사용된 점수는 독립적인 최종 성능 추정이 아닙니다. 기존 팀원 Hybrid Public 0.9003493296, 앞선 단일 신경망 `all`은 0.8663581489였습니다. 각 피처 하나의 개별 인과적 효과나 향후 Private 점수를 보장하지 않습니다.

## 구조

- 이미지: 기존 고정 ResNet-18의 512차원 특징, 전체 RGB 이미지를 224×224로 resize하고 ImageNet 정규화.
- 신경망: 기존 9개 현재 센서 피처, seeds 42/43/44의 3개 Fusion head 확률 평균. 각 head는 16 epoch, AdamW lr=.001, weight_decay=.001, batch256, dropout .2, class weight 없음.
- 트리: 현재 센서 9개 + 과거 방향 피처 16개 = 25개 센서 피처, 이미지 PCA32를 함께 사용한 HistGradientBoostingClassifier.
- 결합: 신경망 평균 확률과 트리 확률을 **50:50**으로 평균한 뒤 argmax+1, 제출 라벨 1~6.
- 트리 설정: learning_rate .06, max_iter250, max_leaf_nodes15, min_samples_leaf32, L2=5, early_stopping=False, random_state42.

추가 시계열 피처: t1/t2의 과거 10/30/60초 변화량과 실제 경과 초로 나눈 변화율, 과거 최고 전압 대비 하락량·비율, 과거 최고 t1/t2 대비 하락량입니다. 같은 experiment 안에서 현재 이하 시점의 제공된 행만 쓰고, 같은 시각의 행 순서에 영향받지 않게 계산합니다. 미래 관측·실험 전체 길이/평균/최댓값·다른 실험의 통계·ID/파일명 파생 정답 정보는 사용하지 않습니다. 과거 행 자체가 제거되면 시계열 피처가 달라질 수 있습니다.

결측 대체·센서 정규화·이미지 정규화·PCA는 Train 또는 CV fold의 학습 부분에서만 fit합니다. 추론에서는 저장된 전처리만 적용합니다. Test에서 fit하거나 별도 캐시/숨긴 CSV를 읽지 않습니다.

## 필요한 파일

코드 PR/브랜치에서 `hybrid_v2_temporal` 폴더를 받고, 별도 전달한 `temporal_tree_public09401_weights.tar.gz`를 저장소 루트에서 풉니다.

```bash
git clone --branch codex/temporal-tree-public09401 https://github.com/Cepil/unid.git unid-temporal
cd unid-temporal
tar -xzf /path/to/temporal_tree_public09401_weights.tar.gz
cd hybrid_v2_temporal
```

반드시 같은 `weights/`에 다음을 둡니다:

```text
weights/model.pt
weights/tree.joblib
weights/manifest.json
weights/validation_metrics.json
```

`model.pt`는 encoder, 신경망 3개, 센서/이미지 전처리, 피처 스키마, 학습 버전을 포함합니다. `tree.joblib`는 PCA와 학습된 트리입니다. 둘 다 있어야 추론됩니다. 모델 재학습이나 ImageNet 다운로드는 필요 없습니다. `MODEL_CARD.json`에 실제 제출 CSV의 이름과 SHA256이 기록되며 제출 CSV 내용 자체는 묶음에 포함하지 않습니다.

## 환경과 재현

사용자 실행 화면 기준 학습 서버는 Linux / 대회 제공 RTX 5080, torch 2.11.0+cu128, torchvision 0.26.0+cu128입니다. Python과 수치 라이브러리의 정확한 버전은 가중치 묶음의 `HANDOFF_ENVIRONMENT.json`과 `requirements.reproduce.txt`를 확인하세요. 저장된 tree 모델은 학습 당시와 같은 scikit-learn 버전이 필요하며 추론 코드가 이를 검사합니다. 패키지 버전은 실제 가중치를 묶을 때 생성되므로 코드 문서에 추정값을 적지 않습니다.

```bash
source ~/venv/bin/activate
python verify_handoff.py
```

운영진 GPU용 torch/torchvision 조합을 유지하고, 다른 패키지는 전달된 정확한 버전을 확인해 준비합니다. 공용 환경을 바꾸기 전 팀원과 조율하세요. `requirements.reproduce.txt`의 torch/torchvision은 버전을 고정하지 않습니다. 환경이 이미 맞으면 재설치는 필요 없습니다.

추론:

```bash
python test.py --input ~/data/battery_data_light/test/test.csv \
  --image_root ~/data/battery_data_light/test \
  --output outputs/teammate_reproduction.csv
```

입력 CSV 순서를 유지한 `sample_id,label` CSV를 만듭니다. 이미지 batch 기본32, workers2입니다. `--model` 기본 경로는 스크립트 폴더의 `weights/model.pt`이며 다른 인자도 기본값이 있습니다. GPU 메모리가 부족하면 `--batch_size 16`을 사용할 수 있으나 새 환경에서 추론 일관성 검사를 실행하세요.

```bash
python verify.py --input ~/data/battery_data_light/test/test.csv \
  --image_root ~/data/battery_data_light/test \
  --model weights/model.pt \
  --final outputs/teammate_reproduction.csv \
  --sample_submission ~/data/battery_data_light/sample_submission.csv
```

제출 CSV의 해시까지 비교하려면:

```bash
python verify_handoff.py --reproduced_submission outputs/teammate_reproduction.csv
```

다른 라이브러리/GPU에서 동일한 label이 나오는지는 실제 검사로 확인해야 합니다. 위 검사는 운영진 공식 `check_submission.py`를 대체하지 않습니다. 최종 파일 선택 전 공식 검사를 추가로 실행하세요.

## 현재 보유자가 가중치 묶기

이 패키징 도구와 `PUBLIC09401_SOURCE.json`, 이 문서를 원래 실행 폴더에 추가한 다음 서버에서 실행합니다. 원본 학습·추론 핵심 소스는 바꾸지 않습니다.

```bash
cd ~/Cepil/hybrid_v2_temporal
source ~/venv/bin/activate
python package_handoff.py
```

기본 입력은 `outputs/selected_temporal_tree/`의 기존 가중치/manifest/Validation 집계와 `outputs/submission_temporal_tree.csv`입니다. 결과는 상위 `~/Cepil`의 `temporal_tree_public09401_weights.tar.gz`와 `.sha256`입니다. 기존 파일은 덮어쓰지 않습니다. 코드 해시, 학습 manifest, recipe/seeds/epoch/Validation 메타데이터와 결과 압축 파일을 검사하며 학습·추론은 하지 않습니다.

VS Code Remote-SSH에서 결과 압축 파일과 `.sha256`를 우클릭 → Download로 내려받아 팀 내부에 직접 전달하세요. 코드 PR은 가중치 바이너리·제출 CSV·데이터·캐시·fold mapping·로그를 포함하지 않습니다. 압축 파일도 데이터 CSV나 원본 제출 CSV는 포함하지 않습니다. 카톡/이메일 자동 전송은 수행하지 않습니다.

## 출처

기존 Hybrid 구조는 팀원 PR2 commit `34d526949b9215a6a5c4d22e522ebd2b53097a84`를 기준으로 비교했습니다. 고정 인코더는 torchvision `ResNet18_Weights.IMAGENET1K_V1`입니다.

- https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.resnet18.html
- https://download.pytorch.org/models/resnet18-f37072fd.pth
- torchvision 코드 BSD-3-Clause: https://github.com/pytorch/vision/blob/v0.20.1/LICENSE

코드 라이선스와 데이터/사전학습 가중치의 이용 조건은 구분해 최종 제출 시 확인하세요. 대회 데이터·정답·이를 담은 화면/로그/CSV를 외부 생성형 AI에 보내거나 공개 레포에 올리지 않습니다.
