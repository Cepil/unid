# Hybrid v2 팀 전달 문서

작성일: 2026-11-14, 한국 시간.

## 팀원이 우선 알아야 할 내용

- 현재 최고 Public: **0.9003493296**, 파일명 **hybrid_v2_submission.csv**.
- 학습된 가중치가 반드시 필요합니다. 코드만 clone해서는 기존 예측을 재현할 수 없습니다. 재학습이 아니라 가중치를 내려받아 추론합니다.
- 필요한 가중치: `hybrid_v2/weights/hybrid.pt`와 **같은 폴더의 `tree.joblib`**. 하나만 전달하면 추론할 수 없습니다.
- `training_metadata.json`은 실제 학습 라이브러리 버전·검증 결과, `SHA256SUMS`는 전달 파일의 무결성 확인에 사용합니다.
- 가중치 묶음은 로컬로 다운로드해 내부 체크섬 3개 모두 검증했습니다. 사용자가 팀 카톡으로 `hybrid_v2_weights.tar.gz`를 직접 전달합니다. 레포에는 바이너리를 올리지 않고 `hybrid_v2/WEIGHTS_MANIFEST.json`에 검증된 해시를 기록했습니다.

## 코드 받기와 가중치 설치

공유 브랜치/PR의 코드를 받습니다. 아직 main에 합쳐지지 않았다면 main만 clone해서는 후보 코드가 보이지 않습니다.

```bash
git clone --branch codex/hybrid-v2-team-handoff https://github.com/Cepil/unid.git
cd unid/hybrid_v2

# 전달받은 가중치 묶음을 현재 hybrid_v2 디렉터리에서 해제합니다.
tar -xzf <hybrid_v2_weights.tar.gz의 경로>
(cd weights && sha256sum -c SHA256SUMS)
```

## 환경

대회 제공 RTX 5080 서버는 운영진 setup.sh로 준비한 GPU용 torch/torchvision을 유지합니다. **Seraph용 torch 2.5.1/cu121로 덮어쓰지 않습니다.** 기본 requirements.txt는 torch/torchvision 설치를 요청하지 않습니다. `requirements-seraph.txt`는 과거 Seraph 환경의 참고이며 대회 제공 서버에서 설치하지 마십시오.

1. 운영진 환경에서 `source ~/venv/bin/activate`.
2. `python -m pip install -r requirements.txt`. 실제 학습 artifact에서 확인한 numpy 1.26.4, pandas 2.3.3, scikit-learn 1.7.2, Pillow 11.1.0, joblib 1.6.0을 고정했습니다. torch/torchvision은 건드리지 않습니다.
3. `weights/training_metadata.json`의 `environment_versions`를 확인합니다. 특히 joblib로 저장된 scikit-learn 모델은 동일 scikit-learn 버전에서 재현해야 합니다. 파일을 받은 후 버전을 맞추세요. NumPy·joblib도 저장된 버전과 맞추는 것을 권장합니다.
4. `python -c "import torch; print(torch.cuda.is_available(), torch.__version__)"`로 GPU 확인. CPU 추론도 지원하지만 느릴 수 있습니다.

PyTorch/torchvision을 교체하지 않고 sklearn 버전을 확인하는 예:

```bash
python - <<'PY'
import json
v=json.load(open('weights/training_metadata.json'))['environment_versions']
for k in ['torch','torchvision','numpy','scikit-learn','joblib']:
    print(k, v[k])
PY
```

정확한 scikit-learn 버전으로 설치하려면 아래를 실행합니다. 실제 artifact를 학습할 때 사용한 버전을 읽어 사용하며 PyTorch는 변경하지 않습니다.

```bash
SKLEARN_VERSION=$(python -c "import json; print(json.load(open('weights/training_metadata.json'))['environment_versions']['scikit-learn'])")
python -m pip install "scikit-learn==$SKLEARN_VERSION"
```

## 추론

아래는 대회 제공 서버 예시입니다. 데이터 경로가 다르면 인자만 바꿉니다.

```bash
python test.py \
  --input ~/data/battery_data_light/test/test.csv \
  --image_root ~/data/battery_data_light/test \
  --output hybrid_v2_submission.csv
```

- 기본 모델 경로는 코드 폴더의 `weights/hybrid.pt`입니다. 형제 파일 `weights/tree.joblib`도 읽습니다.
- ImageNet 다운로드나 전처리 재학습을 수행하지 않습니다.
- 입력 CSV의 현재 행 이미지·센서만 사용합니다. 미래 행, 다른 실험의 관측, 숨겨둔 Test CSV를 읽지 않습니다.
- 출력은 `sample_id,label` 순서이며 입력 행 순서와 라벨 1~6을 유지합니다.
- GPU 메모리가 부족하면 `--batch_size 16`, 경로 문제는 `--image_root`를 확인합니다.

## 제출 전 검사

```bash
python verify_submission.py \
  --input ~/data/battery_data_light/test/test.csv \
  --image_root ~/data/battery_data_light/test \
  --submission hybrid_v2_submission.csv \
  --sample_submission ~/data/battery_data_light/sample_submission.csv \
  --temp_dir /tmp/unid-hybrid-verify
```

검사기는 CSV 열/라벨/ID/공식 양식 순서를 확인하고, 최대 64개 샘플의 부분 입력·반복 실행·행 순서 변경 일관성을 확인합니다. 이는 모든 부분집합에 대한 증명은 아닙니다. 공유 버전은 Seraph 호스트에서 마스터/로그인 노드와 Slurm 할당 여부를 검사하고, 대회 제공 서버에서는 Slurm 없이 실행할 수 있게 했습니다. Seraph에서는 여전히 할당된 컴퓨트 노드에서만 실행해야 합니다.

대회 제공 공식 `check_submission.py`도 별도로 실행해 확인하세요. 임의로 SLURM_JOB_ID를 설정해 마스터 노드 제한을 우회하지 않습니다.

## 실험 결과와 해석

| 모델 | 5-fold 평균 | OOF Macro F1 | 최저 fold |
| --- | --- | --- | --- |
| v1 seed42 | 0.815681 | 0.817944 | 0.754327 |
| 3-seed Fusion | 0.817253 | 0.820242 | 0.774542 |
| 센서 전용 HGB (분석용) | 0.799489 | 0.798934 | 0.779904 |
| 이미지·센서 HGB | 0.851690 | 0.852909 | 0.810572 |
| Hybrid | **0.862176** | **0.863979** | **0.811531** |

- Train만 experiment_id 단위 GroupKFold 5분할. v1과 같은 fold에서 비교했습니다. 전처리·PCA는 fold 학습 부분에서만 fit했습니다.
- Hybrid는 v1 대비 5개 fold 모두 개선했고 OOF 클래스별 F1도 6개 모두 개선했습니다.
- 센서 전용 대비 이미지 특징을 추가한 HGB의 OOF 상승은 0.053975였습니다. 이미지의 보완성을 지지하지만 물리적 원인의 증명은 아닙니다.
- 제공 Validation은 v1 0.778443, Hybrid 0.727172로 반대 순위였습니다. 이번 Public 비교는 CV 순위와 일치했지만 Private까지 보장하지 않습니다.
- 최종 Hybrid는 provided Train 전체만 학습했습니다. Validation+Train 재학습은 하지 않았습니다.
- 16 epoch, seeds 42/43/44, 50:50 혼합, HGB/PCA 설정은 CV 후보 비교 전에 정했고 결과를 보고 가중치를 탐색하지 않았습니다.
- sklearn 버전·입력 행 순서가 다르면 기본 GroupKFold 배정이 달라질 수 있으므로 정확한 재현에는 기존 비공개 fold_assignment.csv를 팀 내부에서 사용합니다. 이 파일은 공개 레포에 올리지 않습니다.

## 가중치 패키징: 현재 Seraph 보유자

현재 학습 파일은 Seraph 개인 작업 폴더의 `unid/hybrid_v2/weights`에 있습니다. 현재 사용자 기준으로 아래 경로를 사용합니다. `package_weights.sh`는 이 공유 코드에서 추가됐으므로 원래 학습 폴더에 복사한 뒤 실행하세요. 이 스크립트는 Python이나 학습을 실행하지 않습니다.

```bash
cd /data/$USER/unid/hybrid_v2
bash package_weights.sh
```

결과는 `hybrid_v2_weights.tar.gz`와 아카이브 체크섬 파일입니다. 맥에서 다음과 같이 내려받습니다.

```bash
scp moana.seraph:/data/jashin0414/unid/hybrid_v2/hybrid_v2_weights.tar.gz \
  "$HOME/Downloads/hybrid_v2_weights.tar.gz"
scp moana.seraph:/data/jashin0414/unid/hybrid_v2/hybrid_v2_weights.tar.gz.sha256 \
  "$HOME/Downloads/hybrid_v2_weights.tar.gz.sha256"
```

맥에서는 `shasum -a 256`으로, Linux에서는 `sha256sum`으로 아카이브 해시를 확인할 수 있습니다. 이번 팀 전달은 사용자가 카톡으로 가중치 아카이브를 직접 보내는 방식입니다. 아래 SHA256 또는 WEIGHTS_MANIFEST.json과 비교한 뒤, `unid/hybrid_v2` 안에서 압축을 풀어 `weights/` 경로가 맞는지 확인하세요. 최종 심사용 다운로드 링크/공개 범위는 별도로 확정해야 합니다.

아카이브 SHA256: `79a3372dc3485eb3c1a71c0df034553f2ec1bccfa2d2622e1b4de477039f3e68`.

## 운영·제출 주의점

- 제출 제한은 팀 전체 하루 30회, 자정 초기화. 서로 제출 횟수와 후보 파일을 공유합니다.
- 11/15 13:00 리더보드 마감 및 최종 파일 1개 선택. 최신 별도 공지 기준 코드·PPT 메일 마감은 **13:40**, GPU 사용 종료 13:50.
- Public 최고 파일이 최종 파일로 선택됐는지는 DAKER에서 별도로 확인해야 합니다. 이 공유 작업으로 선택을 변경하지 않았습니다.
- 코드·가중치·파일명·환경 버전을 함께 보존합니다. v1도 삭제하지 않습니다.
- 최종 심사 모델은 이미지·센서 모두 사용해야 합니다. 센서 전용 HGB는 비교 분석용입니다.
- 실제 데이터·정답·이미지·이를 담은 화면·로그를 외부 생성형 AI에 보내지 않습니다. 공개 레포에는 코드와 집계 결과만 공유합니다.
- Seraph 사용은 문의 중이었던 사항입니다. 외부 GPU 관련 운영진 답변을 팀에 공유하고 허용 범위를 확인하세요. Seraph에서는 마스터의 Python 실행 금지, Slurm 할당·노드 로컬 데이터·자원 제한·사용자 감독 규칙을 지킵니다.

## 팀 채팅에 붙여넣을 요약

> Hybrid v2 Public 0.9003493296 나왔습니다. v1 0.8504558014 대비 약 4.99%p 상승입니다. 동일 GroupKFold 평균 0.862176, 5개 fold 전부 v1보다 개선됐고 제공 Validation은 0.727172입니다. 고정 ResNet-18 이미지 특징+센서 MLP 3시드 평균과 이미지·센서 HGB를 50:50으로 결합합니다. 코드와 실행법은 팀 레포 공유 브랜치/PR에 정리했습니다. 추론에는 hybrid.pt와 tree.joblib 두 파일이 모두 필요하며 가중치 묶음 hybrid_v2_weights.tar.gz는 카톡 파일로 별도 전달하겠습니다. 코드 폴더 hybrid_v2 안에서 풀면 됩니다. 팀 전체 하루 30회 제출이니 추가 제출 전에 공유해주세요.
