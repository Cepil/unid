# hybrid_cum 팀 전달 문서 (11/14 밤)

## 요약

- 현재 팀 최종 후보: **hybrid_cum, Public 0.9447186465**. 제출 파일은 `submission_cum.csv`. 직전 후보 hybrid_bs(0.9384)를 대체한다.
- 구성: hybrid_bs (Hybrid v2 + 주최 측 베이스라인 v7의 30초 센서 피처) + 누적 극값 피처 7개 (트리에만 추가).
- 출처와 경위: README.md의 "설계 근거와 경위" 참고.
- 대회 서버에서 Train만으로 학습했다. Validation은 0.761662.
- 공식 `check_submission.py` 7개 항목을 모두 통과했다.

## 필요한 파일

- 코드: 이 폴더 (`hybrid_cum/`).
- 가중치: 따로 전달하는 `hybrid_cum_weights.tar.gz`. 이 폴더 안에서 풀면 `weights/`가 생긴다.
  - `weights/model.pt`: encoder, MLP 3개, 전처리 값
  - `weights/tree.joblib`: PCA, 트리
  - `weights/manifest.json`, `weights/validation_metrics.json`
  - `weights/SHA256SUMS`
- **model.pt와 tree.joblib 둘 다 있어야** 추론된다.
- 해시 앞 16자리: model.pt `d62e2c87b5fda14b`, tree.joblib `a6493ce6f1c12b12`, 제출 CSV `567ae7043462913b`.

## 설치와 추론 (대회 서버)

```bash
cd hybrid_cum
tar -xzf <경로>/hybrid_cum_weights.tar.gz      # weights/ 생성
(cd weights && sha256sum -c SHA256SUMS)
source ~/venv/bin/activate                      # torch/torchvision은 운영진 설치본 그대로 둔다
python -c "import sklearn; print(sklearn.__version__)"   # 1.7.2여야 한다
python test.py --input ~/data/battery_data_light/test/test.csv \
  --image_root ~/data/battery_data_light/test --output submission_cum_reproduced.csv
sha256sum submission_cum_reproduced.csv          # 567ae7043462913b…와 같으면 재현 성공
```

## 점검

```bash
python test_synthetic.py                        # 가짜 데이터로 학습·추론과 일관성 검사
python check_ties.py --input ~/data/battery_data_light/test/test.csv
cp <check_submission.py 경로> . && python check_submission.py --data_dir ~/data/battery_data_light --final submission_cum_reproduced.csv
```

## 재학습 (필요할 때만)

```bash
python train.py --data_dir ~/data/battery_data_light --output_dir weights_new   # 기존 weights는 덮어쓰지 않는다
```

## 주의

- 대회 데이터, 제출 CSV, 가중치는 GitHub에 올리지 않는다. `.gitignore`로 막아 두었다.
- DAKER 최종 파일은 직접 지정해야 한다. 지정하지 않으면 Public 최고점(사용 보류된 0.9401)이 자동으로 선택된다.
- 서버의 `~/sja/hybrid_cum`는 원본이므로 수정하지 말고, 복사해서 작업한다.
- DAKER 최종 파일: `submission_cum.csv` (0.9447)로 지정.
