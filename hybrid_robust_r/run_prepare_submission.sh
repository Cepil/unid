#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
export UNID_EXPECTED_HOST="${UNID_EXPECTED_HOST:-da-perfect28}"
[[ "$(hostname -s)" == "$UNID_EXPECTED_HOST" ]] || { echo '대회 제공 서버에서만 실행하세요.'; exit 1; }
UNID_PY="${UNID_PY:-$HOME/venv/bin/python}"
UNID_DATA="${UNID_DATA:-$HOME/data/battery_data_light}"
UNID_SOURCE="${UNID_SOURCE:-$HOME/unid_work/outputs/t37_robust_mlp_20261114_204727}"
UNID_CACHE="${UNID_CACHE:-$HOME/unid_work/hv2_cache}"
test -x "$UNID_PY"
test -f "$UNID_SOURCE/metrics.json"
test -f "$UNID_SOURCE/robust_heads.pt"
test -f "$UNID_DATA/test/test.csv"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$HOME/unid_work"
exec 9>"$HOME/unid_work/.prepare_robust_r.lock"
flock -n 9 || { echo 'R 제출 준비가 이미 실행 중입니다.'; exit 1; }
[[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)" ]] || { echo '다른 GPU 작업이 끝난 뒤 실행하세요.'; exit 1; }

UNID_CHECKER="${UNID_CHECKER:-}"
if [[ -z "$UNID_CHECKER" ]]; then
  for path in "$HOME/check_submission.py" "$HOME/unid_work/check_submission.py" "$HOME/data/check_submission.py" \
    "$HOME/unid_work/hybrid_epair/check_submission.py" "$HOME/unid_work/clean/check_submission.py" \
    "$HOME/data/battery_data_light/check_submission.py" "$HOME/data/baseline/uni_d_baseline_v7/check_submission.py"; do
    if [[ -f "$path" ]]; then UNID_CHECKER="$path"; break; fi
  done
fi
if [[ -z "$UNID_CHECKER" ]]; then
  for root in "$HOME/unid_work" "$HOME/data/baseline"; do
    if [[ -d "$root" ]]; then
      UNID_CHECKER="$(find "$root" -maxdepth 4 -type f -name check_submission.py -print -quit)"
      [[ -z "$UNID_CHECKER" ]] || break
    fi
  done
fi
if [[ -z "$UNID_CHECKER" || ! -f "$UNID_CHECKER" ]]; then
  echo '공식 check_submission.py를 찾지 못했습니다. 기존 공식 파일 경로를 UNID_CHECKER로 지정하고 다시 실행하세요.'
  exit 1
fi
echo "Official checker: $UNID_CHECKER"
if [[ "$UNID_CHECKER" != "$PWD/check_submission.py" ]]; then cp "$UNID_CHECKER" ./check_submission.py; fi
"$UNID_PY" check_encoding.py
"$UNID_PY" -u export_r.py --source_dir "$UNID_SOURCE" --data_dir "$UNID_DATA" --cache_dir "$UNID_CACHE" --output_dir "$PWD/weights"
"$UNID_PY" -u test.py --input "$UNID_DATA/test/test.csv" --image_root "$UNID_DATA/test" --output "$PWD/submission_robust_r.csv"
"$UNID_PY" -u check_submission.py --data_dir "$UNID_DATA" --final "$PWD/submission_robust_r.csv" 2>&1 | tee official_check.log
sha256sum submission_robust_r.csv
echo 'CSV 생성 및 공식 검사 실행이 끝났습니다. 공식 검사 출력의 모든 항목이 통과했는지 확인 후 Public에 1회 제출합니다.'
