#!/usr/bin/env bash
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "STOP: allocate a compute node first."
  exit 1
fi
case "$(hostname)" in
  *master*|*login*) echo "STOP: no Python on master/login nodes."; exit 1 ;;
esac
cd "$(dirname "$0")"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCH_HOME="/data/$USER/unid/torch-cache"
UNID_PY="/data/$USER/venvs/unid/bin/python"
mkdir -p logs weights
(
  "$UNID_PY" -u train.py --synthetic_check
  "$UNID_PY" -u train.py \
    --data_dir "/local_datasets/$USER/unid/battery_data_light" \
    --cache_dir "/local_datasets/$USER/unid/feature-cache" \
    --output weights/hybrid.pt
) 2>&1 | tee logs/hybrid_train.log
