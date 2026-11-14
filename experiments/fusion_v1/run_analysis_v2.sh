#!/usr/bin/env bash
# Run inside an existing Slurm allocation, not on moana-master.
set -euo pipefail
if [[ -z "${SLURM_JOB_ID:-}" ]]; then
  echo "STOP: allocate a compute node first."
  exit 1
fi
case "$(hostname)" in
  *master*|*login*) echo "STOP: do not execute Python on master/login nodes."; exit 1 ;;
esac
cd "$(dirname "$0")"
UNID_PY="/data/$USER/venvs/unid/bin/python"
UNID_DATA="/local_datasets/$USER/unid/battery_data_light"
if [[ ! -x "$UNID_PY" || ! -f "$UNID_DATA/train/train.csv" ]]; then
  echo "STOP: check private Python environment and node-local dataset."
  exit 1
fi
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCH_HOME="/data/$USER/unid/torch-cache"
mkdir -p logs
(
  echo "Allocated node: $(hostname); job: $SLURM_JOB_ID"
  "$UNID_PY" -u analyze_v2.py --synthetic_check
  "$UNID_PY" -u analyze_v2.py \
    --data_dir "$UNID_DATA" \
    --cache_dir "/local_datasets/$USER/unid/feature-cache" \
    --output_dir outputs/analysis_v2
) 2>&1 | tee logs/analysis_v2.log
