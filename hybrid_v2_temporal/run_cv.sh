#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "$0")"
if [[ -f "$HOME/venv/bin/activate" ]]; then
  source "$HOME/venv/bin/activate"
fi
UNID_PYTHON="${UNID_PYTHON:-python}"
UNID_DATA="${1:-$HOME/data/battery_data_light}"
UNID_SOURCE="${2:-weights/source_encoder.pt}"
if [[ $# -lt 2 && ! -f "$UNID_SOURCE" && -f ../fusion_v2/weights/fusion_v1.pt ]]; then
  UNID_SOURCE=../fusion_v2/weights/fusion_v1.pt
fi
UNID_RUN="${3:-outputs/paired_cv_$(date +%Y%m%d_%H%M%S)}"
UNID_FOLDS="${4:-}"
UNID_LOG="outputs/$(basename "$UNID_RUN").log"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
if [[ -e "$UNID_RUN" || -e "$UNID_LOG" ]]; then
  echo "Run or log already exists; choose a new output directory." >&2
  exit 1
fi
if [[ ! -f "$UNID_DATA/train/train.csv" || ! -f "$UNID_SOURCE" ]]; then
  echo "Missing Train data or source checkpoint. Arguments: data_dir source_checkpoint [new_run_dir] [private_folds_csv]" >&2
  exit 1
fi
"$UNID_PYTHON" -c 'import torch, torchvision; assert torch.cuda.is_available(), "Use the organizer-provided GPU environment"; print("GPU:", torch.cuda.get_device_name(0), "torch:", torch.__version__, "torchvision:", torchvision.__version__)'
"$UNID_PYTHON" test_synthetic.py
"$UNID_PYTHON" test_hybrid_synthetic.py
mkdir -p outputs
UNID_EXTRA=()
if [[ -n "$UNID_FOLDS" ]]; then
  UNID_EXTRA+=(--folds_csv "$UNID_FOLDS")
fi
echo "Starting paired CV; run directory: $UNID_RUN"
"$UNID_PYTHON" -u cv.py --data_dir "$UNID_DATA" --source_checkpoint "$UNID_SOURCE" \
  --output_dir "$UNID_RUN" --cache_dir cache --epochs 16 --folds 5 \
  --batch_size 32 --workers 2 "${UNID_EXTRA[@]}" 2>&1 | tee "$UNID_LOG"
