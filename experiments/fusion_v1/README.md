# Current-row image/sensor fusion v1

First experiment, not yet trained or verified on competition data. Baseline leaderboard score reported by user: 0.8010471797 (assumed Public, to confirm). No competition data was read during code authoring.

## Design

Frozen ResNet-18 ImageNet embedding (512 dimensions) plus a sensor MLP; concatenated embeddings feed a 6-class MLP. Each image is converted to RGB and resized to 224×224 without cropping, then normalized with ImageNet mean/std. No brightness/color augmentation. No relative_time_s, experiment_id, neighboring rows, future observations, or experiment-level statistics are prediction inputs.

Nine sensor features: the five original measurements, t2−t1, mean(t1,t2), max(t1,t2), and V×I. V×I is not interpreted as internal heat generation. Train-only medians impute missing/nonfinite values. Train-only means/stds standardize sensor features (clipped to ±20) and image embeddings.

The code asserts that provided Train/Validation experiment IDs and sample IDs are disjoint. The image backbone remains in eval mode and frozen. Images are encoded once per split; only the fusion head is trained. Validation Macro F1 over all six classes selects the checkpoint with early stopping. Validation has been used for model selection, so it is not an untouched final performance estimate. Test is not loaded during training.

Inference loads the saved encoder, fusion head and preprocessing without downloading or fitting anything. It preserves input row order and outputs sample_id,label with labels 1..6. No filename-based label restoration or basename search is used. Subset/order/future-row invariance still requires runtime verification; the architecture avoids cross-row prediction dependencies.

## Environment

Prepared target: Seraph allocated compute node, Python 3.10.22, PyTorch 2.5.1+cu121, torchvision 0.20.1+cu121, RTX A2000 6GB. Use the already prepared private environment; do not install requirements into shared Conda or replace the competition GPU server's PyTorch. requirements.txt records this experiment's versions; final submission dependencies must reflect the final verified environment.

Run all Python and compute work only on a Slurm allocated compute node, CPU=4, RAM=16GB, GPU=1 with explicit time limit. Keep data on the compute node's local storage and supervise execution with htop. This code does not provide authorization to use Seraph for the competition: the user has asked organizers and the reply is pending.

## Pretrained source

torchvision 0.20.1 `ResNet18_Weights.IMAGENET1K_V1`, published before the competition. Official weights URL: https://download.pytorch.org/models/resnet18-f37072fd.pth . Documentation: https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.resnet18.html . torchvision code license: BSD-3-Clause (https://github.com/pytorch/vision/blob/v0.20.1/LICENSE). This code license does not establish a separate license for the ImageNet dataset or weight artifact; verify applicable weight/data terms and complete the final README attribution before submission. External training data is not downloaded.

## Commands (allocated compute node only)

### Paired cross-validation

`cv.py` loads only Train and uses `GroupKFold(n_splits=5)` with no shuffle, matching the supplied baseline default when the team used it unchanged and Train row order/scikit-learn version agree. Optionally pass `--folds_csv` with exact `sample_id,fold` assignments from the teammate to guarantee matching folds; do not upload this private mapping to GitHub or AI. The default folds are exported privately to outputs/cv_v1/fold_assignment.csv.

The reference baseline is the unmodified model.py from the user's originally downloaded baseline.zip, copied here as baseline_reference.py. It runs both baseline class_weight settings and fusion on the same folds. Confirm that the teammate used this baseline version before treating scores as an exact reproduction. Image statistics use two workers. The reference baseline's temporal feature logic is retained for comparison, including its documented subset-input caveat.

Frozen image features are computed once for all Train; only fixed ImageNet inference is shared across folds. All fitted normalization/imputation and classifier fitting happen within fold Train. Fusion uses 16 fixed epochs chosen from the separate provided-Validation run, not the best epoch on each held-out fold. This reduces optimistic checkpoint selection inside CV. Results include fold mean/std, per-fold paired differences and pooled OOF Macro F1, which are distinct quantities. No fold checkpoints replace weights/fusion.pt. Comparing two baseline weight settings is not a nested-CV estimate of a selected hyperparameter procedure.

```bash
/data/$USER/venvs/unid/bin/python -u cv.py \
  --data_dir /local_datasets/$USER/unid/battery_data_light \
  --folds 5 --epochs 16 --batch_size 32 --workers 2
```

Assess improvement over the paired baseline across folds, its variability, pooled OOF score, and the provided Validation result before spending one team submission. No numerical threshold guarantees Public or Private improvement. A single Public comparison is evidence for this candidate only and does not validate the whole CV procedure.

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TORCH_HOME=/data/$USER/unid/torch-cache
UNID_PY=/data/$USER/venvs/unid/bin/python
UNID_DATA=/local_datasets/$USER/unid/battery_data_light

# Short code/path/device check first; not a reportable model result.
$UNID_PY train.py --data_dir "$UNID_DATA" --epochs 2 --output weights/smoke.pt

# First actual experiment, no class weighting.
$UNID_PY -u train.py --data_dir "$UNID_DATA" --output weights/fusion.pt

# Optional separate comparison: square-root inverse frequency loss weights.
$UNID_PY -u train.py --data_dir "$UNID_DATA" --class_weight sqrt --output weights/fusion_sqrt.pt

$UNID_PY test.py --input "$UNID_DATA/test/test.csv" --image_root "$UNID_DATA/test" --output submission.csv
```

Image embeddings are retained in memory during training, not reused from disk, so each new train.py execution re-encodes both splits. Batch size can be reduced using --batch_size 16 if needed. No competition data or derived arrays should be uploaded to GitHub or external AI. Keep logs and checkpoints private. Final README must add team details, selected file, validation design/results, final inference command and licensing information.
