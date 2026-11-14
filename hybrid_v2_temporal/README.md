# Hybrid v2 + causal sensor features

Selected model: **temporal_tree**, observed Public Macro F1 **0.9401** (leaderboard rounded display), team 13 observed in first place. The user's same-run CV mean was 0.878267 / OOF 0.880200 with 5/5 fold wins over the control; full-Train fixed-16-epoch fit gave Validation 0.751640. Input consistency checks were reported PASS before submission. Private performance and official-checker status are not yet verified.

See **[TEAM_HANDOFF.md](TEAM_HANDOFF.md)** for the selected model's exact recipe, code/weights delivery, recipient inference and environment checks. The sections below retain the experiment/retraining workflow. Existing original Hybrid and single-head temporal models remain preserved.

The user's optimization direction is now to approach Public Macro F1 1.0 while conserving the reported five personal submissions remaining. This is an aspiration, not a measured or guaranteed result; future candidates require held-out evidence before another submission.

## Why this experiment

The original Hybrid has stronger reported paired CV/Public evidence than the single-head temporal model, while the temporal model improved the original single-head Fusion. Combine these two directions in a limited, fixed candidate panel rather than retuning to individual Public results. Keep all original checkpoints and submitted files unchanged.

| Recipe | Three-seed MLP | HGB tree | Purpose |
|---|---|---|---|
| `hybrid_control` | original 9 current-row sensors | original 9 sensors + PCA32 image features | Same-run Hybrid reference |
| `temporal_tree` | same heads as control | 25 current+causal sensors + same PCA32 images | Isolate causal features on the tree branch |
| `temporal_both` | 25 current+causal sensors | 25 sensors + same PCA32 images | Test causal features on both branches |

All candidates use the original frozen ResNet-18, seeds 42/43/44, 16 fixed epochs, and equal 50/50 MLP/tree probability averaging. MLP architecture and AdamW settings match the original Hybrid. Tree settings remain learning_rate=.06, max_iter=250, max_leaf_nodes=15, min_samples_leaf=32, l2_regularization=5, early_stopping=False, seed42. PCA32 and all fitted preprocessing use only the training side of each fold. No thresholds or mixture weights are searched using Test or Public scores.

The temporal columns are the previously implemented 10/30/60-second t1/t2 differences and elapsed-time-adjusted rates, voltage decline from its historical peak (absolute and relative), and t1/t2 decline from their historical peaks. Group by experiment; only observations at or before current time participate. Tied timestamps are aggregated consistently, missing history stays missing until Train-fitted imputation, and original input order is restored. Removing past observations can change predictions; removing future observations or other experiments must preserve them.

## First run on the organizer's GPU server

Upload the code-only ZIP into `~/Cepil`, alongside the existing `fusion_v2` directory. The ZIP contains no dataset, derived arrays, labels, logs or model weights. The runner reuses `../fusion_v2/weights/fusion_v1.pt` solely for the unchanged frozen encoder. That encoder is identical to the encoder in the received Hybrid checkpoint. Original classifiers are not changed.

```bash
cd ~/Cepil
python -m zipfile -e hybrid_v2_temporal_code.zip .
cd hybrid_v2_temporal
source ~/venv/bin/activate
nvidia-smi
```

Coordinate the shared GPU with teammates. Start tmux as a separate command and wait for its prompt:

```bash
tmux new -s hybrid-temporal
```

Inside tmux:

```bash
cd ~/Cepil/hybrid_v2_temporal
bash run_cv.sh
```

If already in tmux, do not start another session. The script checks the existing GPU environment and runs synthetic checks before the 5-fold experiment. It does not install or replace packages. Keep the organizer's torch/torchvision pair; the previous v2 environment already has the needed ordinary libraries. The code has no Slurm requirement on the organizer server. Use only organizer-approved resources for actual competition training.

Default dataset: `~/data/battery_data_light`. For other paths or the exact private original fold mapping:

```bash
bash run_cv.sh /path/to/battery_data_light /path/to/source_checkpoint.pt \
  outputs/new_paired_run /path/to/private_fold_assignment.csv
```

Without a mapping, all three candidates are compared on the same generated GroupKFold splits. A private original mapping is needed to guarantee the same splits as an earlier external run. Library/hardware differences also mean the new control should be judged as a same-run baseline, not assumed to exactly reproduce previously reported scores.

## What the first run does

Only Train CSV/images are used for CV. No provided Validation or Test is loaded. Frozen embeddings are extracted once, then cached privately under `cache/`. Cache identity includes source-checkpoint SHA256, CSV contents, image paths/sizes/mtimes, preprocessing code and library versions. It is not a full hash of image contents; re-extract if images were edited while preserving their metadata.

Each fold trains two sets of three heads and two trees. The tree-only candidate reuses the control heads; the both-temporal candidate reuses the temporal tree. All model comparisons use identical folds and a fixed epoch count. No fold-best checkpoint selection occurs.

Results: `outputs/paired_cv_<timestamp>/metrics.json`, per-fold aggregate progress, and a private `fold_assignment.csv`. The terminal prints fold mean, pooled OOF Macro F1, worst fold and paired wins. No raw rows, per-sample predictions, IDs or images are printed. Logs contain aggregate metrics only. Caches, fold mappings and any files containing competition rows or labels remain private and must not be provided to external AI.

Use fold-by-fold deltas, pooled OOF, class F1 and worst-fold behavior to decide which candidate deserves a full Train fit. The winning score from this panel is a model-selection result, not an unbiased final estimate. Approaching 1.0 is an optimization direction, not an expected or guaranteed result. Do not submit every variant automatically.

## Train one selected recipe

Example, only if `temporal_tree` is selected after CV:

```bash
python -u train.py --data_dir ~/data/battery_data_light \
  --source_checkpoint ../fusion_v2/weights/fusion_v1.pt \
  --recipe temporal_tree --epochs 16 --output_dir outputs/selected_temporal_tree
```

This fits Train only for the fixed epoch count, then evaluates the provided Validation. It does not select an epoch using Validation and does not fit Test. Matching private image caches are reused. Output includes `model.pt`, sibling `tree.joblib`, checksums, exact trained library versions, and aggregate Validation metrics. Existing artifacts are not overwritten. No fold model is substituted for the final Train model.

## Infer, verify and submit

```bash
python test.py --input ~/data/battery_data_light/test/test.csv \
  --image_root ~/data/battery_data_light/test \
  --model outputs/selected_temporal_tree/model.pt \
  --output outputs/submission_temporal_tree.csv

python verify.py --input ~/data/battery_data_light/test/test.csv \
  --image_root ~/data/battery_data_light/test \
  --model outputs/selected_temporal_tree/model.pt \
  --final outputs/submission_temporal_tree.csv \
  --sample_submission ~/data/battery_data_light/sample_submission.csv
```

The fitted tree must remain next to model.pt. Inference requires the SAME scikit-learn version recorded by training; the code checks it before loading the trusted tree artifact. It performs no fitting, external download, hidden CSV loading or feature-cache lookup. Other inference arguments have defaults.

Before final code delivery place the selected `model.pt`, `tree.joblib`, and manifest in `weights/`, and run the mandatory three-argument `test.py` command to generate the selected CSV. Use format-3 artifacts with this code, not original v1 or Hybrid format-2 code. Run the organizer's official `check_submission.py` before final selection. The local verifier supplements that official checker; synthetic tests do not establish other-hardware or real-data prediction invariance.

## Software tests and provenance

```bash
python test_synthetic.py
python test_hybrid_synthetic.py
python test_integration_synthetic.py
```

The first two suites cover causal features, sparse/tied times, history isolation, serialization, no inference fit, experiment grouping, class probability mapping, and neural training. `reference/` contains the unmodified teammate base/hybrid code from PR2 commit `34d526949b9215a6a5c4d22e522ebd2b53097a84`. A test recreates its control on invented data and compares full probability matrices exactly. The integration test exercises the CLI, cache reuse, paired CV, saved manifests, inference and consistency checker using invented CSVs/images/random encoder weights only. Its scores are meaningless for the competition.

No competition rows/images/labels/data-containing logs were used to implement or test this version. Known aggregate results in this document were supplied by the user or public team code documentation. Record final OS/Python/GPU/library versions after server validation. Keep original pretrained attribution: torchvision ResNet18_Weights.IMAGENET1K_V1, https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.resnet18.html ; original weights https://download.pytorch.org/models/resnet18-f37072fd.pth ; torchvision code BSD-3-Clause https://github.com/pytorch/vision/blob/v0.20.1/LICENSE . Applicable dataset/weight terms are separate from the code license.
