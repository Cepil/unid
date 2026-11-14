# Second experiment: stability and nonlinear sensor/image interactions

Current accepted reference reported by user: Fusion v1 Public 0.8504558014, provided Validation 0.778443, Train-only CV mean 0.815681 / pooled OOF 0.817944. Preserve original checkpoint and submission. No improved score is claimed for this second experiment.

## Predeclared candidate panel

- v1_seed42: exact architecture/hyperparameter/fixed-16-epoch reference used by cv.py.
- fusion_seedavg: equal probability average of seeds 42,43,44. Same frozen ResNet-18 and sensor MLP. Tests whether stochastic head training variation can be reduced.
- sensor_hgb_diagnostic: histogram gradient boosting on the nine current-row sensor features. Diagnostic only; not eligible as the final model, which must use images and sensors.
- fusion_hgb: the same tree hyperparameters plus 32 PCA dimensions of frozen image embeddings. Tests nonlinear sensor interactions and image contribution within a matched tree family. Adding PCA also changes model capacity, so the difference is an empirical feature-addition comparison rather than proof of a physical mechanism.
- fusion_hybrid: one fixed 50/50 probability average of fusion_seedavg and fusion_hgb. Tests complementarity without a mixture-weight search on held-out labels or Public feedback. Confirm any organizer clarification about ensemble rules before final adoption.

HGB settings fixed in advance: 250 boosting iterations, learning rate .06, 15 max leaf nodes, minimum leaf size 32, L2=5, seed=42, early_stopping=False. These are moderate-complexity starting settings, not optimized values or a promise of improvement. PCA, median imputation, and scaling are fit independently on fold Train. No per-fold best epoch selection. Hold epochs at 16, previously selected using provided Validation.

## Runtime and data boundaries

Use the same private environment on a Slurm allocated compute node with 1 GPU, 4 CPUs, 16GB RAM and an explicit time limit. Master/login node execution is prohibited. Supervise execution with htop. Do not run this concurrently with another GPU workload. Existing interactive allocation may be used while valid; do not start a second GPU allocation if one is still running.

No new packages required beyond the prepared environment. Train only is read. The fixed pretrained image embedding is cached in the requested compute-node-local cache directory. Cache identity includes CSV hash, image path/size/mtime, preprocessing code and relevant library versions. If images were edited while preserving both size and mtime, remove this cache and re-extract; metadata is not a full content hash of every image. Every learned transformation remains fold-specific, including PCA. Validation and Test are not read by this script.

Each fold trains three small MLPs and two CPU tree models. This is more work than cv.py; exact duration depends on the node. Completed-fold probability snapshots are saved but automatic resume is not implemented. Cache reuse avoids repeated image extraction when rerunning. No data arrays, fold assignment files, checkpoints, sample IDs or per-sample outputs may be sent to external AI or pushed to public GitHub. Share only the aggregate SUMMARY, consistent with the workflow used so far.

## Checks and execution

`--synthetic_check` exercises missing-value handling, PCA, CPU fitting and probability shape/normalization with generated data. It must also run on an allocated compute node. It is a smoke check, not a real-data accuracy test. Local authoring validation is limited to syntax and archive inspection until this check runs remotely.

```bash
UNID_PY=/data/$USER/venvs/unid/bin/python
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export TORCH_HOME=/data/$USER/unid/torch-cache

$UNID_PY analyze_v2.py --synthetic_check
$UNID_PY -u analyze_v2.py \
  --data_dir /local_datasets/$USER/unid/battery_data_light \
  --cache_dir /local_datasets/$USER/unid/feature-cache \
  --output_dir outputs/analysis_v2
```

Review both fold mean and pooled OOF, paired fold differences, worst-fold performance and class F1. The seed42 reference should reproduce the previous CV up to environment/determinism differences; investigate material deviations before comparing candidates. Prefer improvements distributed across folds. Do not select the sensor-only diagnostic as final even if its score is best. Repeated use of this CV panel can itself overfit the validation procedure; keep experiments limited and justified. Validate the chosen candidate on provided Validation before one further Public submission. No train/test.py interface changes or deployment artifacts are made by this analysis patch.

References: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html and https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html .
