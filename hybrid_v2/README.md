# Hybrid v2: selected from paired Train-only CV

Candidate: fixed 50/50 probability mean of (a) three-seed image/sensor MLP average and (b) an image/sensor histogram gradient boosting classifier. Both components use a frozen torchvision ResNet-18 ImageNet embedding plus nine current-row sensor features. No temporal inputs or Test-derived fitting. Original fusion_v1 artifacts remain outside this directory and unchanged.

## Selection evidence reported by user

| Model | Train-only 5-fold mean | Pooled OOF | Worst fold |
| --- | --- | --- | --- |
| Fusion v1 seed42 | 0.815681 | 0.817944 | 0.754327 |
| Fusion three-seed mean | 0.817253 | 0.820242 | 0.774542 |
| Sensor-only HGB (diagnostic) | 0.799489 | 0.798934 | 0.779904 |
| Image/sensor HGB | 0.851690 | 0.852909 | 0.810572 |
| Selected hybrid | 0.862176 | 0.863979 | 0.811531 |

Hybrid improved all five paired folds over v1. OOF per-class F1 also improved for all six classes. Adding PCA image features within the HGB family improved pooled OOF by 0.053975 compared with the sensor-only diagnostic. This supports empirical complementarity, not a guaranteed physical mechanism or Private/Public score. V1's user-reported Public is 0.8504558014; hybrid Public subsequently reported by the user: **0.9003493296**, provided Validation **0.727172**. No Private score is known.

The candidate panel and 50/50 mixture were specified before those results. CV is used for model selection, so its best-candidate score is not an unbiased final generalization estimate. Provided Validation is an additional diagnostic but has previously influenced design/epoch selection. No independent untouched holdout or statistical significance is claimed.

## Exact training recipe

Three Fusion heads use seeds 42,43,44, 16 fixed epochs each, batch size 256, AdamW lr=.001/weight_decay=.001, dropout=.2, cross entropy without class weights. Same Fusion architecture and training protocol as analyze_v2.py. No validation-based early stopping, mixture search or class threshold fitting.

Sensor inputs: voltage, current, temperature1, temperature2, pressure, t2−t1, average temperature, maximum temperature, V×I. Train medians impute nonfinite values, Train mean/std standardize sensors, clipped to ±20. Train mean/std normalize 512 image features; PCA32 is fit on these Train features for the tree branch. Tree settings: learning_rate=.06, max_iter=250, max_leaf_nodes=15, min_samples_leaf=32, l2_regularization=5, early_stopping=False, random_state=42. All learned preprocessing uses Train only. Images convert to RGB, resize the complete frame to 224×224, then ImageNet normalization.

`run_train.sh` first runs synthetic consistency checks, fits provided Train, evaluates provided Validation and saves artifacts. No Validation rows are used in fitting, and Test is not loaded. Shared private local image cache from analysis_v2 is reused when CSV/code/library/image metadata match. Images modified while preserving size and mtime require cache removal/re-extraction. Dataset/cache paths are specific to the allocated compute node.

## Inference and required artifacts

`weights/hybrid.pt` stores the encoder, three heads and fitted preprocessing. `weights/tree.joblib` stores fitted PCA and HGB; it must remain next to hybrid.pt. Both artifacts are required. Only load the joblib file produced by trusted own training. `weights/training_metadata.json` records actual versions and aggregate Validation metrics.

```bash
python test.py --input <CSV> --image_root <image directory> --output <CSV>
```

Other args have defaults: --model weights/hybrid.pt (relative to script), --batch_size 32, --workers 2. Inference does not download weights, fit preprocessing, use labels, or read hidden original CSVs. ID/order/subset checks must be rerun for this candidate; v1's successful tests do not establish v2 invariance. The bundled verifier supports the new default checkpoint and uses sampled consistency checks, not an exhaustive proof over all subsets.

## Environment and attribution

Private Seraph environment: /data/jashin0414/venvs/unid, Python 3.10, torch 2.5.1+cu121, torchvision 0.20.1+cu121, RTX A2000 6GB. Run Python only on a Slurm allocated compute node with explicit GPU=1, CPU=4, RAM=16GB and time limits; supervise with htop. Do not run the competition GPU server setup.sh here. requirements-seraph.txt describes the historical training environment; the default requirements.txt deliberately does not install torch/torchvision; record exact trained sklearn version from training_metadata.json for reproduction. Keep the target server's GPU-compatible torch build rather than blindly replacing it.

Image encoder: torchvision 0.20.1 ResNet18_Weights.IMAGENET1K_V1, official pretrained artifact https://download.pytorch.org/models/resnet18-f37072fd.pth , documentation https://docs.pytorch.org/vision/0.20/models/generated/torchvision.models.resnet18.html . torchvision code BSD-3-Clause: https://github.com/pytorch/vision/blob/v0.20.1/LICENSE . Code license does not by itself establish dataset or pretrained-weight licensing; complete the final attribution and applicable terms verification before final submission. No external training dataset is loaded.

scikit-learn HGB/PCA docs: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html and https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html .

Before final delivery add team details, chosen final filename, final environment versions, inference verification and trained artifact links as required by organizers. Keep competition data, arrays, prediction CSVs, logs, fold mappings and credentials out of public GitHub and external AI. Confirm any organizer clarification about ensembles if applicable.
