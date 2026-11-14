#!/usr/bin/env bash
# Lightweight packaging only: no Python, inference, data reads or recursive search.
set -euo pipefail
cd "$(dirname "$0")"
for file in hybrid.pt tree.joblib training_metadata.json; do
  if [[ ! -f "weights/$file" ]]; then
    echo "Missing required artifact: weights/$file"
    exit 1
  fi
done
(
  cd weights
  sha256sum hybrid.pt tree.joblib training_metadata.json > SHA256SUMS
)
COPYFILE_DISABLE=1 tar -czf hybrid_v2_weights.tar.gz \
  weights/hybrid.pt weights/tree.joblib weights/training_metadata.json weights/SHA256SUMS
sha256sum hybrid_v2_weights.tar.gz > hybrid_v2_weights.tar.gz.sha256
echo "Created hybrid_v2_weights.tar.gz and archive checksum. No dataset or predictions included."
