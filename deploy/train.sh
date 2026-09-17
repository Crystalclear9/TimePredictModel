#!/usr/bin/env bash
set -euo pipefail
# Run on the training server. Input is an immutable exported snapshot.
: "${DATASET:?Set DATASET to the snapshot JSONL path}"
: "${MODEL_DIR:?Set MODEL_DIR to the candidate output directory}"
exec timepredict train --data "$DATASET" --output "$MODEL_DIR" --data-kind observed --threads "${TRAIN_THREADS:-4}" --rounds 500 "$@"
