#!/usr/bin/env bash
set -euo pipefail
# 任一必需变量缺失立即退出；只读取既有 JSONL 快照，不负责采集或导出。
# Run on the training server. Input is an immutable exported snapshot.
: "${DATASET:?Set DATASET to the snapshot JSONL path}"
: "${MODEL_DIR:?Set MODEL_DIR to the candidate output directory}"
# observed 会触发正式来源证据校验；额外命令行参数原样交给训练入口。
exec timepredict train --data "$DATASET" --output "$MODEL_DIR" --data-kind observed --threads "${TRAIN_THREADS:-4}" --rounds 500 "$@"
