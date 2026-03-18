#!/bin/bash
set -euo pipefail

EVAL_SCRIPT="<REDACTED_PATH>"

NUM_WORKERS=100

QA_PATHS=(
  "<REDACTED_PATH>"
  "<REDACTED_PATH>"
  "<REDACTED_PATH>"
  "<REDACTED_PATH>"
  "<REDACTED_PATH>"
  "<REDACTED_PATH>"
)

OUT_PATHS=(
  "<REDACTED_PATH>"
  "<REDACTED_PATH>"
  "<REDACTED_PATH>"
  "<REDACTED_PATH>"
  "<REDACTED_PATH>"
  "<REDACTED_PATH>"
)

NUM_DATASETS=${#QA_PATHS[@]}

if [ ${NUM_DATASETS} -ne ${#OUT_PATHS[@]} ]; then
  echo "[ERROR] QA_PATHS 和 OUT_PATHS 数量不一致"
  exit 1
fi

echo ""
echo "=========================================="
echo "Total datasets to run: ${NUM_DATASETS}"
echo "=========================================="

for (( i=0; i<NUM_DATASETS; i++ )); do
  QA_PATH="${QA_PATHS[$i]}"
  OUT_PATH="${OUT_PATHS[$i]}"

  echo ""
  echo "=========================================="
  echo "Dataset $((i+1)) / ${NUM_DATASETS}"
  echo "QA_PATH:  ${QA_PATH}"
  echo "OUT_PATH: ${OUT_PATH}"
  echo "Workers:  ${NUM_WORKERS}"
  echo "=========================================="
  echo ""

  set +e
  bash "${EVAL_SCRIPT}" "${QA_PATH}" "${OUT_PATH}" "${NUM_WORKERS}"
  RET=$?
  set -e

  if [ $RET -ne 0 ]; then
    echo "[WARN] Dataset ${QA_PATH} finished with exit code ${RET}, continue to next dataset..."
  fi

  echo ""
  echo "====== Dataset $((i+1)) finished ======"
done

echo ""
echo "=========================================="
echo "All datasets finished!"
echo "Total: ${NUM_DATASETS}"
echo "=========================================="
