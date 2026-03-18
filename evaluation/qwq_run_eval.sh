#!/bin/bash
set -euo pipefail

export SUMMARY_API_KEY="<REDACTED_API_KEY>"
export SUMMARY_API_BASE="<REDACTED_URL>"
export SUMMARY_MODEL_NAME="gpt-5.1"

export VLLM_PORTS="8004,8005,8006,8007"
export VLLM_MODEL_NAME="QwQ-32B"

QA_PATH_DEFAULT="<REDACTED_PATH>"
OUT_PATH_DEFAULT="<REDACTED_PATH>"

QA_PATH="${1:-$QA_PATH_DEFAULT}"
OUT_PATH="${2:-$OUT_PATH_DEFAULT}"
NUM_WORKERS="${3:-100}"

SCRIPT_DIR="<REDACTED_PATH>"
cd "$SCRIPT_DIR"

echo ""
echo "=========================================="
echo "Qwen3 Model Evaluation Pipeline"
echo "=========================================="
echo "Test Set:    ${QA_PATH}"
echo "Output Path: ${OUT_PATH}"
echo "Workers:     ${NUM_WORKERS}"
echo "vLLM Ports:  ${VLLM_PORTS}"
echo "Model:       ${VLLM_MODEL_NAME}"
echo "=========================================="
echo ""

if [ ! -f "${QA_PATH}" ]; then
  echo "[ERROR] Test set not found: ${QA_PATH}"
  exit 1
fi

echo "[INFO] Checking vLLM services..."
ALL_SERVICES_OK=true

for PORT in $(echo "${VLLM_PORTS}" | tr ',' ' '); do
  if curl -s -f "<REDACTED_URL>" >/dev/null 2>&1; then
    echo "[✓] Port ${PORT} is healthy"
  else
    echo "[✗] Port ${PORT} is NOT responding"
    ALL_SERVICES_OK=false
  fi
done

if [ "$ALL_SERVICES_OK" = false ]; then
  echo ""
  echo "[ERROR] Some vLLM services are not running!"
  echo "        Start them with: bash start_vllm_servers.sh"
  echo ""
  echo "Example commands:"
  echo "  CUDA_VISIBLE_DEVICES=4 vllm serve ${VLLM_MODEL_NAME} --port 8004 --trust-remote-code --enable-reasoning --reasoning-parser qwen3 &"
  echo "  CUDA_VISIBLE_DEVICES=5 vllm serve ${VLLM_MODEL_NAME} --port 8005 --trust-remote-code --enable-reasoning --reasoning-parser qwen3 &"
  echo "  CUDA_VISIBLE_DEVICES=6 vllm serve ${VLLM_MODEL_NAME} --port 8006 --trust-remote-code --enable-reasoning --reasoning-parser qwen3 &"
  echo "  CUDA_VISIBLE_DEVICES=7 vllm serve ${VLLM_MODEL_NAME} --port 8007 --trust-remote-code --enable-reasoning --reasoning-parser qwen3 &"
  exit 1
fi

echo "[INFO] All vLLM services are ready!"
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EVAL_DIR="/ossfs/workspace/process/eval_shards_${TIMESTAMP}"
mkdir -p "${EVAL_DIR}"

echo "[INFO] Evaluation directory: ${EVAL_DIR}"

TOTAL_LINES=$(wc -l < "${QA_PATH}")
if [ "${TOTAL_LINES}" -eq 0 ]; then
  echo "[ERROR] Test set is empty"
  exit 1
fi

echo "[INFO] Total test cases: ${TOTAL_LINES}"

LINES_PER_SHARD=$(( (TOTAL_LINES + NUM_WORKERS - 1) / NUM_WORKERS ))
echo "[INFO] Cases per worker: ${LINES_PER_SHARD}"

split -l "${LINES_PER_SHARD}" -d -a 3 "${QA_PATH}" "${EVAL_DIR}/shard_"

ACTUAL_SHARDS=$(ls -1 "${EVAL_DIR}"/shard_* 2>/dev/null | wc -l)
echo "[INFO] Created ${ACTUAL_SHARDS} shards"
echo ""

PIDS=()
echo "[INFO] Starting ${ACTUAL_SHARDS} evaluation workers..."
echo ""

IDX=0
for SHARD_FILE in "${EVAL_DIR}"/shard_*; do
  SHARD_ID=$(printf "%03d" "${IDX}")
  OUT_FILE="${EVAL_DIR}/eval_out_${SHARD_ID}.jsonl"
  LOG_FILE="${EVAL_DIR}/worker_${SHARD_ID}.log"

  echo "[Worker ${SHARD_ID}] ${SHARD_FILE} -> ${OUT_FILE}"

  python -u "${SCRIPT_DIR}/qwq_eval.py" \
    --mode eval \
    --qa_path "${SHARD_FILE}" \
    --out_path "${OUT_FILE}" \
    --enable_thinking \
    > "${LOG_FILE}" 2>&1 &

  PIDS+=($!)
  IDX=$((IDX + 1))

  sleep 0.5
done

echo ""
echo "[INFO] Launched ${#PIDS[@]} workers (PIDs: ${PIDS[*]})"
echo ""

echo "[INFO] Monitoring evaluation progress (Ctrl+C to stop monitoring)..."
echo ""

monitor_progress() {
  LAST_COUNT=0
  START_TIME=$(date +%s)

  while true; do
    sleep 30

    COMPLETED=0
    for OUT_FILE in "${EVAL_DIR}"/eval_out_*.jsonl; do
      if [ -f "${OUT_FILE}" ]; then
        LINES=$(wc -l < "${OUT_FILE}" 2>/dev/null || echo 0)
        COMPLETED=$((COMPLETED + LINES))
      fi
    done

    PROGRESS=$(awk "BEGIN {printf \"%.1f\", ($COMPLETED / $TOTAL_LINES) * 100}")

    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    if [ ${ELAPSED} -gt 0 ]; then
      SPEED=$(awk "BEGIN {printf \"%.2f\", $COMPLETED / $ELAPSED}")
    else
      SPEED="N/A"
    fi

    if [ ${COMPLETED} -gt 0 ] && [ "${SPEED}" != "N/A" ] && [ "$(echo "$SPEED > 0" | bc)" -eq 1 ]; then
      REMAINING=$((TOTAL_LINES - COMPLETED))
      ETA_SECONDS=$(awk "BEGIN {printf \"%.0f\", $REMAINING / $SPEED}")
      ETA_MIN=$((ETA_SECONDS / 60))
      ETA="~${ETA_MIN}min"
    else
      ETA="N/A"
    fi

    echo "[$(date '+%H:%M:%S')] Progress: ${COMPLETED}/${TOTAL_LINES} (${PROGRESS}%) | Speed: ${SPEED} cases/s | ETA: ${ETA}"

    LAST_COUNT=${COMPLETED}

    ALL_DONE=true
    for PID in "${PIDS[@]}"; do
      if kill -0 "${PID}" 2>/dev/null; then
        ALL_DONE=false
        break
      fi
    done

    if [ "$ALL_DONE" = true ]; then
      echo "[INFO] All workers have finished!"
      break
    fi
  done
}

monitor_progress &
MONITOR_PID=$!

FAILED_WORKERS=0

for i in "${!PIDS[@]}"; do
  PID="${PIDS[$i]}"
  SHARD_ID=$(printf "%03d" "$i")

  if wait "${PID}"; then
    echo "[✓] Worker ${SHARD_ID} (PID ${PID}) completed successfully"
  else
    EXIT_CODE=$?
    echo "[✗] Worker ${SHARD_ID} (PID ${PID}) failed with exit code ${EXIT_CODE}"
    FAILED_WORKERS=$((FAILED_WORKERS + 1))
  fi
done

kill "${MONITOR_PID}" 2>/dev/null || true

echo ""
if [ ${FAILED_WORKERS} -gt 0 ]; then
  echo "[WARN] ${FAILED_WORKERS} worker(s) failed"
else
  echo "[INFO] All workers completed successfully"
fi
echo ""

echo "[INFO] Merging evaluation results..."

mkdir -p "$(dirname "${OUT_PATH}")"
cat "${EVAL_DIR}"/eval_out_*.jsonl > "${OUT_PATH}"

FINAL_LINES=$(wc -l < "${OUT_PATH}")

echo "[INFO] Merged ${FINAL_LINES} results into ${OUT_PATH}"
echo ""

echo "=========================================="
echo "Evaluation Summary"
echo "=========================================="
echo "Test Set:      ${QA_PATH}"
echo "Results:       ${OUT_PATH}"
echo "Total Cases:   ${TOTAL_LINES}"
echo "Evaluated:     ${FINAL_LINES}"
echo "Workers:       ${ACTUAL_SHARDS}"
echo "Failed:        ${FAILED_WORKERS}"
echo "Work Dir:      ${EVAL_DIR}"
echo ""

SUCCESS=$(grep -o '"status":"success"' "${OUT_PATH}" 2>/dev/null | wc -l)
FAILED=$(grep -o '"status":"failed"' "${OUT_PATH}" 2>/dev/null | wc -l)

if [ ${FINAL_LINES} -gt 0 ]; then
  SUCCESS_RATE=$(awk "BEGIN {printf \"%.2f\", ($SUCCESS / $FINAL_LINES) * 100}")
else
  SUCCESS_RATE="N/A"
fi

echo "Successful:    ${SUCCESS} (${SUCCESS_RATE}%)"
echo "Failed:        ${FAILED}"

if command -v jq &> /dev/null; then
  THINKING_COUNT=$(jq -r 'select(.thinking_trace != "") | .case_id' "${OUT_PATH}" 2>/dev/null | wc -l)
  echo "With Thinking: ${THINKING_COUNT}"
else
  echo ""
  echo "[INFO] Install 'jq' for detailed statistics"
fi

echo "=========================================="
echo ""

echo "[INFO] Evaluation pipeline finished (with ${FAILED_WORKERS} failed worker(s))."
echo ""
exit 0
