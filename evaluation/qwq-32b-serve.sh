#!/bin/bash
set -euo pipefail

MODEL_PATH="${1:-<REDACTED_PATH>"
SERVED_MODEL_NAME="QwQ-32B"

declare -A GPU_PORT_MAP
GPU_PORT_MAP[4]=8004
GPU_PORT_MAP[5]=8005
GPU_PORT_MAP[6]=8006
GPU_PORT_MAP[7]=8007

MAX_MODEL_LEN=131072
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.90

ENABLE_REASONING=true
REASONING_PARSER="deepseek_r1"

ENABLE_AUTO_TOOL_CHOICE=true
TOOL_CALL_PARSER="hermes"

LOG_DIR="/ossfs/workspace/vllm_logs"
mkdir -p "${LOG_DIR}"

echo "[INFO] Checking port availability..."

for GPU_ID in "${!GPU_PORT_MAP[@]}"; do
  PORT="${GPU_PORT_MAP[$GPU_ID]}"

  if lsof -Pi ":${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[WARN] Port ${PORT} is already in use!"
    echo "       Stopping existing services..."
    bash "$(dirname "$0")/stop_vllm_servers.sh" || true
    sleep 3
    break
  fi
done

for GPU_ID in "${!GPU_PORT_MAP[@]}"; do
  PORT="${GPU_PORT_MAP[$GPU_ID]}"
  if lsof -Pi ":${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[ERROR] Port ${PORT} is still in use!"
    exit 1
  fi
done

PIDS=()
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "=========================================="
echo "Starting vLLM Services for QwQ-32B"
echo "=========================================="
echo "Model Path:    ${MODEL_PATH}"
echo "Model Name:    ${SERVED_MODEL_NAME}"
echo "Max Length:    ${MAX_MODEL_LEN}"
echo "TP Size:       ${TENSOR_PARALLEL_SIZE}"
echo "GPU Util:      ${GPU_MEMORY_UTILIZATION}"
echo "Timestamp:     ${TIMESTAMP}"
echo "=========================================="
echo ""

for GPU_ID in "${!GPU_PORT_MAP[@]}"; do
  PORT="${GPU_PORT_MAP[$GPU_ID]}"
  LOG_FILE="${LOG_DIR}/vllm_gpu${GPU_ID}_port${PORT}_${TIMESTAMP}.log"

  echo "[INFO] Starting vLLM on GPU ${GPU_ID}, Port ${PORT}"

  CMD=(vllm serve "${MODEL_PATH}"
    --port ${PORT}
    --host 0.0.0.0
    --served-model-name ${SERVED_MODEL_NAME}
    --max-model-len ${MAX_MODEL_LEN}
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE}
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION}
    --trust-remote-code
    --disable-log-requests
  )

  if [ "${ENABLE_REASONING}" = true ]; then
    CMD+=(--enable-reasoning --reasoning-parser ${REASONING_PARSER})
  fi

  if [ "${ENABLE_AUTO_TOOL_CHOICE}" = true ]; then
    CMD+=(--enable-auto-tool-choice --tool-call-parser ${TOOL_CALL_PARSER})
  fi

  CUDA_VISIBLE_DEVICES=${GPU_ID} nohup "${CMD[@]}" \
    > "${LOG_FILE}" 2>&1 &

  PID=$!
  PIDS+=($PID)
  echo ${PID} > "${LOG_DIR}/vllm_gpu${GPU_ID}_port${PORT}.pid"
  sleep 5
done

echo "[INFO] Waiting for services to start..."
MAX_WAIT=36000
WAIT_INTERVAL=10

for GPU_ID in "${!GPU_PORT_MAP[@]}"; do
  PORT="${GPU_PORT_MAP[$GPU_ID]}"
  ELAPSED=0

  echo -n "[INFO] Checking GPU ${GPU_ID} (Port ${PORT})... "

  while [ ${ELAPSED} -lt ${MAX_WAIT} ]; do
    if curl -s -f "<REDACTED_URL>" >/dev/null 2>&1; then
      echo "✅ READY"
      break
    fi
    sleep ${WAIT_INTERVAL}
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    echo -n "."
  done

  if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
    echo "❌ TIMEOUT"
    tail -n 50 "${LOG_DIR}/vllm_gpu${GPU_ID}_port${PORT}_${TIMESTAMP}.log"
    exit 1
  fi
done

echo ""
echo "=========================================="
echo "Service Status Summary"
echo "=========================================="

ALL_HEALTHY=true

for GPU_ID in "${!GPU_PORT_MAP[@]}"; do
  PORT="${GPU_PORT_MAP[$GPU_ID]}"
  ACTUAL_MAX_LEN=$(curl -s "<REDACTED_URL>" 2>/dev/null | grep -o '"max_model_len":[0-9]*' | cut -d':' -f2 || echo "N/A")

  if [ "$ACTUAL_MAX_LEN" = "$MAX_MODEL_LEN" ]; then
    echo "GPU ${GPU_ID} | Port ${PORT} | Max Len: ${ACTUAL_MAX_LEN} ✅"
  else
    echo "GPU ${GPU_ID} | Port ${PORT} | Max Len: ${ACTUAL_MAX_LEN} ❌ (Expected ${MAX_MODEL_LEN})"
    ALL_HEALTHY=false
  fi
done

echo "=========================================="

if [ "$ALL_HEALTHY" = true ]; then
  echo "[SUCCESS] All services running for QwQ-32B!"
  echo "Endpoints: <REDACTED_URL>"
else
  echo "[WARN] Context length mismatch found."
fi

CONFIG_FILE="${LOG_DIR}/vllm_cluster_${TIMESTAMP}.conf"
cat > "${CONFIG_FILE}" <<EOF
MODEL_PATH=${MODEL_PATH}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
EOF
