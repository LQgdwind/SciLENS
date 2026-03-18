#!/bin/bash
set -euo pipefail

MODEL_PATH="${1:-<REDACTED_PATH>"
SERVED_MODEL_NAME="Qwen3-8B"

declare -A GPU_PORT_MAP
GPU_PORT_MAP[4]=8004
GPU_PORT_MAP[5]=8005
GPU_PORT_MAP[6]=8006
GPU_PORT_MAP[7]=8007

MAX_MODEL_LEN=131072
TENSOR_PARALLEL_SIZE=1
GPU_MEMORY_UTILIZATION=0.85

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
    echo "[ERROR] Port ${PORT} is still in use after cleanup!"
    echo "        Manually kill with: kill \$(lsof -t -i:${PORT})"
    exit 1
  else
    echo "[✓] Port ${PORT} is available"
  fi
done

PIDS=()
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo ""
echo "=========================================="
echo "Starting vLLM Services for Qwen3"
echo "=========================================="
echo "Model Path:    ${MODEL_PATH}"
echo "Model Name:    ${SERVED_MODEL_NAME}"
echo "Max Length:    ${MAX_MODEL_LEN}"
echo "GPU Memory:    ${GPU_MEMORY_UTILIZATION}"
echo "Reasoning:     Enabled (${REASONING_PARSER})"
echo "Tool Calling:  Enabled (${TOOL_CALL_PARSER})"
echo "Timestamp:     ${TIMESTAMP}"
echo "=========================================="
echo ""

for GPU_ID in "${!GPU_PORT_MAP[@]}"; do
  PORT="${GPU_PORT_MAP[$GPU_ID]}"
  LOG_FILE="${LOG_DIR}/vllm_gpu${GPU_ID}_port${PORT}_${TIMESTAMP}.log"

  echo "[INFO] Starting vLLM on GPU ${GPU_ID}, Port ${PORT}"
  echo "       Log: ${LOG_FILE}"

  CUDA_VISIBLE_DEVICES=${GPU_ID} nohup vllm serve "${MODEL_PATH}" \
    --port ${PORT} \
    --host 0.0.0.0 \
    --served-model-name ${SERVED_MODEL_NAME} \
    --max-model-len ${MAX_MODEL_LEN} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --trust-remote-code \
    --disable-log-requests \
    --enable-reasoning \
    --reasoning-parser ${REASONING_PARSER} \
    --enable-auto-tool-choice \
    --tool-call-parser ${TOOL_CALL_PARSER} \
    > "${LOG_FILE}" 2>&1 &

  PID=$!
  PIDS+=($PID)
  echo ${PID} > "${LOG_DIR}/vllm_gpu${GPU_ID}_port${PORT}.pid"
  echo "       PID: ${PID}"
  echo ""
  sleep 2
done

echo "[INFO] Waiting for services to start (may take 3-5 minutes for 128K context)..."
echo ""

MAX_WAIT=1200
WAIT_INTERVAL=5

for GPU_ID in "${!GPU_PORT_MAP[@]}"; do
  PORT="${GPU_PORT_MAP[$GPU_ID]}"
  ELAPSED=0

  echo -n "[INFO] Checking GPU ${GPU_ID} (Port ${PORT})... "

  while [ ${ELAPSED} -lt ${MAX_WAIT} ]; do
    PID_FILE="${LOG_DIR}/vllm_gpu${GPU_ID}_port${PORT}.pid"
    if [ -f "${PID_FILE}" ]; then
      PID=$(cat "${PID_FILE}")
      if ! kill -0 ${PID} 2>/dev/null; then
        echo "❌ FAILED (Process died)"
        echo "[ERROR] Last 30 lines of log:"
        tail -n 30 "${LOG_DIR}/vllm_gpu${GPU_ID}_port${PORT}_${TIMESTAMP}.log"
        exit 1
      fi
    fi

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
    echo "[ERROR] Last 50 lines of log:"
    tail -n 50 "${LOG_DIR}/vllm_gpu${GPU_ID}_port${PORT}_${TIMESTAMP}.log"
    exit 1
  fi
done

echo ""
echo "=========================================="
echo "Service Status Summary"
echo "=========================================="

ALL_HEALTHY=true
ALL_CORRECT_LEN=true

for GPU_ID in "${!GPU_PORT_MAP[@]}"; do
  PORT="${GPU_PORT_MAP[$GPU_ID]}"
  PID_FILE="${LOG_DIR}/vllm_gpu${GPU_ID}_port${PORT}.pid"
  PID=$(cat "${PID_FILE}" 2>/dev/null || echo "UNKNOWN")

  ACTUAL_MAX_LEN=$(curl -s "<REDACTED_URL>" 2>/dev/null | grep -o '"max_model_len":[0-9]*' | cut -d':' -f2 || echo "N/A")

  if [ "$ACTUAL_MAX_LEN" = "131072" ]; then
    echo "GPU ${GPU_ID} | Port ${PORT} | PID ${PID} | Max Len: ${ACTUAL_MAX_LEN} ✅"
  else
    echo "GPU ${GPU_ID} | Port ${PORT} | PID ${PID} | Max Len: ${ACTUAL_MAX_LEN} ❌ (expected 131072)"
    ALL_CORRECT_LEN=false
  fi

  if ! curl -s -f "<REDACTED_URL>" >/dev/null 2>&1; then
    ALL_HEALTHY=false
  fi
done

echo "=========================================="
echo ""

if [ "$ALL_HEALTHY" = true ] && [ "$ALL_CORRECT_LEN" = true ]; then
  echo "[SUCCESS] All vLLM services are running with correct configuration!"
  echo ""
  echo "Service Endpoints:"
  for PORT in "${GPU_PORT_MAP[@]}"; do
    echo "  • <REDACTED_URL>"
  done
  echo ""
  echo "Configuration:"
  echo "  • Model Name: ${SERVED_MODEL_NAME}"
  echo "  • Max Context: ${MAX_MODEL_LEN} tokens (128K)"
  echo "  • Thinking Mode: Enabled (${REASONING_PARSER})"
  echo "  • Tool Calling: Enabled (${TOOL_CALL_PARSER})"
  echo ""
  echo "Verify:"
  echo "  curl -s <REDACTED_URL> | jq '.data[0].max_model_len'"
  echo ""
  echo "To stop:"
  echo "  bash stop_vllm_servers.sh"
  echo ""
elif [ "$ALL_HEALTHY" = true ]; then
  echo "[WARN] Services are running but max_model_len is incorrect!"
  echo "       Expected: 131072, check logs for details"
else
  echo "[ERROR] Some services failed to start"
  exit 1
fi

CONFIG_FILE="${LOG_DIR}/vllm_cluster_${TIMESTAMP}.conf"
cat > "${CONFIG_FILE}" <<EOF
TIMESTAMP=${TIMESTAMP}
MODEL_PATH=${MODEL_PATH}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION}
PORTS=${GPU_PORT_MAP[@]}
PIDS=${PIDS[@]}
LOG_DIR=${LOG_DIR}
ENABLE_REASONING=${ENABLE_REASONING}
REASONING_PARSER=${REASONING_PARSER}
ENABLE_AUTO_TOOL_CHOICE=${ENABLE_AUTO_TOOL_CHOICE}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER}
EOF

echo "[INFO] Configuration saved to: ${CONFIG_FILE}"
echo ""
