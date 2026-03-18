#!/bin/bash
set -euo pipefail

MODEL_PATH="${1:-<REDACTED_PATH>"
SERVED_MODEL_NAME="qwen2.5"

INSTANCE1_GPUS="4,5"
INSTANCE1_PORT=8004
INSTANCE1_MASTER_PORT=45711

INSTANCE2_GPUS="6,7"
INSTANCE2_PORT=8005
INSTANCE2_MASTER_PORT=45721

MAX_MODEL_LEN=32768
TENSOR_PARALLEL_SIZE=2
GPU_MEMORY_UTILIZATION=0.80

TOOL_CALL_PARSER="hermes"

LOG_DIR="/ossfs/workspace/vllm_logs"
mkdir -p "${LOG_DIR}"

PORTS=(${INSTANCE1_PORT} ${INSTANCE2_PORT})

echo "[INFO] Checking ports..."
for PORT in "${PORTS[@]}"; do
  if lsof -Pi ":${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[WARN] Port ${PORT} is already in use!"
    bash "$(dirname "$0")/stop_vllm_servers.sh" || true
    sleep 3
    break
  fi
done

for PORT in "${PORTS[@]}"; do
  if lsof -Pi ":${PORT}" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo "[ERROR] Port ${PORT} is still in use!"
    exit 1
  fi
done

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
PIDS=()

start_instance () {
  local GPUS="$1"
  local PORT="$2"
  local NAME="$3"
  local MASTER_PORT="$4"

  local LOG_FILE="${LOG_DIR}/vllm_${NAME}_gpus${GPUS//,/}_port${PORT}_${TIMESTAMP}.log"

  echo "[INFO] Starting ${NAME} on GPUs ${GPUS}, Port ${PORT}, MASTER_PORT=${MASTER_PORT}"

  MASTER_ADDR=127.0.0.1 MASTER_PORT=${MASTER_PORT} \
  CUDA_VISIBLE_DEVICES=${GPUS} nohup vllm serve "${MODEL_PATH}" \
    --port ${PORT} \
    --host 0.0.0.0 \
    --served-model-name ${SERVED_MODEL_NAME} \
    --max-model-len ${MAX_MODEL_LEN} \
    --tensor-parallel-size ${TENSOR_PARALLEL_SIZE} \
    --gpu-memory-utilization ${GPU_MEMORY_UTILIZATION} \
    --trust-remote-code \
    --disable-log-requests \
    --dtype bfloat16 \
    --enable-auto-tool-choice \
    --tool-call-parser ${TOOL_CALL_PARSER} \
    > "${LOG_FILE}" 2>&1 &

  local PID=$!
  PIDS+=("${PID}")
  echo "${PID}" > "${LOG_DIR}/vllm_${NAME}_port${PORT}.pid"
}

start_instance "${INSTANCE1_GPUS}" "${INSTANCE1_PORT}" "instance1" "${INSTANCE1_MASTER_PORT}"
sleep 5
start_instance "${INSTANCE2_GPUS}" "${INSTANCE2_PORT}" "instance2" "${INSTANCE2_MASTER_PORT}"
sleep 5

echo "[INFO] Waiting for services to start..."
MAX_WAIT=36000
WAIT_INTERVAL=10

for PORT in "${PORTS[@]}"; do
  ELAPSED=0
  echo -n "[INFO] Checking Port ${PORT}... "

  while [ ${ELAPSED} -lt ${MAX_WAIT} ]; do
    if curl -s -f "<REDACTED_URL>" >/dev/null 2>&1; then
      echo "READY"
      break
    fi
    sleep ${WAIT_INTERVAL}
    ELAPSED=$((ELAPSED + WAIT_INTERVAL))
    echo -n "."
  done

  if [ ${ELAPSED} -ge ${MAX_WAIT} ]; then
    echo "TIMEOUT"
    LOG_FILE=$(ls "${LOG_DIR}"/vllm_*_port${PORT}_${TIMESTAMP}.log 2>/dev/null | head -n 1 || echo "")
    if [ -n "${LOG_FILE}" ]; then
      echo "[INFO] Tail log: ${LOG_FILE}"
      tail -n 50 "${LOG_FILE}"
    fi
    exit 1
  fi
done

echo ""
echo "=========================================="
echo "Service Status Summary"
echo "=========================================="

ALL_HEALTHY=true

for PORT in "${PORTS[@]}"; do
  ACTUAL_MAX_LEN=$(curl -s "<REDACTED_URL>" 2>/dev/null | grep -o '"max_model_len":[0-9]*' | cut -d':' -f2 || echo "N/A")
  if [ "${ACTUAL_MAX_LEN}" = "${MAX_MODEL_LEN}" ]; then
    echo "Port ${PORT} | Max Len: ${ACTUAL_MAX_LEN} OK"
  else
    echo "Port ${PORT} | Max Len: ${ACTUAL_MAX_LEN} MISMATCH (Expected ${MAX_MODEL_LEN})"
    ALL_HEALTHY=false
  fi
done

echo "=========================================="

if [ "${ALL_HEALTHY}" = true ]; then
  echo "[SUCCESS] All services running correctly!"
  echo "Endpoints: <REDACTED_URL> <REDACTED_URL>"
else
  echo "[WARN] Context length mismatch found."
fi

CONFIG_FILE="${LOG_DIR}/vllm_cluster_${TIMESTAMP}.conf"
cat > "${CONFIG_FILE}" <<EOF
MODEL_PATH=${MODEL_PATH}
MAX_MODEL_LEN=${MAX_MODEL_LEN}
INSTANCE1_GPUS=${INSTANCE1_GPUS}
INSTANCE1_PORT=${INSTANCE1_PORT}
INSTANCE2_GPUS=${INSTANCE2_GPUS}
INSTANCE2_PORT=${INSTANCE2_PORT}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE}
EOF
