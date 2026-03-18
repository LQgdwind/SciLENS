#!/bin/bash
set -euo pipefail

LOG_DIR="/ossfs/workspace/vllm_logs"

echo "=========================================="
echo "Stopping all vLLM services"
echo "=========================================="

STOPPED_COUNT=0
FAILED_COUNT=0

if [ -d "${LOG_DIR}" ]; then
  for PID_FILE in "${LOG_DIR}"/vllm_*.pid; do
    if [ ! -f "${PID_FILE}" ]; then
      continue
    fi

    PID=$(cat "${PID_FILE}" 2>/dev/null || echo "")

    if [ -z "${PID}" ]; then
      echo "[WARN] Invalid PID file: ${PID_FILE}"
      continue
    fi

    BASENAME=$(basename "${PID_FILE}" .pid)

    if kill -0 ${PID} 2>/dev/null; then
      echo -n "[INFO] Stopping ${BASENAME} (PID ${PID})... "

      kill ${PID} 2>/dev/null || true

      for i in {1..10}; do
        if ! kill -0 ${PID} 2>/dev/null; then
          echo "✅ Stopped"
          STOPPED_COUNT=$((STOPPED_COUNT + 1))
          rm -f "${PID_FILE}"
          break
        fi
        sleep 1
      done

      if kill -0 ${PID} 2>/dev/null; then
        echo "⚠️  Force killing..."
        kill -9 ${PID} 2>/dev/null || true
        sleep 1

        if kill -0 ${PID} 2>/dev/null; then
          echo "❌ Failed to stop"
          FAILED_COUNT=$((FAILED_COUNT + 1))
        else
          echo "✅ Force stopped"
          STOPPED_COUNT=$((STOPPED_COUNT + 1))
          rm -f "${PID_FILE}"
        fi
      fi
    else
      echo "[INFO] ${BASENAME} (PID ${PID}) already stopped"
      rm -f "${PID_FILE}"
    fi
  done
fi

echo ""
echo "[INFO] Killing any remaining vLLM processes..."
pkill -9 -f "vllm serve" 2>/dev/null || true

for PORT in 8000 8001 8002 8003 8004 8005 8006 8007; do
  PID=$(lsof -t -i:${PORT} 2>/dev/null || echo "")
  if [ ! -z "${PID}" ]; then
    echo "[INFO] Killing process on port ${PORT} (PID: ${PID})"
    kill -9 ${PID} 2>/dev/null || true
  fi
done

echo ""
echo "=========================================="
echo "Stopped: ${STOPPED_COUNT} | Failed: ${FAILED_COUNT}"
echo "=========================================="

echo ""
echo "[INFO] Verifying ports are free..."
ALL_FREE=true
for PORT in 8004 8005 8006 8007; do
  if lsof -i:${PORT} > /dev/null 2>&1; then
    echo "[✗] Port ${PORT} still in use"
    ALL_FREE=false
  else
    echo "[✓] Port ${PORT} is free"
  fi
done

if [ "$ALL_FREE" = true ]; then
  echo ""
  echo "[SUCCESS] All vLLM services stopped and ports released"
else
  echo ""
  echo "[WARN] Some ports are still in use"
  echo "       Manually check with: lsof -i:8004"
fi
