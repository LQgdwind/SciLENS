#!/bin/bash

TARGET_GPUS=(0 1 2 3)

MODEL_PATH="Qwen/Qwen3-Embedding-8B"

BASE_PORT=8000

echo "Starting vLLM servers for Qwen3-Embedding-8B ..."
echo "Target GPUs: ${TARGET_GPUS[*]}"

PIDS=()
main_ports=()

for gpu_id in "${TARGET_GPUS[@]}"; do
  port=$((BASE_PORT + gpu_id))

  main_ports+=($port)

  echo "  -> starting server on GPU $gpu_id, port $port"

  CUDA_VISIBLE_DEVICES=$gpu_id vllm serve "$MODEL_PATH" \
    --task embed \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port "$port" \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --disable-log-requests \
    > "vllm_gpu${gpu_id}_port${port}.log" 2>&1 &

  PIDS+=($!)
done

timeout=600
start_time=$(date +%s)
declare -A server_status

for port in "${main_ports[@]}"; do
  server_status[$port]=false
done

echo "Waiting for the following ports to start: ${main_ports[*]}"

while true; do
  all_ready=true

  for port in "${main_ports[@]}"; do
    if [ "${server_status[$port]}" = "false" ]; then
      if curl -s -f <REDACTED_URL> > /dev/null 2>&1; then
        echo "Server (port $port) is ready!"
        server_status[$port]=true
      else
        all_ready=false
      fi
    fi
  done

  if [ "$all_ready" = "true" ]; then
    echo "All designated servers are ready for inference!"
    break
  fi

  current_time=$(date +%s)
  elapsed=$((current_time - start_time))
  if [ $elapsed -gt $timeout ]; then
    echo -e "\nError: Server startup timeout after ${timeout} seconds"

    for port in "${main_ports[@]}"; do
      if [ "${server_status[$port]}" = "false" ]; then
        echo "  -> Port $port failed to start."
      fi
    done

    if [ ${#PIDS[@]} -gt 0 ]; then
      echo "Killing started vLLM processes: ${PIDS[*]}"
      kill "${PIDS[@]}" 2>/dev/null
    fi

    exit 1
  fi

  printf '.'
  sleep 5
done

echo "Script finished successfully."
