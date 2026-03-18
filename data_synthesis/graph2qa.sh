#!/usr/bin/env bash
set -euo pipefail

WORK_DIR=$(pwd)
GEN_SCRIPT="${WORK_DIR}/gen_qa_from_subgraphs.py"
SUBGRAPH_FILE="/ossfs/workspace/OAG/KG/0104_paper_subgraphs_textual.jsonl"

TOTAL_SUBGRAPHS=249628
SEGMENTS=16
SEGMENT_SIZE=$((TOTAL_SUBGRAPHS / SEGMENTS))

MODELS=(
  "claude-3-5-sonnet-20241022"
  "claude-3-7-sonnet-20250219"
  "claude-opus-4-1-20250805"
  "claude-opus-4-20250514"
  "claude-sonnet-4-20250514"
  "claude-sonnet-4-5-20250929"
  "gemini-2.0-flash"
  "gpt-4-turbo-2024-04-09"
  "gpt-4.1-2025-04-14"
  "gpt-4.1-mini-2025-04-14"
  "gpt-4o-2024-08-06"
  "gpt-4o-2024-11-20"
  "gpt-4o-mini-2024-07-18"
  "gpt-5-2025-08-07"
  "gpt-5-chat-2025-08-07"
  "gpt-5-mini-2025-08-07"
  "gpt-5.1"
  "gpt-5.1-chat"
  "o3-2025-04-16"
  "o3-mini-2025-01-31"
  "gpt-5.2"
  "gpt-5.2-chat"
)
NUM_MODELS=${#MODELS[@]}

mkdir -p /ramdata/seg
mkdir -p /ramdata/qa

echo "========== Stage 1: Splitting Subgraph File =========="

prepare_segment() {
    local segment_index=$1
    local start_line=$((segment_index * SEGMENT_SIZE + 1))
    local end_line=$(( (segment_index + 1) * SEGMENT_SIZE ))
    if (( segment_index == SEGMENTS - 1 )); then
        end_line=${TOTAL_SUBGRAPHS}
    fi
    local part_file="/ramdata/seg/paper_subgraphs_textual_seg${segment_index}.jsonl"

    echo "[Split] Generating Segment ${segment_index} (${start_line}-${end_line})..."
    sed -n "${start_line},${end_line}p" "${SUBGRAPH_FILE}" > "${part_file}"
}

for seg in $(seq 0 $((SEGMENTS-1))); do
    prepare_segment "${seg}" &
done
wait

echo "========== All segments generated. Starting QA generation =========="

run_for_segment_and_model() {
  local segment_index=$1
  local model_index=$2

  local model_name="${MODELS[$model_index]}"
  local safe_model_name=$(echo "${model_name}" | sed 's/[^A-Za-z0-9_-]/_/g')
  local part_file="/ramdata/seg/paper_subgraphs_textual_seg${segment_index}.jsonl"

  local k=$((model_index + 1))
  local seg_model_file="/ramdata/seg/paper_subgraphs_textual_seg${segment_index}_m${model_index}.jsonl"

  awk -v n="${NUM_MODELS}" -v k="${k}" '{
    if (NR % n == k % n) print $0
  }' "${part_file}" > "${seg_model_file}"

  local model_subgraphs=$(wc -l < "${seg_model_file}")

  if (( model_subgraphs == 0 )); then
    echo "[WARN] Model ${model_name} has 0 subgraphs in segment ${segment_index} (Check Logic!)"
    return 0
  fi

  local out_file="/ramdata/qa/${safe_model_name}_seg${segment_index}_paper_qa.jsonl"

  python "${GEN_SCRIPT}" \
    --work_dir "${WORK_DIR}" \
    --max_subgraphs "${model_subgraphs}" \
    --max_qa_per_subgraph 1 \
    --max_turns 10 \
    --llm_model "${model_name}" \
    --subgraph_file "${seg_model_file}" \
    --output_file "${out_file}"
}

export WORK_DIR GEN_SCRIPT NUM_MODELS MODELS
export -f run_for_segment_and_model

for seg in $(seq 0 $((SEGMENTS-1))); do
  for mid in $(seq 0 $((NUM_MODELS-1))); do
    run_for_segment_and_model "${seg}" "${mid}" &
  done
done

wait
echo "========== All Done =========="
