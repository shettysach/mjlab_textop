#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${REPO_ID:-Yochish/TextOp-Data}"
MOTION_REL="${MOTION_REL:-TextOpTracker/artifacts/Data10k-open/homejrhangmr_dataset_pbhc_contact_maskACCADFemale1Walking_c3dB3-walk1_posespkl/motion.npz}"
DATA_DIR="${DATA_DIR:-/tmp/textop-data}"
OUTPUT_FILE="${OUTPUT_FILE:-/tmp/textop_walk_mjlab.npz}"
DEVICE="${DEVICE:-cuda:0}"
NUM_ENVS="${NUM_ENVS:-1}"
VIEWER="${VIEWER:-viser}"
AGENT="${AGENT:-zero}"
EXTRA="${EXTRA:-cu128}"

echo "[1/3] Downloading TextOp motion"
uvx hf download "${REPO_ID}" \
  --repo-type dataset \
  --include "${MOTION_REL}" \
  --local-dir "${DATA_DIR}"

INPUT_FILE="${DATA_DIR}/${MOTION_REL}"

echo "[2/3] Normalizing TextOp motion for MJLab"
uv run --extra "${EXTRA}" normalize-textop-npz \
  --input-file "${INPUT_FILE}" \
  --output-file "${OUTPUT_FILE}" \
  --device "${DEVICE}"

echo "[3/3] Launching MJLab viewer"
if [[ "${AGENT}" == "trained" ]]; then
  : "${CHECKPOINT_FILE:?Set CHECKPOINT_FILE=/path/to/model.pt when AGENT=trained}"
  uv run --extra "${EXTRA}" play Mjlab-Tracking-Flat-Unitree-G1 \
    --agent trained \
    --checkpoint-file "${CHECKPOINT_FILE}" \
    --motion-file "${OUTPUT_FILE}" \
    --num-envs "${NUM_ENVS}" \
    --device "${DEVICE}" \
    --viewer "${VIEWER}"
else
  uv run --extra "${EXTRA}" play Mjlab-Tracking-Flat-Unitree-G1 \
    --agent "${AGENT}" \
    --motion-file "${OUTPUT_FILE}" \
    --num-envs "${NUM_ENVS}" \
    --device "${DEVICE}" \
    --viewer "${VIEWER}" \
    --no-terminations True
fi
