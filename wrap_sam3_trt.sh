#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

ONNX_DIR="${ONNX_DIR:-${ROOT}/weights/SAM3-trt/onnx}"
VISION_ENGINE_DIR="${VISION_ENGINE_DIR:-${ROOT}/weights/SAM3-trt/usls_engines_b2p2}"
ENGINE_DIR="${ENGINE_DIR:-${ROOT}/weights/SAM3-trt/usls_engines}"
IMAGE_SIZE="${IMAGE_SIZE:-1008}"
PROMPT_BATCH="${PROMPT_BATCH:-6}"
TEXT_SEQ_LEN="${TEXT_SEQ_LEN:-32}"
WORKSPACE_GB="${WORKSPACE_GB:-8}"
PRECISION="${PRECISION:-fp16}"

choose_onnx() {
  local env_value="$1"
  local preferred="$2"
  local fallback="$3"
  if [[ -n "${env_value}" ]]; then
    echo "${env_value}"
  elif [[ -s "${preferred}" ]]; then
    echo "${preferred}"
  else
    echo "${fallback}"
  fi
}

VISION_ONNX="$(choose_onnx "${VISION_ONNX:-}" "${ONNX_DIR}/vision-encoder-fp16.onnx" "${ONNX_DIR}/vision-encoder.onnx")"
TEXT_ONNX="$(choose_onnx "${TEXT_ONNX:-}" "${ONNX_DIR}/text-encoder-fp16.onnx" "${ONNX_DIR}/text-encoder.onnx")"
DECODER_ONNX="$(choose_onnx "${DECODER_ONNX:-}" "${ONNX_DIR}/decoder-fp16.onnx" "${ONNX_DIR}/decoder.onnx")"

VISION_ENGINE="${VISION_ENGINE:-${VISION_ENGINE_DIR}/vision_b2_fp16.engine}"
TEXT_ENGINE="${TEXT_ENGINE:-${ENGINE_DIR}/text_b${PROMPT_BATCH}_fp16.engine}"
DECODER_ENGINE="${DECODER_ENGINE:-${ENGINE_DIR}/decoder_b1_p32_fp16.engine}"

require_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "Missing file: ${path}" >&2
    echo "Run: bash download_sam3_onnx.sh runtime" >&2
    exit 1
  fi
}

require_file "${VISION_ONNX}"
require_file "${TEXT_ONNX}"
require_file "${DECODER_ONNX}"
mkdir -p "${VISION_ENGINE_DIR}" "${ENGINE_DIR}"

builder=(python SAM3-trt/build_sam3_image_trt.py --workspace-gb "${WORKSPACE_GB}" --precision "${PRECISION}")

echo "[wrap] vision: ${VISION_ONNX} -> ${VISION_ENGINE}"
"${builder[@]}" \
  --onnx "${VISION_ONNX}" \
  --output "${VISION_ENGINE}" \
  --profile "images:1x3x${IMAGE_SIZE}x${IMAGE_SIZE}:1x3x${IMAGE_SIZE}x${IMAGE_SIZE}:1x3x${IMAGE_SIZE}x${IMAGE_SIZE}"

echo "[wrap] text: ${TEXT_ONNX} -> ${TEXT_ENGINE}"
"${builder[@]}" \
  --onnx "${TEXT_ONNX}" \
  --output "${TEXT_ENGINE}" \
  --profile "input_ids:${PROMPT_BATCH}x${TEXT_SEQ_LEN}:${PROMPT_BATCH}x${TEXT_SEQ_LEN}:${PROMPT_BATCH}x${TEXT_SEQ_LEN}" \
  --profile "attention_mask:${PROMPT_BATCH}x${TEXT_SEQ_LEN}:${PROMPT_BATCH}x${TEXT_SEQ_LEN}:${PROMPT_BATCH}x${TEXT_SEQ_LEN}"

decoder_extra_args=()
if [[ -n "${DECODER_EXTRA_ARGS:-}" ]]; then
  read -r -a decoder_extra_args <<< "${DECODER_EXTRA_ARGS}"
fi

echo "[wrap] decoder: ${DECODER_ONNX} -> ${DECODER_ENGINE}"
"${builder[@]}" \
  --onnx "${DECODER_ONNX}" \
  --output "${DECODER_ENGINE}" \
  --profile "fpn_feat_0:1x256x288x288:1x256x288x288:1x256x288x288" \
  --profile "fpn_feat_1:1x256x144x144:1x256x144x144:1x256x144x144" \
  --profile "fpn_feat_2:1x256x72x72:1x256x72x72:1x256x72x72" \
  --profile "fpn_pos_2:1x256x72x72:1x256x72x72:1x256x72x72" \
  --profile "prompt_features:1x${TEXT_SEQ_LEN}x256:1x${TEXT_SEQ_LEN}x256:1x${TEXT_SEQ_LEN}x256" \
  --profile "prompt_mask:1x${TEXT_SEQ_LEN}:1x${TEXT_SEQ_LEN}:1x${TEXT_SEQ_LEN}" \
  "${decoder_extra_args[@]}"

echo "[ok] TensorRT engines are ready:"
echo "  ${VISION_ENGINE}"
echo "  ${TEXT_ENGINE}"
echo "  ${DECODER_ENGINE}"
