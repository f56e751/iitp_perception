#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="${ONNX_DIR:-${ROOT}/weights/SAM3-trt/onnx}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${ROOT}/weights/usls}"
RELEASE_BASE="${RELEASE_BASE:-https://github.com/jamjamjon/assets/releases/download/sam3}"
FORCE="${FORCE:-0}"
ASSET_SET="${1:-runtime}"

runtime_assets=(
  "vision-encoder.onnx"
  "vision-encoder-fp16.onnx"
  "text-encoder.onnx"
  "decoder.onnx"
  "decoder-fp16.onnx"
  "tokenizer.json"
)

fp16_assets=(
  "vision-encoder-fp16.onnx"
  "text-encoder-fp16.onnx"
  "decoder-fp16.onnx"
  "tokenizer.json"
)

metadata_assets=(
  "config.json"
  "configuration.json"
  "merges.txt"
  "processor_config.json"
  "special_tokens_map.json"
  "tokenizer_config.json"
  "vocab.json"
)

case "${ASSET_SET}" in
  runtime)
    assets=("${runtime_assets[@]}")
    ;;
  fp16)
    assets=("${fp16_assets[@]}")
    ;;
  metadata)
    assets=("${metadata_assets[@]}")
    ;;
  all)
    assets=("${runtime_assets[@]}" "${fp16_assets[@]}" "${metadata_assets[@]}")
    ;;
  *)
    echo "Usage: $0 [runtime|fp16|metadata|all]" >&2
    exit 2
    ;;
esac

mkdir -p "${ONNX_DIR}" "${TOKENIZER_DIR}"

download() {
  local name="$1"
  local dst="${ONNX_DIR}/${name}"
  local tmp="${dst}.tmp"
  local url="${RELEASE_BASE}/${name}"

  if [[ "${FORCE}" != "1" && -s "${dst}" ]]; then
    echo "[skip] ${dst}"
    return
  fi

  echo "[download] ${url}"
  rm -f "${tmp}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 5 --retry-delay 2 -o "${tmp}" "${url}"
  elif command -v wget >/dev/null 2>&1; then
    wget -O "${tmp}" "${url}"
  else
    echo "curl or wget is required." >&2
    exit 1
  fi
  mv "${tmp}" "${dst}"
}

declare -A seen=()
for asset in "${assets[@]}"; do
  if [[ -n "${seen[${asset}]:-}" ]]; then
    continue
  fi
  seen["${asset}"]=1
  download "${asset}"
done

if [[ -s "${ONNX_DIR}/tokenizer.json" ]]; then
  cp "${ONNX_DIR}/tokenizer.json" "${TOKENIZER_DIR}/tokenizer.json"
  echo "[ok] tokenizer copied to ${TOKENIZER_DIR}/tokenizer.json"
fi

echo "[ok] assets are ready under ${ONNX_DIR}"
