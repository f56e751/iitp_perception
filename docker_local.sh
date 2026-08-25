#!/bin/bash
# Run the live pipeline (RealSense USB passthrough) in the iitp_local image.
# The image carries both backends; arguments are forwarded to main.py, e.g.
#   ./docker_local.sh                  SAM3 TensorRT (main.py's default)
#   ./docker_local.sh --backend dino   GroundingDINO
#   ./docker_local.sh --port 8081
# --privileged + /dev mount is the simplest way to expose USB devices.

# Persistent HuggingFace cache so bert-base-uncased (GroundingDINO text encoder)
# is not re-downloaded every run. Offline flags skip the huggingface.co HEAD
# checks that otherwise hang on this host's broken IPv6 route.
HF_CACHE="${HF_CACHE:-/PublicSSD/iitp/hf_cache}"

# -t only when attached to a terminal, so nohup runs do not fail.
tty_flags=(-i)
[[ -t 0 ]] && tty_flags=(-i -t)

docker run "${tty_flags[@]}" --rm \
  --network host \
  --gpus all \
  --ipc=host \
  --privileged \
  -v /dev:/dev \
  -v "$PWD":/mnt \
  -v "$HF_CACHE":/root/.cache/huggingface \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  --name iitp_local \
  iitp_local:latest \
  bash -c "cd /mnt && exec python main.py $*"
