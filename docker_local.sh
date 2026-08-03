#!/bin/bash
# Run grounded_sam container with RealSense USB passthrough + local capture script.
# --privileged + /dev mount is the simplest way to expose USB devices.

# Persistent HuggingFace cache so bert-base-uncased (GroundingDINO text encoder)
# is not re-downloaded every run. Offline flags skip the huggingface.co HEAD
# checks that otherwise hang on this host's broken IPv6 route.
HF_CACHE="${HF_CACHE:-/PublicSSD/iitp/hf_cache}"

docker run -it --rm \
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
  bash -c "cd /mnt && python main.py"
