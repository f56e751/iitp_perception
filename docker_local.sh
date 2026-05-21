#!/bin/bash
# Run grounded_sam container with RealSense USB passthrough + local capture script.
# --privileged + /dev mount is the simplest way to expose USB devices.

docker run -it --rm \
  --network host \
  --gpus all \
  --ipc=host \
  --privileged \
  -v /dev:/dev \
  -v "$PWD":/mnt \
  --name iitp_local \
  iitp_local:latest \
  bash -c "cd /mnt && python main.py"
