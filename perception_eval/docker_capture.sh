#!/bin/bash
# Run the iitp_local image with USB passthrough and invoke the capture-only
# command. All extra args (e.g. -i 0.5, -o some/dir) are forwarded to
# `main.py capture`. cd's to the project root so it works no matter where
# you invoke the script from.

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

docker run -i --rm \
  --network host \
  --gpus all \
  --ipc=host \
  --privileged \
  -v /dev:/dev \
  -v "$PWD":/mnt \
  --name iitp_capture \
  iitp_local:latest \
  bash -c "cd /mnt && python main.py capture $*"
