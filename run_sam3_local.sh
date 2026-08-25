#!/usr/bin/env bash
# Run the live pipeline with the SAM3 TensorRT backend on the machine the
# RealSense is plugged into.
#
# The SAM3 runtime (tensorrt / pycuda / pyrealsense2) lives in a conda env on
# the host, not in an image -- see README.md. The container is only here for
# USB pass-through: the camera nodes are root-owned and this host has no
# librealsense udev rules, so a plain host run gets "No device connected".
# The env is bind-mounted at its own absolute path because conda hard-codes it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

SAM3_ENV="${SAM3_ENV:-/PublicSSD/iitp/sam3_env/miniforge3/envs/sam3}"
IMAGE="${SAM3_IMAGE:-iitp_local:latest}"

if [[ ! -x "${SAM3_ENV}/bin/python" ]]; then
  echo "SAM3 env not found: ${SAM3_ENV}/bin/python" >&2
  echo "Build it first (see the SAM3 TensorRT backend section of README.md)." >&2
  exit 1
fi

# -t only when attached to a terminal, so nohup/CI runs do not fail.
tty_flags=(-i)
[[ -t 0 ]] && tty_flags=(-i -t)

exec docker run "${tty_flags[@]}" --rm \
  --network host \
  --gpus all \
  --ipc=host \
  --privileged \
  -v /dev:/dev \
  -v "${SAM3_ENV}":"${SAM3_ENV}" \
  -v "${ROOT}":"${ROOT}" \
  -w "${ROOT}" \
  -e CUDA_MODULE_LOADING=LAZY \
  --name iitp_sam3 \
  "${IMAGE}" \
  "${SAM3_ENV}/bin/python" main.py --backend sam3 "$@"
