#!/bin/bash
# One full perception-eval pipeline: capture -> detect -> group.
# Usage: ./perception_eval/run_one.sh <run_name> [duration_seconds]
# Example: ./perception_eval/run_one.sh run_6 20

set -e

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

RUN_NAME=${1:?"usage: $0 <run_name> [duration_seconds]"}
DURATION=${2:-20}
ROOT=tmp_results/perception_eval_260520/$RUN_NAME

echo "=== capture ==="
./perception_eval/docker_capture.sh -o "$ROOT" --duration "$DURATION"

echo "=== eval ==="
docker run -i --rm --gpus all --ipc=host -v "$PWD":/mnt \
  --name iitp_eval chaehyeonsong/grounded_sam:latest \
  bash -c "cd /mnt && python main.py eval -i $ROOT"

echo "=== chown ==="
docker run --rm -v "$PWD":/mnt iitp_local:latest \
  chown -R "$(id -u):$(id -g)" "/mnt/$ROOT"

echo "=== group ==="
python3 main.py group -i "$ROOT"
