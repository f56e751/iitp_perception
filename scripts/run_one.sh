#!/bin/bash
# One full perception-eval pipeline: capture -> detect -> group.
# Usage: ./scripts/run_one.sh <run_name> [duration_seconds]
# Example: ./scripts/run_one.sh run_6 20

set -e

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."

RUN_NAME=${1:?"usage: $0 <run_name> [duration_seconds]"}
DURATION=${2:-20}
# Date folder defaults to today; override with DATE_DIR=perception_eval_YYMMDD
DATE_DIR=${DATE_DIR:-perception_eval_$(date +%y%m%d)}
ROOT=perception_tests/results/$DATE_DIR/$RUN_NAME

echo "=== capture ==="
./scripts/docker_capture.sh -o "$ROOT" --duration "$DURATION"

echo "=== eval ==="
docker run -i --rm --gpus all --ipc=host -v "$PWD":/mnt \
  --name iitp_eval chaehyeonsong/grounded_sam:latest \
  bash -c "cd /mnt && python scripts/eval_detector.py -i $ROOT"

echo "=== chown ==="
docker run --rm -v "$PWD":/mnt iitp_local:latest \
  chown -R "$(id -u):$(id -g)" "/mnt/$ROOT"

echo "=== group ==="
python3 scripts/group_by_object.py -i "$ROOT"
