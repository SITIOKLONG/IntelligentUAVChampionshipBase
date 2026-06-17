#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

docker run -it --gpus all --net host --name drone_ws --rm \
  -v "${SCRIPT_DIR}/src:/drone_ws/src" \
  basic_dev
