#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-routebook/main.toml}"
CKPT="${2:-runs/policyuq.pt}"

python -m policyuq_vlm.summit evaluate \
    --config "${CONFIG}" \
    --checkpoint "${CKPT}"
