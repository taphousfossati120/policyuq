#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-routebook/main.toml}"

torchrun --nproc_per_node=4 --standalone \
    -m policyuq_vlm.summit train \
    --config "${CONFIG}" \
    --out runs/policyuq.pt

# Single-node SLURM example:
#   srun --gres=gpu:a100:4 bash approaches/launch_train.sh routebook/main.toml
