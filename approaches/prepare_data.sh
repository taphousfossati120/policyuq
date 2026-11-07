#!/usr/bin/env bash
set -euo pipefail

# Expected on-disk roots (set in routebook/*.toml [data].root):
#   data/pi_cai        PI-CAI public training (CC BY-NC 4.0)
#   data/prostatex     PROSTATEx (CC BY 3.0)
#   data/prostate158   Prostate158 (139/19 split)
#
# Each root holds per-case T2W / DWI / ADC volumes resampled to 192x192x48
# with a lesion bounding box and a csPCa label. Dataset-specific readers
# must be implemented before training or evaluation.

mkdir -p data/pi_cai data/prostatex data/prostate158
echo "place preprocessed cases under data/<dataset>/ before training on real cohorts"
