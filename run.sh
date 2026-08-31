#!/usr/bin/env bash
set -euo pipefail

cd /code
export CUDA_VISIBLE_DEVICES=""
export VF_ADVJPEG_DEEPROBUST_RESNET="/data/third_party/DeepRobust/deeprobust/image/netmodels/resnet.py"
export VF_ADVJPEG_ROBUSTBENCH_ROOT="/data/third_party/RobustBench"
export VF_ADVJPEG_AUTOATTACK_ROOT="/data/third_party/AutoAttack"
export MPLBACKEND=Agg
mkdir -p /results
python -u scripts/run_public_release.py --config configs/public_release.yaml
