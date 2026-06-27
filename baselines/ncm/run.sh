#!/usr/bin/env bash
# =============================================================================
# run.sh — Single-command launcher for NCM training
#
# Usage:
#   bash run.sh                          # default: NCM-ML, CIFAR-100, ResNet-50
#   MODE=ncm bash run.sh                 # plain NCM (no metric learning)
#   MODE=ncm_mm bash run.sh              # NCM with sub-means (EM)
#   DATASET=imagefolder ROOT=/data bash run.sh   # ImageNet-style dataset
# =============================================================================

set -euo pipefail

# ---------- configurable via environment variables --------------------------
MODE="${MODE:-ncm_ml}"          # ncm | ncm_ml | ncm_mm
DATASET="${DATASET:-cifar100}"  # cifar10 | cifar100 | imagefolder
ROOT="${ROOT:-./data}"
METRIC_DIM="${METRIC_DIM:-256}"
NUM_SUB_MEANS="${NUM_SUB_MEANS:-1}"
EPOCHS="${EPOCHS:-30}"
LR="${LR:-0.01}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
CKPT_DIR="${CKPT_DIR:-checkpoints/ncm}"
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "================================================================"
echo "  NCM Baseline — Mensink et al. TPAMI 2013"
echo "  Mode        : ${MODE}"
echo "  Dataset     : ${DATASET}  (root: ${ROOT})"
echo "  metric_dim  : ${METRIC_DIM}"
echo "  sub_means   : ${NUM_SUB_MEANS}"
echo "  Epochs      : ${EPOCHS}"
echo "  LR          : ${LR}"
echo "  Seed        : ${SEED}"
echo "  Device      : ${DEVICE}"
echo "  Checkpoints : ${CKPT_DIR}"
echo "================================================================"

python "${SCRIPT_DIR}/train.py" \
    --config "${SCRIPT_DIR}/config.yaml" \
    --override \
        training.mode="${MODE}" \
        data.dataset="${DATASET}" \
        data.root="${ROOT}" \
        model.metric_dim="${METRIC_DIM}" \
        model.num_sub_means="${NUM_SUB_MEANS}" \
        training.epochs="${EPOCHS}" \
        training.lr="${LR}" \
        seed="${SEED}" \
        device="${DEVICE}" \
        checkpoint_dir="${CKPT_DIR}"

echo ""
echo "Training complete. Best checkpoint: ${CKPT_DIR}/ncm_best.pt"
