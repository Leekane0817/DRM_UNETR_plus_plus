#!/bin/bash
#
# Training script for UNETR_PP_SDRM on ACDC dataset
#
# Usage:
#   bash scripts/train.sh
#
# Prerequisites:
#   - unetr_plus_plus installed and on PYTHONPATH
#   - ACDC dataset prepared at $DATA_DIR
#   - This project root on PYTHONPATH
#
set -e

WORK_DIR=/home/featurize/work
DATA_DIR=$WORK_DIR/DATASET_Acdc/DATASET_Acdc
PROJECT_DIR=$WORK_DIR/acdc_sdrm_model

# nnU-Net style data paths
export PYTHONPATH="$WORK_DIR/unetr_plus_plus:$PROJECT_DIR:$PYTHONPATH"
export RESULTS_FOLDER="$PROJECT_DIR/output/train"
export unetr_pp_preprocessed="$DATA_DIR/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC"
export unetr_pp_raw_data_base="$DATA_DIR/unetr_pp_raw"

mkdir -p "$RESULTS_FOLDER"

echo "============================================"
echo "UNETR_PP_SDRM Training on ACDC"
echo "============================================"
echo "RESULTS_FOLDER: $RESULTS_FOLDER"
echo "Preprocessed: $unetr_pp_preprocessed"
echo "Project dir: $PROJECT_DIR"
echo "============================================"

cd "$WORK_DIR/unetr_plus_plus/unetr_pp/run"

python run_training.py \
    3d_fullres \
    unetr_pp_trainer_acdc_sdrm \
    1 \
    0 \
    --fp32

echo ""
echo "Training completed!"
echo "Results saved to: $RESULTS_FOLDER"
