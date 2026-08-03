#!/bin/bash
#
# Validation script for UNETR_PP_SDRM on ACDC dataset
#
# Usage:
#   bash scripts/validate.sh
#
set -e

WORK_DIR=/home/featurize/work
DATA_DIR=$WORK_DIR/DATASET_Acdc/DATASET_Acdc
PROJECT_DIR=$WORK_DIR/acdc_sdrm_model

export PYTHONPATH="$WORK_DIR/unetr_plus_plus:$PROJECT_DIR:$PYTHONPATH"
export RESULTS_FOLDER="$PROJECT_DIR/output/train"
export unetr_pp_preprocessed="$DATA_DIR/unetr_pp_raw/unetr_pp_raw_data/Task01_ACDC"
export unetr_pp_raw_data_base="$DATA_DIR/unetr_pp_raw"

echo "============================================"
echo "SDRM ACDC Validation (fold 0)"
echo "============================================"

cd "$WORK_DIR/unetr_plus_plus/unetr_pp/run"

python run_training.py \
    3d_fullres \
    unetr_pp_trainer_acdc_sdrm \
    1 \
    0 \
    --fp32 \
    -val \
    --valbest

echo ""
echo "Done! Summary at: $RESULTS_FOLDER/unetr_pp/3d_fullres/Task001_ACDC/unetr_pp_trainer_acdc_sdrm__unetr_pp_Plansv2.1/fold_0/validation_raw/summary.json"
