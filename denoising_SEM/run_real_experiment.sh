#!/bin/bash
# =============================================================================
# One-click script: Download real SEM data, train all models, evaluate
# Run from denoising_SEM/ directory on the GPU server
# =============================================================================
set -e

echo "=============================================="
echo "Real SEM Denoising Experiment (NFFA-EUROPE)"
echo "=============================================="

# Step 1: Download and prepare dataset
echo ""
echo "[STEP 1/3] Preparing real SEM dataset..."
python data/prepare_nffa.py --max-images 200 --data-dir data/nffa --output-dir data/nffa_sem

# Step 2: Train all models on all noise levels
echo ""
echo "[STEP 2/3] Training all models on real data..."
python train_real.py --model all --noise all --epochs 100 --batch-size 16

# Step 3: Evaluate
echo ""
echo "[STEP 3/3] Evaluating all models..."
python evaluate_real.py --noise all

echo ""
echo "=============================================="
echo "ALL_DONE (REAL DATA EXPERIMENT)"
echo "=============================================="
