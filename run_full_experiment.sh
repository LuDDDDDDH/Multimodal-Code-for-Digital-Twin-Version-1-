# run_full_experiment.sh
#!/bin/bash

# ====================================================================================
#      MAIN CONTROLLER SCRIPT FOR END-TO-END REPEATED K-FOLD EXPERIMENT (SCHEME A)
# ====================================================================================
#
# USAGE:
#   1. Make sure you are in the correct conda environment:
#      $ conda activate multimodal_env_v2
#   2. Give execute permission to this script:
#      $ chmod +x run_full_experiment.sh
#   3. Run the script (preferably inside a tmux or screen session):
#      $ ./run_full_experiment.sh
#
# This script automates the entire Scheme A workflow. It will loop through a list of
# random seeds and, for each seed, execute the full pipeline:
#   - Train K heavy models.
#   - Extract K sets of CT features.
#   - Train K light models.
#   - Run SHAP analysis for the K light models.
#
# ====================================================================================


# --- 1. Configuration ---

# Define the number of repeats by listing the seeds. 
# To run 5 repeats, add two more seeds.
SEEDS=(42 123 2024)
# SEEDS=(42)
# Define the number of folds used in the experiment (must match config.py).
# This is used to loop the correct number of times for feature extraction and SHAP.
K_FOLDS=3 

# Define the main parent directory for all experiment outputs.
BASE_OUTPUT_DIR="outputs/full_experiment"


# --- 2. Safety and Setup ---

# The 'set -e' command ensures that the script will exit immediately if any
# command fails. This is a crucial safety measure to prevent the script from
# continuing with incorrect or incomplete data, thus saving valuable time.
set -e

# Clear and descriptive start message.
echo "============================================================"
echo "      STARTING FULL END-TO-END REPEATED K-FOLD EXPERIMENT"
echo "============================================================"
echo "Scheme: A (Full End-to-End Repeats)"
echo "Number of Repeats: ${#SEEDS[@]}"
echo "Number of Folds per Repeat: ${K_FOLDS}"
echo "Base Output Directory: ${BASE_OUTPUT_DIR}"
echo "------------------------------------------------------------"


# --- 3. Main Loop: Iterate through each seed for a full repeat ---

for seed in "${SEEDS[@]}"; do
    
    # Define the specific directory for all outputs of the current repeat.
    REPEAT_DIR="${BASE_OUTPUT_DIR}/repeat_seed_${seed}"

    # Print a clear header for the current repeat.
    echo "" # Add a blank line for better readability in the log.
    echo "************************************************************"
    echo ">>> STARTING REPEAT WITH SEED: ${seed}"
    echo ">>> Output Directory: ${REPEAT_DIR}"
    echo "************************************************************"

    # --- Step 1 of 4: Train K heavy models for the current seed ---
    echo ""
    echo "--- [Step 1/4] Training Heavy Models ---"
    python train_heavy.py --seed ${seed}
    echo "--- Heavy model training COMPLETED for seed ${seed} ---"

    # --- Step 2 of 4: Extract CT features for each of the K folds ---
    echo ""
    echo "--- [Step 2/4] Extracting CT Features ---"
    # This loop runs K_FOLDS times (e.g., for k=1, 2, 3).
    for k in $(seq 1 ${K_FOLDS}); do
        echo "  - Extracting features for Fold ${k}..."
        python extract_ct_features.py \
            --heavy_model_path "${REPEAT_DIR}/best_heavy_model_fold_${k}.pth" \
            --output_path "${REPEAT_DIR}/precomputed_ct_features_fold_${k}.csv"
    done
    echo "--- CT feature extraction COMPLETED for seed ${seed} ---"

    # --- Step 3 of 4: Train K light models using the fold-specific features ---
    echo ""
    echo "--- [Step 3/4] Training Light Models ---"
    python train_light.py --seed ${seed}
    echo "--- Light model training COMPLETED for seed ${seed} ---"
    
    # --- Step 4 of 4: Run SHAP analysis for each of the K light models ---
    echo ""
    echo "--- [Step 4/4] Running SHAP Analysis ---"
    # This loop runs K_FOLDS times.
    for k in $(seq 1 ${K_FOLDS}); do
        echo "  - Running SHAP for Fold ${k}..."
        python run_explainability_analysis.py \
            --fold ${k} \
            --input_dir ${REPEAT_DIR}
    done
    echo "--- SHAP analysis COMPLETED for seed ${seed} ---"

    # Print a clear footer for the completed repeat.
    echo ""
    echo "************************************************************"
    echo ">>> REPEAT WITH SEED ${seed} FULLY COMPLETED"
    echo "************************************************************"
done


# --- 4. Final Success Message ---

echo ""
echo "============================================================"
echo "      ALL REPEATS COMPLETED SUCCESSFULLY!"
echo "      You are now ready for the final aggregation analysis."
echo "      Check the '${BASE_OUTPUT_DIR}' directory for all results."
echo "============================================================"

!/bin/bash

# SEEDS=(42 123 2025)
SEEDS=(42)  # ★ 临时只跑两个
K_FOLDS=3
BASE_OUTPUT_DIR="outputs/full_experiment"
set -e

echo "============================================================"
echo "      RE-GENERATING SHAP PACKAGES ONLY"
echo "============================================================"

for seed in "${SEEDS[@]}"; do
    REPEAT_DIR="${BASE_OUTPUT_DIR}/repeat_seed_${seed}"
    echo ""
    echo "************************************************************"
    echo ">>> PROCESSING REPEAT WITH SEED: ${seed}"
    echo "************************************************************"

    # --- 跳过步骤 1, 2, 3 ---
    # echo "--- [Step 1/4] SKIPPING Heavy Models ---"
    # echo "--- [Step 2/4] SKIPPING CT Features ---"
    # echo "--- [Step 3/4] SKIPPING Light Models ---"

    # --- 只执行步骤 4: 运行SHAP分析 ---
    echo "--- [Step 4/4] Running SHAP Analysis ---"
    for k in $(seq 1 ${K_FOLDS}); do
        echo "  - Running SHAP for Fold ${k}..."
        python run_explainability_analysis.py \
            --fold ${k} \
            --input_dir ${REPEAT_DIR}
    done
    echo "--- SHAP analysis COMPLETED for seed ${seed} ---"

done

echo "============================================================"
echo "      ALL SHAP PACKAGES HAVE BEEN RE-GENERATED."
echo "============================================================"