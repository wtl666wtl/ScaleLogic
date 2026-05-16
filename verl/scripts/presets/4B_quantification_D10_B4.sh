#!/bin/bash
# "+ Quantification" tier (the full first-order setting): implications +
# ∧ + ¬ + ∨ + ∀ (universal quantifier sampled with p_forall=0.5).
# Paper Table 1, + Quantification at D=10, B=4.
#
# Generate the data first:
#   python data_generation/aug_generator.py \
#       --n_train 100000 --n_test 1000 \
#       --total_edges 40 --depth 10 --branches 4 \
#       --num_persons 2 --p_forall 0.5
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_DIR=${DATA_DIR:-./data}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/search_logic_40_10_4_2_50%forall_train.parquet}
export VAL_FILE=${VAL_FILE:-${DATA_DIR}/search_logic_40_10_4_2_50%forall_test.parquet}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B}
export WANDB_PROJECT=${WANDB_PROJECT:-scalelogic-quantification}

bash "${SCRIPT_DIR}/../train.sh"
