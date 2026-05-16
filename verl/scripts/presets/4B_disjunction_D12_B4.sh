#!/bin/bash
# "+ Disjunction" tier: implications + ∧ + ¬ + ∨.  Quantifiers disabled.
# Paper Table 1, + Disjunction at D=12, B=4.
#
# Generate the data first:
#   python data_generation/aug_generator.py \
#       --n_train 100000 --n_test 1000 \
#       --total_edges 48 --depth 12 --branches 4 \
#       --num_persons 1 --p_forall 0.0 --no_reuse
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_DIR=${DATA_DIR:-./data}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/search_logic_48_12_4_1_0%forall_no_reuse_train.parquet}
export VAL_FILE=${VAL_FILE:-${DATA_DIR}/search_logic_48_12_4_1_0%forall_no_reuse_test.parquet}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B}
export WANDB_PROJECT=${WANDB_PROJECT:-scalelogic-disjunction}

bash "${SCRIPT_DIR}/../train.sh"
