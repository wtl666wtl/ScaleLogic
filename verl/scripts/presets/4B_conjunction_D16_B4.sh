#!/bin/bash
# "+ Conjunction" tier: implications + conjunction (∧).  Negation, disjunction,
# and quantifiers are disabled.  Paper Table 1, + Conjunction at D=16, B=4.
#
# Generate the data first:
#   python data_generation/aug_generator.py \
#       --n_train 100000 --n_test 1000 \
#       --total_edges 64 --depth 16 --branches 4 \
#       --num_persons 1 --p_forall 0.0 --no_reuse \
#       --no_disjunction --no_negation
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_DIR=${DATA_DIR:-./data}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/search_logic_64_16_4_1_0%forall_no_reuse_no_disjunction_no_negation_train.parquet}
export VAL_FILE=${VAL_FILE:-${DATA_DIR}/search_logic_64_16_4_1_0%forall_no_reuse_no_disjunction_no_negation_test.parquet}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B}
export WANDB_PROJECT=${WANDB_PROJECT:-scalelogic-conjunction}

bash "${SCRIPT_DIR}/../train.sh"
