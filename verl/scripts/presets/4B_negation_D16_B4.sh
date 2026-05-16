#!/bin/bash
# "+ Negation" tier: implications + ∧ + ¬.  Disjunction and quantifiers
# disabled.  Paper Table 1, + Negation at D=16, B=4.
#
# Generate the data first:
#   python data_generation/aug_generator.py \
#       --n_train 100000 --n_test 1000 \
#       --total_edges 64 --depth 16 --branches 4 \
#       --num_persons 1 --p_forall 0.0 --no_reuse \
#       --no_disjunction
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_DIR=${DATA_DIR:-./data}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/search_logic_64_16_4_1_0%forall_no_reuse_no_disjunction_train.parquet}
export VAL_FILE=${VAL_FILE:-${DATA_DIR}/search_logic_64_16_4_1_0%forall_no_reuse_no_disjunction_test.parquet}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B}
export WANDB_PROJECT=${WANDB_PROJECT:-scalelogic-negation}

bash "${SCRIPT_DIR}/../train.sh"
