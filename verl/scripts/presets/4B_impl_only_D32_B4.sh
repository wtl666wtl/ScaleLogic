#!/bin/bash
# "Impl. only" tier of the expressiveness ladder: implication chains, no
# operators (∧, ∨, ¬, ∀ all disabled), no rule reuse, single-person universe.
# Paper Table 1, Impl. only at D=32, B=4 (edges=128).
#
# Generate the data first:
#   python data_generation/aug_generator.py \
#       --n_train 100000 --n_test 1000 \
#       --total_edges 128 --depth 32 --branches 4 \
#       --num_persons 1 --p_forall 0.0 \
#       --no_reuse --no_disjunction --no_negation --no_conjunction
#
# That writes
#   ./search_logic_128_32_4_1_0%forall_no_reuse_no_disjunction_no_negation_no_conjunction_{train,test}.parquet
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_DIR=${DATA_DIR:-./data}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/search_logic_128_32_4_1_0%forall_no_reuse_no_disjunction_no_negation_no_conjunction_train.parquet}
export VAL_FILE=${VAL_FILE:-${DATA_DIR}/search_logic_128_32_4_1_0%forall_no_reuse_no_disjunction_no_negation_no_conjunction_test.parquet}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-4B}
export WANDB_PROJECT=${WANDB_PROJECT:-scalelogic-impl-only}

bash "${SCRIPT_DIR}/../train.sh"
