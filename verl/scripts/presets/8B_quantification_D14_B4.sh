#!/bin/bash
# 8B "+ Quantification" at D=14, B=4.  Used in the model-scale comparison.
#
# Generate the data first:
#   python data_generation/aug_generator.py \
#       --n_train 100000 --n_test 1000 \
#       --total_edges 56 --depth 14 --branches 4 \
#       --num_persons 2 --p_forall 0.5
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export DATA_DIR=${DATA_DIR:-./data}
export TRAIN_FILE=${TRAIN_FILE:-${DATA_DIR}/search_logic_56_14_4_2_50%forall_train.parquet}
export VAL_FILE=${VAL_FILE:-${DATA_DIR}/search_logic_56_14_4_2_50%forall_test.parquet}
export BASE_MODEL=${BASE_MODEL:-Qwen/Qwen3-8B}
export WANDB_PROJECT=${WANDB_PROJECT:-scalelogic-8B-quantification}

bash "${SCRIPT_DIR}/../train.sh"
