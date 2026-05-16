#!/bin/bash
# ============================================================================
# Master training script for ScaleLogic.
#
# Single-node DAPO/GRPO RL fine-tuning of a Qwen3 base model on the synthetic
# logic-reasoning dataset produced by data_generation/aug_generator.py.
#
# All knobs are env-vars (with safe defaults).  See scripts/presets/*.sh for
# the exact configurations used in the paper.
# ============================================================================
#
# Required env-vars
#   TRAIN_FILE   path to the train .parquet produced by aug_generator.py
#   VAL_FILE     path to the matched test .parquet
#   BASE_MODEL   path to (or HF id of) the base model, e.g. Qwen/Qwen3-4B
#
# Common knobs (defaults in []):
#   N_GPUS              [8]    # GPUs on this node
#   ADV_ESTIMATOR       [grpo]
#   MAX_PROMPT_LEN      [8192]
#   MAX_RESPONSE_LEN    [8192]
#   N_RESP_PER_PROMPT   [8]
#   TRAIN_BSZ           [256]
#   GEN_BSZ             [384]
#   MINI_BSZ            [64]
#   LR                  [1e-6]
#   TOTAL_EPOCHS        [5]    # just an upper bound; runs early-stop via
#                              # ADAPTIVE_THRESHOLD long before this is reached
#   SAVE_FREQ           [8]
#   TEST_FREQ           [4]    # SAVE_FREQ / TEST_FREQ are automatically tightened
#                              # as val-acc approaches ADAPTIVE_THRESHOLD
#   ADAPTIVE_THRESHOLD  [0.90] # +trainer.adaptive_threshold (early-stop @ acc)
#
# Optional logging:
#   WANDB_PROJECT   [scalelogic]
#   WANDB_API_KEY   (export externally)
#   RUN_NAME        [local-<timestamp>]
#   CHECKPOINT_DIR  [./checkpoints/<WANDB_PROJECT>/<RUN_NAME>]
#
# Optional resume:
#   RESUME_CKPT_DIR     name of an existing run dir under CHECKPOINT_DIR's parent
# ----------------------------------------------------------------------------

set -e

# `python -m recipe.dapo.main_dapo` requires CWD to be the verl/ root so that
# `recipe` and `verl` resolve as top-level modules.  This script lives at
# verl/scripts/, so step one level up.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/.."

# ----- required inputs --------------------------------------------------------
: "${TRAIN_FILE:?TRAIN_FILE must be set (path to *_train.parquet)}"
: "${VAL_FILE:?VAL_FILE must be set (path to *_test.parquet)}"
: "${BASE_MODEL:?BASE_MODEL must be set (path or HF id of Qwen3 base)}"

# ----- defaults --------------------------------------------------------------
N_GPUS=${N_GPUS:-8}
ADV_ESTIMATOR=${ADV_ESTIMATOR:-grpo}
MAX_PROMPT_LEN=${MAX_PROMPT_LEN:-8192}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-8192}
N_RESP_PER_PROMPT=${N_RESP_PER_PROMPT:-8}
TRAIN_BSZ=${TRAIN_BSZ:-256}
GEN_BSZ=${GEN_BSZ:-384}
MINI_BSZ=${MINI_BSZ:-64}
LR=${LR:-1e-6}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
SAVE_FREQ=${SAVE_FREQ:-8}
TEST_FREQ=${TEST_FREQ:-4}
ADAPTIVE_THRESHOLD=${ADAPTIVE_THRESHOLD:-0.90}

WANDB_PROJECT=${WANDB_PROJECT:-scalelogic}
RUN_NAME=${RUN_NAME:-"local-$(date +%Y%m%d-%H%M%S)"}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"./checkpoints/${WANDB_PROJECT}/${RUN_NAME}"}
mkdir -p "${CHECKPOINT_DIR}"

if [[ -n "${RESUME_CKPT_DIR:-}" ]]; then
    WANDB_EXPERIMENT_NAME="${RESUME_CKPT_DIR}"
else
    WANDB_EXPERIMENT_NAME="${RUN_NAME}-${BASE_MODEL##*/}"
fi

# ----- runtime / sanity ------------------------------------------------------
export NCCL_DEBUG=WARN
export HYDRA_FULL_ERROR=1
export VLLM_USE_V1=1
unset ROCR_VISIBLE_DEVICES
export NCCL_IB_DISABLE=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}

env | egrep 'CUDA_VISIBLE|HIP_VISIBLE|ROCR_VISIBLE' || true
nvidia-smi -L || true

train_files="['${TRAIN_FILE}']"
test_files="['${VAL_FILE}']"

# ----- RL hyper-parameters (DAPO) --------------------------------------------
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 2))
overlong_penalty_factor=1.0
loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

temperature=1.0
top_p=1.0
top_k=-1

sp_size=1
gen_tp=1
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 2))
infer_ppo_max_token_len=$(((MAX_PROMPT_LEN + MAX_RESPONSE_LEN) * 2))
offload=True

# ----- Ray ------------------------------------------------------------------
ray stop --force || true
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/${USER:-runner}/ray}
mkdir -p "${RAY_TMPDIR}" "${RAY_TMPDIR}/spill"
export RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1
export TMPDIR="${RAY_TMPDIR}"

# ----- Launch ---------------------------------------------------------------
python -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=${ADV_ESTIMATOR} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    algorithm.filter_groups.enable=${enable_filter_groups} \
    algorithm.filter_groups.metric=${filter_groups_metric} \
    algorithm.filter_groups.max_num_gen_batches=${max_num_gen_batches} \
    +algorithm.dynamic_rollout.enable=False \
    data.train_files="${train_files}" \
    data.val_files="${test_files}" \
    data.prompt_key=prompt \
    data.truncation='right' \
    data.max_prompt_length=${MAX_PROMPT_LEN} \
    data.max_response_length=${MAX_RESPONSE_LEN} \
    data.train_batch_size=${TRAIN_BSZ} \
    data.gen_batch_size=${GEN_BSZ} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.strategy="fsdp" \
    actor_rollout_ref.actor.optim.lr=${LR} \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.warmup_style=constant \
    actor_rollout_ref.actor.optim.min_lr_ratio=0. \
    actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BSZ} \
    actor_rollout_ref.actor.ppo_micro_batch_size=null \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_micro_batch_size=null \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=${N_RESP_PER_PROMPT} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=null \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${top_p} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.model.path=${BASE_MODEL} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    reward_model.reward_manager=dapo \
    reward_model.overlong_buffer.enable=${enable_overlong_buffer} \
    reward_model.overlong_buffer.len=${overlong_buffer_len} \
    reward_model.overlong_buffer.penalty_factor=${overlong_penalty_factor} \
    trainer.logger=['console','wandb'] \
    trainer.project_name=${WANDB_PROJECT} \
    trainer.experiment_name=${WANDB_EXPERIMENT_NAME} \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=${N_GPUS} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    +trainer.adaptive_threshold=${ADAPTIVE_THRESHOLD} \
    +trainer.curiculum_sampling=False \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${CHECKPOINT_DIR}"
