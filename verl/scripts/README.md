# Training scripts

All paper experiments are launched through one parameterized script, `train.sh`, plus a handful of preset wrappers under `presets/` that fix the data file and a few size-specific knobs.

## Layout


| File                                  | Purpose                                                                |
| ------------------------------------- | ---------------------------------------------------------------------- |
| `train.sh`                            | Master DAPO / GRPO launcher. All knobs are env-vars (see top of file). |
| `presets/4B_impl_only_D32_B4.sh`      | "Implication only" (no operators)                                      |
| `presets/4B_conjunction_D16_B4.sh`    | "+ Conjunction" (∧)                                                    |
| `presets/4B_negation_D16_B4.sh`       | "+ Negation" (∧ ¬)                                                     |
| `presets/4B_disjunction_D12_B4.sh`    | "+ Disjunction" (∧ ¬ ∨)                                                |
| `presets/4B_quantification_D10_B4.sh` | "+ Quantification" (∧ ¬ ∨ ∀)                                           |
| `presets/8B_quantification_D14_B4.sh` | Same as above, Qwen3-8B base.                                          |


## Reproducing the scaling curves

Each preset trains *one* point on the (depth, T_steps) scaling curve for its expressiveness tier. To reproduce a full curve, regenerate the data and call `train.sh` directly with the new depth/edges/flags. Example for "+ Conjunction" at the six paper depths {8,12,16,20,24,28}:

```bash
for D in 8 12 16 20 24 28; do
    EDGES=$((D * 4))   # branches = 4
    python data_generation/aug_generator.py \
        --n_train 100000 --n_test 1000 \
        --total_edges ${EDGES} --depth ${D} --branches 4 \
        --num_persons 1 --p_forall 0.0 \
        --no_reuse --no_disjunction --no_negation
    TRAIN_FILE=./search_logic_${EDGES}_${D}_4_1_0%forall_no_reuse_no_disjunction_no_negation_train.parquet \
    VAL_FILE=./search_logic_${EDGES}_${D}_4_1_0%forall_no_reuse_no_disjunction_no_negation_test.parquet  \
    BASE_MODEL=Qwen/Qwen3-4B \
    WANDB_PROJECT=scalelogic-conjunction \
    bash verl/scripts/train.sh
done
```

The trainer writes the per-checkpoint accuracy to W&B; the "T_steps to 90 % accuracy" used in the paper is read off that curve.

## Reward / data source registration

`train.sh` always passes `reward_model.reward_manager=dapo`. The reward function dispatch is in `verl/verl/utils/reward_score/__init__.py`: rows whose `data_source` is `search_logic_multi_choice` (which is what `aug_generator.py` emits) are routed to `verl/verl/utils/reward_score/search.py`, the reward scorer we added on top of upstream verl.

## WandB

Export your key before launching:

```bash
export WANDB_API_KEY=...   # never commit this
export WANDB_PROJECT=scalelogic
```

If `WANDB_API_KEY` is unset the run will fall back to console-only logging.

## Resuming

```bash
RUN_DIR=<your-run-dir>          # e.g. the auto-named local-<timestamp>-Qwen3-4B
RESUME_CKPT_DIR=${RUN_DIR} \
CHECKPOINT_DIR=./checkpoints/scalelogic/${RUN_DIR} \
bash verl/scripts/presets/4B_quantification_D10_B4.sh
```

(`train.sh` passes `trainer.resume_mode=auto` and looks for `latest_checkpointed_iteration.txt` under `CHECKPOINT_DIR`.)