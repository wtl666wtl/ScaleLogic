# Training scripts

Training launchers consist of one parameterized script, `train.sh`, plus preset wrappers under `presets/` that fix the data file and a few size-specific knobs. See the [project page](https://wtl666wtl.github.io/projects/scalelogic/) for the research overview and results.

Run the examples below from the repository root. `train.sh` changes its working directory to `verl/`; use absolute paths for `TRAIN_FILE`, `VAL_FILE`, and any local `BASE_MODEL` directory. After generating a dataset in the repository root, set `DATA_DIR="$(pwd)"` when calling a preset. Without this override, presets look for data under `verl/data/`.

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
    TRAIN_FILE="$(pwd)/search_logic_${EDGES}_${D}_4_1_0%forall_no_reuse_no_disjunction_no_negation_train.parquet" \
    VAL_FILE="$(pwd)/search_logic_${EDGES}_${D}_4_1_0%forall_no_reuse_no_disjunction_no_negation_test.parquet" \
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

The launcher always enables both console and W&B logging. You can also authenticate with `wandb login`; leaving `WANDB_API_KEY` unset does not automatically switch to console-only logging.

## Resuming

```bash
RUN_DIR=local-20260929-120000   # replace with the existing checkpoint directory name
DATA_DIR="$(pwd)" \
RESUME_CKPT_DIR=${RUN_DIR} \
CHECKPOINT_DIR="$(pwd)/verl/checkpoints/scalelogic-quantification/${RUN_DIR}" \
bash verl/scripts/presets/4B_quantification_D10_B4.sh
```

`train.sh` passes `trainer.resume_mode=auto` and looks for `latest_checkpointed_iteration.txt` under `CHECKPOINT_DIR`. The default checkpoint path is `verl/checkpoints/<WANDB_PROJECT>/<RUN_NAME>`; the `-Qwen3-4B` suffix is used for the W&B experiment name, not the checkpoint directory. `RESUME_CKPT_DIR` changes the experiment name; `CHECKPOINT_DIR` selects the checkpoint to resume.
