# ScaleLogic: How Deep Can LLMs Learn to Reason? Expressiveness Is Key

[![arXiv](https://img.shields.io/badge/arXiv-2605.06638-b31b1b.svg)](https://arxiv.org/abs/2605.06638)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://wtl666wtl.github.io/projects/scalelogic/)

**[Project Page](https://wtl666wtl.github.io/projects/scalelogic/)** · **[Interactive Proof Explorer](https://wtl666wtl.github.io/projects/scalelogic/#proof-explorer)** · **[Data Generation](data_generation/README.md)** · **[Training Guide](verl/scripts/README.md)**

We build a controlled synthetic logical reasoning testbed where the *reasoning depth* of a problem and the *logical expressiveness* of the language are both dialed independently. Training a model with RL until it reaches 90% accuracy, we find the required training compute follows a clean power law in depth, $T \propto D^{\gamma}$ — and the exponent $\gamma$ grows monotonically with logical expressiveness (from 1.04 to 2.60). RL *can* teach long-horizon reasoning, but the cost of doing so is governed by how expressive the target logic is.

Visit the [project page](https://wtl666wtl.github.io/projects/scalelogic/) for the research overview, results, and an interactive explorer that walks through generated proofs across the five logical settings.

## Task overview

Each problem is a set of natural language facts plus $B$ candidate conclusions; exactly one conclusion is provable, the other $B-1$ are made unprovable by corrupting a single axiom in their proof tree. The **reasoning depth** $D$ is the length of the provable chain; the **logical expressiveness** controls which logical operators may appear: $\text{Implication-only} \subset \text{+ Conjunction} \subset \text{+ Negation} \subset \text{+ Disjunction} \subset \text{+ Quantification}$.

![Task overview](assets/task_overview.png)

## Key findings

For every expressiveness setting, the RL training steps T required to reach 90% validation accuracy follow a power law in reasoning depth $D$ ($T \propto D^{\gamma}, R^2 \ge 0.99$). The fitted exponent $\gamma$ rises monotonically as more logical machinery is allowed, from near-linear ($\gamma \approx 1.04$) to strongly super-linear ($\gamma \approx 2.60$). And the downstream performances also depend strongly on the logical expressiveness of the training environment, with more expressive settings yielding larger and more compute-efficient transfer.

![Scaling law and per-setting exponent](assets/scaling_panel.png)

## Repository layout

```
.
├── data_generation/                          # 3-file synthetic-task generator
│   ├── aug_generator.py                      #   entry point (parquet emitter)
│   ├── generator.py                          #   graph generator
│   ├── graph_to_qa.py                        #   verbalize graph to MCQ
│   └── README.md
│
└── verl/                                     # vendored copy of verl 0.5.0 (Apache 2.0)
    ├── recipe/dapo/dapo_ray_trainer.py       # modified: adaptive early-stop hooks
    ├── verl/utils/reward_score/...     	  # added:    ScaleLogic reward scorer
    ├── scripts/                              # added:    training launchers
    │   ├── train.sh                          #             parameterized launcher
    │   ├── presets/                          #             one wrapper per setting
    │   └── README.md
    └── ...                                   # everything else: byte-identical upstream
```

## Environment

Run the commands below from the repository root. Data generation needs `pip install datasets`.

Training uses the vendored verl 0.5.0 and requires Python 3.10+, a Linux environment with NVIDIA GPUs, and a compatible CUDA/PyTorch/vLLM stack. Follow the bundled [installation guide](verl/docs/start/install.rst) to prepare that stack, then install this repository's copy with `pip install -e ./verl`. The launcher defaults to **8 GPUs**; set `N_GPUS` to match your machine and adjust batch sizes and token limits to fit GPU memory. See the [training guide](verl/scripts/README.md) for the available controls.

## Quick start

### 1.  Generate a dataset

```bash
python data_generation/aug_generator.py \
    --n_train 100000 --n_test 1000 \
    --total_edges 40 --depth 10 --branches 4 \
    --num_persons 2 --p_forall 0.5
```

This emits `search_logic_40_10_4_2_50%forall_train.parquet` and `search_logic_40_10_4_2_50%forall_test.parquet` in the current directory. Examples cover reasoning depths up to `--depth`; see the [data-generation guide](data_generation/README.md) for all flags and expressiveness settings.

### 2.  Train

```bash
# Point the preset to the dataset generated above.
DATA_DIR="$(pwd)" \
bash verl/scripts/presets/4B_quantification_D10_B4.sh

# Or set the inputs and training options explicitly.
TRAIN_FILE="$(pwd)/search_logic_40_10_4_2_50%forall_train.parquet" \
VAL_FILE="$(pwd)/search_logic_40_10_4_2_50%forall_test.parquet" \
BASE_MODEL=Qwen/Qwen3-4B \
MAX_RESPONSE_LEN=8192 \
bash verl/scripts/train.sh
```

Use absolute dataset paths: the launcher switches its working directory to `verl/`, so relative paths are resolved there. Presets default to `DATA_DIR=./data` (that is, `verl/data/`); the explicit `DATA_DIR` above points to the generated files instead.

The launcher enables console and W&B logging. Authenticate with `wandb login` or set `WANDB_API_KEY` before training; an unset API key does not automatically disable W&B. See the [logging instructions](verl/scripts/README.md#wandb).

The trainer tracks an early-stop threshold (default 90% validation accuracy). The number of training steps at the time the threshold is hit is what we call `T_steps` in the paper. Checkpoints are saved as FSDP shards; merge them with `python -m verl.model_merger merge --backend fsdp ...` and evaluate with any standard benchmark harness.

## Citation

```bibtex
@article{wang2026scalelogic,
  title={How Deep Can LLMs Learn to Reason? Expressiveness Is Key},
  author={Wang, Tianle and Wang, Zhaoyang and Lan, Guangchen and Wei, Xinpeng and Zhang, Sipeng and Qiu, Guanwen and Saparov, Abulhair},
  journal={arXiv preprint arXiv:2605.06638},
  year={2026},
  url={https://arxiv.org/abs/2605.06638}
}
```

## License

The vendored verl tree is Apache 2.0 (see [verl/LICENSE](verl/LICENSE)). Our additions (data generators, modified DAPO trainer, reward scorer, training scripts) are released under MIT — see [LICENSE](LICENSE).
