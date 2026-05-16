# Synthetic Logic Reasoning Task Generator

This folder contains the **data-generation code** used to produce the synthetic reasoning datasets evaluated in the paper.  Running it locally reproduces every train / test parquet file referenced in the main text and appendix.

---

## Contents

| File | Purpose |
| --- | --- |
| `aug_generator.py` | **Entry point.**  Builds train / test datasets with a uniform distribution over reasoning depths via subgraph augmentation, then dumps them to parquet. |
| `generator.py`        | Core graph-construction routines (sampling rules, branching, depth control, operator toggles). |
| `graph_to_qa.py`      | Verbalizes a logic graph into a natural-language multiple-choice question and the corresponding boolean / answer string. |

---

## Requirements

```bash
python >= 3.9
pip install datasets    # for parquet I/O via the HuggingFace `datasets` API
```

---

## Quick start

Generate a dataset with branching factor `B = 4`, reasoning depth `D = 8`, total of `4 * 8 = 32` edges, 2 persons in the universe, "+ Quantification" setting, 1k train and 1k test samples:

```bash
python aug_generator.py \
    --n_train 100000 --n_test 1000 \
    --total_edges 32 --depth 8 --branches 4
```

The filename pattern is `search_logic_<total_edges>_<depth>_<branches>_<num_persons>_<suffix>_{train,test}.parquet`.

---

## Reproducing the five paper settings

Each row of Table 1 / Figure 2 corresponds to a specific operator subset ("Implication only", "+ Conjunction", "+ Negation", "+ Disjunction", "+ Quantification").  Switch operators on or off via the corresponding `--no_*` flags and `--p_forall`.  For example:

```bash
# Implication only
--no_conjunction --no_disjunction --no_negation --p_forall 0.0 --no_reuse --num_persons 1

# + Conjunction
--no_disjunction --no_negation --p_forall 0.0 --no_reuse --num_persons 1

# + Negation
--no_disjunction --p_forall 0.0 --no_reuse --num_persons 1

# + Disjunction
--p_forall 0.0 --no_reuse --num_persons 1

# + Quantification
--p_forall 0.5
```

All flags accepted by `aug_generator.py`:

| Flag | Default | Meaning |
| --- | --- | --- |
| `--n_train` | 100000 | # of training examples |
| `--n_test`  | 1000   | # of test examples |
| `--total_edges` | 16 | total edges in the logic graph (must equal `branches * depth`) |
| `--depth`   | 8      | maximum reasoning depth `D` |
| `--branches`| 2      | branching factor `B` (number of candidate conclusions per question) |
| `--num_persons` | 2  | size of the universe |
| `--max_arity`   | 2  | maximum arity of a relation |
| `--p_small_arity` | 1.0 | probability of using arity-1 relations |
| `--p_forall`      | 0.5 | probability of inserting a `∀` quantifier |
| `--no_disjunction`| off | disable `∨` |
| `--no_negation`   | off | disable `¬` |
| `--no_conjunction`| off | disable `∧` |
| `--no_reuse`      | off | forbid reusing intermediate nodes across branches |
| `--seed`          | 0   | random seed |
| `--suffix`        | ""  | appended to the output filename |

---

## Output schema

Each parquet row has the following fields:

| field | type | content |
| --- | --- | --- |
| `index` | `int` | 0-based row index |
| `data_source` | `str` | `"search_logic_multi_choice"` |
| `prompt` | `list[dict]` | chat-format prompt: `[{"content": <question>, "role": "user"}]` |
| `reward_model` | `dict` | `{"style": "rule", "ground_truth": [<answer-string>]}` |
| `extra_info` | `dict` | depth metadata (`target_depth`, `num_edges`, ...) used for stratified evaluation |
