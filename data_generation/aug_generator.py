"""Build train / test parquet datasets for the ScaleLogic synthetic
reasoning task.

`generate_uniform_depth_samples` constructs `B` independent subgraphs per
example (one is the "true" branch whose root is the gold answer, the others
are perturbed so that they evaluate to a different label), pads them up to
`total_edges` implication edges of distractor rules, and verbalizes the
whole bundle into a multiple-choice prompt via `graphs_to_qa`.

The script writes two parquet files in the current directory; the filename
encodes every knob that affects the resulting distribution so that runs are
reproducible from the filename alone (see the bottom of `__main__`).

CLI example (`+ Quantification` setting at depth 8, branching 4):

    python aug_generator.py --n_train 1000 --n_test 1000 \
        --total_edges 32 --depth 8 --branches 4 \
        --num_persons 2 --p_forall 0.5
"""

import argparse
import copy
import random
from typing import List, Dict, Any, Tuple

from graph_to_qa import graphs_to_qa
from generator import (
    generate_logic_graph_batch_with_uniform_depth,
    get_implication_edges_count,
)


def generate_uniform_depth_samples(
    n_train: int,
    n_test: int,
    num_persons: int,
    total_edges: int,
    branches: int,
    max_depth: int,
    max_arity: int = 4,
    no_reuse: bool = False,
    no_disjunction: bool = False,
    no_negation: bool = False,
    no_conjunction: bool = False,
    p_small_arity: float = 0.9,
    p_forall: float = 0.3,
    p_negate_node: float = 0.3,
    p_sibling_neg: float = 0.5,
    seed: int = 0,
) -> Tuple[List[Dict], List[Dict], List[Dict], List[Dict]]:
    """Generate `n_train + n_test` samples with a uniform distribution
    over reasoning depth in [1, max_depth].

    Each sample is a *bundle* of `branches` subgraphs that share a final
    prompt: one is the correct branch (root = True), the rest are perturbed
    so their roots are False or otherwise inconsistent.  After bundling, the
    graph is padded with distractor implication edges up to (a random count
    just below) `total_edges`.

    Returns
    -------
    train_graphs, train_extra_info, test_graphs, test_extra_info
        The two lists are aligned: `extra_info[i]` carries per-sample
        metadata (`target_depth`, `root_node_ids`, `root_node_values`, ...)
        consumed by `graph_to_qa.graphs_to_qa`.
    """
    if seed is not None:
        random.seed(seed)

    # ---- per-depth budget --------------------------------------------------
    n_train_per_depth = n_train // max_depth
    n_test_per_depth  = n_test  // max_depth
    train_remainder   = n_train %  max_depth
    test_remainder    = n_test  %  max_depth

    print(
        f"Generating {n_train} train + {n_test} test samples "
        f"(uniform over depth 1..{max_depth})"
    )
    print(
        f"Per depth: ~{n_train_per_depth} train + {n_test_per_depth} test"
    )
    print("-" * 32)

    # ---- one pool of base graphs over all depths --------------------------
    max_edges_per_subgraph = total_edges // branches
    base_graphs = generate_logic_graph_batch_with_uniform_depth(
        n=n_train + n_test,
        num_persons=num_persons,
        max_edges=max_edges_per_subgraph,
        max_depth=max_depth,
        max_arity=max_arity,
        no_reuse=no_reuse,
        no_disjunction=no_disjunction,
        no_negation=no_negation,
        no_conjunction=no_conjunction,
        p_small_arity=p_small_arity,
        p_forall=p_forall,
        p_negate_node=p_negate_node,
        p_sibling_neg=p_sibling_neg,
        root_neg=False,
        seed=seed,
    )

    # bucket by depth so we can later sample from a specific depth bin
    depth_graphs: List[List[Dict]] = [[] for _ in range(max_depth + 1)]
    for graph in base_graphs:
        depth_graphs[graph["max_depth"]].append(graph)

    train_graphs, train_extra_info = [], []
    test_graphs,  test_extra_info  = [], []

    for depth in range(1, max_depth + 1):
        n_train_this = n_train_per_depth + (1 if depth <= train_remainder else 0)
        n_test_this  = n_test_per_depth  + (1 if depth <= test_remainder  else 0)
        n_total_this = n_train_this + n_test_this

        print(
            f"Depth {depth:>2}: assembling {n_train_this} train + "
            f"{n_test_this} test bundles"
        )

        for i in range(n_total_this):
            # ---- 1. Pick `branches` independent subgraphs of this depth ----
            num_subgraphs = branches
            subgraph_graphs = [
                copy.deepcopy(g)
                for g in random.sample(depth_graphs[depth], num_subgraphs)
            ]

            info: Dict[str, Any] = {
                "num_subgraphs":     num_subgraphs,
                "subgraphs_depth":   [g["max_depth"] for g in subgraph_graphs],
                "subgraphs_edges":   [g["num_edges"] for g in subgraph_graphs],
                "target_depth":      depth,
            }

            # ---- 2. With p=0.5, flip the root of a subgraph so its label
            #         becomes False (skipped under no_negation).
            root_node_values = [True] * len(subgraph_graphs)
            if not no_negation:
                for j, graph in enumerate(subgraph_graphs):
                    if random.random() < 0.5:
                        for edge in graph["edges"]:
                            for conclusion in edge["conclusions"]:
                                if conclusion["node"] == "node0":
                                    conclusion["neg"] = True
                                    root_node_values[j] = False

            # ---- 3. Re-number node ids so the `branches` subgraphs don't
            #         overlap when concatenated.
            max_nodes = -1
            root_node_ids = [0]
            for graph in subgraph_graphs:
                max_current_nodes = max_nodes
                for edge in graph["edges"]:
                    for lit in edge["premises"] + edge["conclusions"]:
                        old = int(lit["node"].split("node")[1])
                        new = max_nodes + 1 + old
                        lit["node"] = f"node{new}"
                        max_current_nodes = max(max_current_nodes, new)
                max_nodes = max_current_nodes
                root_node_ids.append(max_nodes + 1)
            root_node_ids = root_node_ids[:-1]   # drop the trailing sentinel

            info["root_node_ids"]    = root_node_ids
            info["root_node_values"] = root_node_values

            # ---- 4. Concatenate.  The first subgraph is kept intact.  For
            #         every other subgraph we either drop one of its edges
            #         OR flip a literal in one edge — this guarantees that
            #         the *bundle* still has only one provable root.
            final_graph = copy.deepcopy(subgraph_graphs[0])
            for graph in subgraph_graphs[1:]:
                edge_idx = random.choice(range(1, len(graph["edges"])))
                if random.random() < 0.5 or no_negation:
                    # drop edge_idx
                    for j, edge in enumerate(graph["edges"]):
                        if j == edge_idx:
                            continue
                        final_graph["edges"].append(copy.deepcopy(edge))
                else:
                    # flip one literal inside edge_idx
                    for j, edge in enumerate(graph["edges"]):
                        if j == edge_idx:
                            if edge["type"] == "assign":
                                edge["conclusions"][0]["neg"] = (
                                    not edge["conclusions"][0]["neg"]
                                )
                            else:
                                k = random.choice(range(
                                    len(edge["premises"]) +
                                    len(edge["conclusions"])
                                ))
                                if k < len(edge["premises"]):
                                    edge["premises"][k]["neg"] = (
                                        not edge["premises"][k]["neg"]
                                    )
                                else:
                                    cc = k - len(edge["premises"])
                                    edge["conclusions"][cc]["neg"] = (
                                        not edge["conclusions"][cc]["neg"]
                                    )
                        final_graph["edges"].append(copy.deepcopy(edge))

            # ---- 5. Pad with distractor implication edges until reaching
            #         the (per-sample randomized) total-edges budget.
            next_node_id = max_nodes + 1
            expected_edges = random.randint(
                get_implication_edges_count(final_graph["edges"]),
                min(total_edges,
                    get_implication_edges_count(final_graph["edges"]) + 5),
            )
            info["total_edges"] = expected_edges

            while get_implication_edges_count(final_graph["edges"]) < expected_edges:
                current_node_id = next_node_id
                num_prem = random.randint(1, min(max_arity, current_node_id))
                num_conc = random.randint(1, min(max_arity, current_node_id))
                if no_conjunction:
                    num_prem = 1
                if no_disjunction:
                    num_conc = 1

                # which side (if any) reuses existing nodes
                side = random.choice(["premises", "conclusions", "none"])
                # person 0 acts as a universal ∀-marker
                if p_forall > 0:
                    person = 0 if random.random() < p_forall else random.randint(1, num_persons)
                else:
                    person = random.randint(1, num_persons)

                def _make_literals(num: int, reuse_side: bool):
                    out = []
                    nonlocal next_node_id
                    if reuse_side:
                        nodes = random.sample(range(current_node_id), num)
                        neg = (False if random.random() < 0.5 else True)
                        if no_negation: neg = False
                        for k in range(num):
                            out.append({"node": f"node{nodes[k]}",
                                        "person": person, "neg": neg, "depth": -1})
                    else:
                        for _ in range(num):
                            neg = (False if random.random() < 0.5 else True)
                            if no_negation: neg = False
                            out.append({"node": f"node{next_node_id}",
                                        "person": person, "neg": neg, "depth": -1})
                            next_node_id += 1
                    return out

                premises    = _make_literals(num_prem, side == "premises")
                conclusions = _make_literals(num_conc, side == "conclusions")
                final_graph["edges"].append({
                    "type": "implication",
                    "premises": premises,
                    "conclusions": conclusions,
                })

            # ---- 6. Split into train / test for this depth bin ----
            if i < n_train_this:
                train_graphs.append(final_graph)
                train_extra_info.append(info)
            else:
                test_graphs.append(final_graph)
                test_extra_info.append(info)

    # shuffle once more so depth doesn't appear in row order
    if train_graphs:
        combined = list(zip(train_graphs, train_extra_info))
        random.shuffle(combined)
        train_graphs, train_extra_info = map(list, zip(*combined))
    if test_graphs:
        combined = list(zip(test_graphs, test_extra_info))
        random.shuffle(combined)
        test_graphs, test_extra_info = map(list, zip(*combined))

    return train_graphs, train_extra_info, test_graphs, test_extra_info


# -----------------------------------------------------------------------------
# CLI entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a ScaleLogic train/test parquet pair."
    )
    parser.add_argument("--n_train",       type=int,   default=100000)
    parser.add_argument("--n_test",        type=int,   default=1000)
    parser.add_argument("--total_edges",   type=int,   default=16,
                        help="must equal branches * depth")
    parser.add_argument("--depth",         type=int,   default=8)
    parser.add_argument("--branches",      type=int,   default=2)
    parser.add_argument("--num_persons",   type=int,   default=2)
    parser.add_argument("--max_arity",     type=int,   default=2)
    parser.add_argument("--p_small_arity", type=float, default=1.0,
                        help="probability of sampling a small (≤2) arity edge")
    parser.add_argument("--p_forall",      type=float, default=0.5,
                        help="probability of inserting a ∀ quantifier")
    parser.add_argument("--no_reuse",       action="store_true",
                        help="disable node reuse across edges")
    parser.add_argument("--no_disjunction", action="store_true", help="disable ∨")
    parser.add_argument("--no_negation",    action="store_true", help="disable ¬")
    parser.add_argument("--no_conjunction", action="store_true", help="disable ∧")
    parser.add_argument("--seed",          type=int,   default=0)
    parser.add_argument("--suffix",        type=str,   default="",
                        help="optional tag appended to the output filename")
    args = parser.parse_args()

    assert args.branches >= 2, "branches should be ≥ 2"
    assert args.total_edges == args.branches * args.depth, \
        "total_edges should equal branches * depth"

    train_graphs, train_extra_info, test_graphs, test_extra_info = \
        generate_uniform_depth_samples(
            n_train=args.n_train,
            n_test=args.n_test,
            num_persons=args.num_persons,
            total_edges=args.total_edges,
            branches=args.branches,
            max_depth=args.depth,
            max_arity=args.max_arity,
            no_reuse=args.no_reuse,
            no_disjunction=args.no_disjunction,
            no_negation=args.no_negation,
            no_conjunction=args.no_conjunction,
            p_small_arity=args.p_small_arity,
            p_forall=args.p_forall,
            p_negate_node=0.5,
            p_sibling_neg=0.5,
            seed=args.seed,
        )

    print(f"Generated {len(train_graphs)} train + {len(test_graphs)} test bundles")
    print("-" * 32)

    # depth histogram (sanity)
    def _hist(infos):
        out: Dict[int, int] = {}
        for x in infos:
            out[x["target_depth"]] = out.get(x["target_depth"], 0) + 1
        return dict(sorted(out.items()))

    print(f"Train depth distribution: {_hist(train_extra_info)}")
    print(f"Test  depth distribution: {_hist(test_extra_info)}")
    print("-" * 32)

    train_inputs, train_outputs = graphs_to_qa(
        train_graphs, train_extra_info, seed=args.seed)
    test_inputs,  test_outputs  = graphs_to_qa(
        test_graphs,  test_extra_info,  seed=args.seed + 1)

    print("Sample extra_info  :", train_extra_info[0])
    print("Sample input       :", train_inputs[0][:240], "..." if len(train_inputs[0]) > 240 else "")
    print("Sample gold output :", train_outputs[0])
    print("-" * 32)

    import datasets

    def _to_dataset(inputs, outputs, extras):
        return datasets.Dataset.from_dict({
            "index":        list(range(len(inputs))),
            "data_source":  ["search_logic_multi_choice"] * len(inputs),
            "prompt":       [[{"content": x, "role": "user"}] for x in inputs],
            "reward_model": [{"style": "rule", "ground_truth": [x]} for x in outputs],
            "extra_info":   extras,
        })

    train_dataset = _to_dataset(train_inputs, train_outputs, train_extra_info)
    test_dataset  = _to_dataset(test_inputs,  test_outputs,  test_extra_info)

    # build the suffix that encodes every distribution-affecting flag
    suffix = f"{args.p_forall*100:.0f}%forall"
    suffix += "_no_reuse"       if args.no_reuse       else ""
    suffix += "_no_disjunction" if args.no_disjunction else ""
    suffix += "_no_negation"    if args.no_negation    else ""
    suffix += "_no_conjunction" if args.no_conjunction else ""
    suffix += f"_{args.suffix}" if args.suffix         else ""

    fname_base = (
        f"./search_logic_{args.total_edges}_{args.depth}_"
        f"{args.branches}_{args.num_persons}_{suffix}"
    )
    train_dataset.to_parquet(f"{fname_base}_train.parquet")
    test_dataset .to_parquet(f"{fname_base}_test.parquet")

    print("Wrote:")
    print(f"  {fname_base}_train.parquet")
    print(f"  {fname_base}_test.parquet")
