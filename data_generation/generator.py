"""Logic-graph sampler used by :pyfile:`aug_generator.py`.

This file exposes three callables:

* :func:`generate_logic_graph` — sample a single proof-tree-shaped logic
  graph with controlled max-depth and max-edges.  This is the core
  generator; the depth-uniform batching below just rejects / accepts its
  outputs.
* :func:`generate_logic_graph_batch_with_uniform_depth` — wrap the above
  to return `n` graphs whose `max_depth` is uniform over `[1, max_depth]`.
* :func:`get_implication_edges_count` — small utility.

Every other variant we tried during development (random-edges batch,
fixed-depth batch, per-(depth, edges) bucket batch, calibration sweep)
has been removed; only the path used by the released training scripts
remains.
"""

import copy
import random
from typing import List, Dict, Any, Set, Tuple


def get_implication_edges_count(edges: List[Dict[str, Any]]) -> int:
    """Count the `type == "implication"` edges in a graph."""
    return sum(1 for e in edges if e["type"] == "implication")


# ---------------------------------------------------------------------------
# Core single-graph generator
# ---------------------------------------------------------------------------

def generate_logic_graph(
    num_persons: int,
    max_edges: int,
    max_depth: int,
    max_arity: int = 4,
    no_reuse: bool = False,
    no_disjunction: bool = False,
    no_negation: bool = False,
    no_conjunction: bool = False,
    p_assign: float = 0.0,
    p_small_arity: float = 0.9,
    p_forall: float = 0.5,
    p_negate_node: float = 0.5,
    p_sibling_neg: float = 0.5,
    deep_first: bool = False,
    root_neg: bool = False,
    seed: int = 0,
) -> Dict[str, Any]:
    """Sample one logic graph rooted at ``node0``.

    The graph is grown breadth- or depth-first by repeatedly expanding the
    frontier of "needs-justification" nodes (initially just ``node0``).  At
    each step we either:

    1. **Stop** — emit an `assign` edge that turns the node into a leaf
       fact (forced once we hit ``max_depth`` or run out of edge budget).
    2. **Expand with an implication** — sample one new edge of the form
       ``p_1 ∧ … ∧ p_k → c_1 ∨ … ∨ c_m`` whose conclusion contains the
       current node.  Premises become the new frontier.
    3. **Expand under a ∀-quantifier** — same as (2) but with
       ``person == 0`` (which the verbaliser later realises as "for any X
       …").  Existing ∀-edges may also be re-used with different person
       instantiations.

    A final pass randomly negates a subset of node ids globally so the
    resulting truth values are not trivially monotone.

    Returns a dict ``{"edges", "key_conclusions", "num_edges", "max_depth"}``.
    """
    if seed is not None:
        random.seed(seed)

    edges: List[Dict[str, Any]] = []
    max_depth_used = 0
    next_node_id = 1

    # Bookkeeping for node reuse.
    reusable_nodes: List[int] = []                    # node ids that can host another person
    reusable_nodes_count: Dict[str, int] = {}         # person counter per node id
    forall_nodes: List[str] = []                      # nodes already attached under ∀
    depth_reusable_nodes: Dict[int, List[Tuple[str, int, bool]]] = {}

    # Frontier as (node_id, depth, person, value).
    queue: List[Tuple[str, int, int, bool]] = [("node0", 0, 1, True)]
    key_conclusions: List[Dict[str, Any]] = []        # one per expansion, used by graph_to_qa

    def choose_person_for_condition() -> int:
        """Return 0 (∀ over all persons) with probability p_forall, else 1."""
        return 0 if random.random() < p_forall else 1

    def add_assign_edge(node_id: str, node_person: int, depth: int, value: bool) -> None:
        """Emit a single-literal `assign` edge realising `node_id := value`."""
        edges.append({
            "type": "assign",
            "premises": [],
            "conclusions": [{
                "node": node_id,
                "person": node_person,
                "neg": (not value),
                "depth": depth,
            }],
        })

    # ∀-edge pool: implication edges originally introduced under ∀ that can
    # later be re-instantiated for any of the remaining person ids.
    forall_edges: List[Dict[str, Any]] = []
    forall_edges_person_set: List[Set[int]] = []

    while queue:
        if deep_first and max_depth_used < max_depth - 1:
            # depth-first variant: always pop the deepest pending node first
            max_d = max(d for _, d, _, _ in queue)
            idx_candidates = [i for i, (_, d, _, _) in enumerate(queue) if d == max_d]
            idx = random.choice(idx_candidates)
        else:
            idx = random.randrange(len(queue))
        node_id, depth, node_person, value = queue.pop(idx)

        key_conclusions.append({
            "node":   [node_id],
            "person": [node_person],
            "neg":    [not value],
        })

        depth_reusable_nodes.setdefault(depth, []).append((node_id, node_person, value))

        if depth > max_depth_used:
            max_depth_used = depth

        # all earlier edges that touch the current node
        previous_edges = [
            e for e in edges
            if node_id in [p["node"] for p in e["premises"]]
            or node_id in [c["node"] for c in e["conclusions"]]
        ]

        # Remaining capacity if we add ONE implication here.
        max_new_nodes = max_edges - get_implication_edges_count(edges)

        # Force a leaf assignment when we can't go deeper or wider.
        if depth >= max_depth or max_new_nodes <= 0 or random.random() < p_assign:
            add_assign_edge(node_id, node_person, depth, value)
            continue

        # -----------------------------------------------------------------
        # Decide whether the new edge is a ∀-rule or a per-person rule.
        # -----------------------------------------------------------------
        condition_person = choose_person_for_condition()
        if node_id == "node0" and len(edges) == 0:
            condition_person = 1            # never put ∀ at the very root
        if node_id in forall_nodes:
            condition_person = 1            # avoid stacking ∀ on the same node twice
        if depth >= max_depth - 1:
            condition_person = 1            # would exceed depth budget

        # ------- 1. Re-use an existing ∀-edge with a fresh person -------
        if condition_person == 0:
            if random.random() >= 1.0 / (len(forall_edges) + 1):
                index = random.randint(0, len(forall_edges) - 1)
                edge = copy.deepcopy(forall_edges[index])
                person_set = forall_edges_person_set[index]
                cur_person = random.choice(list(person_set))
                person_set.remove(cur_person)
                forall_edges_person_set[index] = person_set
                if not person_set:
                    forall_edges.pop(index)
                    forall_edges_person_set.pop(index)

                # premises become next frontier (depth + 2 accounts for ∀ + link edge)
                for premise in edge["premises"]:
                    queue.append((premise["node"], depth + 2, cur_person, True))

                # link one of the ∀-conclusions to the current node
                selected_index = random.choice(range(len(edge["conclusions"])))
                link_edge = {
                    "type": "implication",
                    "premises": [{
                        "node":   edge["conclusions"][selected_index]["node"],
                        "person": cur_person,
                        "neg":    False,
                        "depth":  depth + 1,
                    }],
                    "conclusions": [{
                        "node":   node_id,
                        "person": node_person,
                        "neg":    (not value),
                        "depth":  depth,
                    }],
                }
                edges.append(link_edge)
                key_conclusions.append({
                    "node":   [edge["conclusions"][selected_index]["node"]],
                    "person": [cur_person],
                    "neg":    [False],
                })

                # remaining conclusions either come back into the queue or
                # get an inline copy of the link edge (cheap re-use).
                for index in range(len(edge["conclusions"])):
                    if index == selected_index:
                        continue
                    conclusion = edge["conclusions"][index]
                    if (random.random() < p_sibling_neg
                            or get_implication_edges_count(edges) >= max_edges):
                        queue.append((conclusion["node"], depth, cur_person, False))
                    else:
                        copy_link_edge = copy.deepcopy(link_edge)
                        copy_link_edge["premises"][0]["node"] = conclusion["node"]
                        edges.append(copy_link_edge)
                        key_conclusions[-1]["node"  ].append(conclusion["node"])
                        key_conclusions[-1]["person"].append(cur_person)
                        key_conclusions[-1]["neg"   ].append(False)
                continue
            else:
                forall_nodes.append(node_id)

        # ------- 2. Otherwise pick a (num_prem, num_conc) pair -------
        candidates: List[Tuple[int, int]]       = []
        small_candidates: List[Tuple[int, int]] = []
        for num_prem in range(1, max_arity + 1):
            for num_conc in range(1, max_arity + 1):
                if no_conjunction and num_prem >= 2: continue
                if no_disjunction and num_conc >= 2: continue
                k = num_prem + (num_conc - 1)
                if k > max_new_nodes:                continue
                # disallow OR at the very root
                if node_id == "node0" and len(edges) == 0 and num_conc >= 2:
                    continue
                candidates.append((num_prem, num_conc))
                if num_prem <= 2 and num_conc <= 2:
                    small_candidates.append((num_prem, num_conc))

        if not candidates:
            add_assign_edge(node_id, node_person, depth, value)
            continue

        if random.random() < p_small_arity:
            num_prem, num_conc = random.choice(small_candidates)
        else:
            num_prem, num_conc = random.choice(candidates)

        # ------- 3. Build premises -------
        premises: List[Dict[str, Any]] = []
        higher_depth_reusable_nodes: List[Tuple[str, int, bool]] = []
        for i in range(depth + 1, max_depth):
            if i in depth_reusable_nodes:
                higher_depth_reusable_nodes.extend(depth_reusable_nodes[i])
            else:
                break

        for _ in range(num_prem):
            # case A: reuse a node from a deeper level (same person)
            if (condition_person != 0
                    and random.random() >= 1 / (len(higher_depth_reusable_nodes) + 1)
                    and not no_reuse):
                idx = random.randint(0, len(higher_depth_reusable_nodes) - 1)
                nid, cur_person, cur_value = higher_depth_reusable_nodes.pop(idx)
                premises.append({
                    "node": nid, "person": cur_person,
                    "neg": (not cur_value), "depth": depth + 1,
                })
                continue

            # case B: reuse a node, different person
            if (condition_person != 0
                    and random.random() >= 1 / (len(reusable_nodes) + 1)
                    and not no_reuse):
                index = random.randint(0, len(reusable_nodes) - 1)
                nid = f"node{reusable_nodes[index]}"
                reusable_nodes_count[nid] += 1
                cur_person = reusable_nodes_count[nid]
                if cur_person >= num_persons:
                    reusable_nodes.pop(index)
            # case C: per-person fresh node
            elif condition_person != 0:
                nid = f"node{next_node_id}"
                cur_person = 1
                if cur_person < num_persons:
                    reusable_nodes.append(next_node_id)
                reusable_nodes_count[nid] = 1
                next_node_id += 1
            # case D: ∀-fresh node
            else:
                nid = f"node{next_node_id}"
                cur_person = 0
                next_node_id += 1
                forall_nodes.append(nid)

            premises.append({
                "node": nid, "person": cur_person, "neg": False, "depth": depth + 1,
            })
            # queue the new premise for further expansion
            queue.append((nid, depth + 1,
                          cur_person if cur_person != 0 else node_person, True))

        # ------- 4. Build conclusions -------
        conclusions: List[Dict[str, Any]] = []

        if num_conc == 1:
            # single-conclusion implication: just realise node_id with its target value
            conclusions.append({
                "node":   node_id,
                "person": node_person if condition_person != 0 else 0,
                "neg":    (not value),
                "depth":  depth,
            })
        else:
            # disjunctive implication: first conclusion is `node_id`, the
            # rest are siblings.  Each sibling can be either (a) a fresh
            # node that's forced False so node_id keeps its target value,
            # or (b) a copy of an earlier edge structurally rewritten to
            # share the truth value.
            conclusions.append({
                "node":   node_id,
                "person": node_person if condition_person != 0 else 0,
                "neg":    (not value),
                "depth":  depth,
            })
            for _ in range(num_conc - 1):
                # avoid a corner case when negation is disabled but we'd need it
                if ((len(previous_edges) != 1
                        or get_implication_edges_count(edges) >= max_edges - 1)
                        and no_negation):
                    continue

                # pick a node id for the sibling
                if (condition_person != 0
                        and random.random() >= 1 / (len(reusable_nodes) + 1)
                        and not no_reuse):
                    index = random.randint(0, len(reusable_nodes) - 1)
                    nid = f"node{reusable_nodes[index]}"
                    reusable_nodes_count[nid] += 1
                    cur_person = reusable_nodes_count[nid]
                    if cur_person >= num_persons:
                        reusable_nodes.pop(index)
                elif condition_person != 0:
                    nid = f"node{next_node_id}"
                    cur_person = 1
                    if cur_person < num_persons:
                        reusable_nodes.append(next_node_id)
                    reusable_nodes_count[nid] = 1
                    next_node_id += 1
                else:
                    nid = f"node{next_node_id}"
                    cur_person = 0
                    next_node_id += 1
                    forall_nodes.append(nid)

                # (a) sibling = False, push to queue
                if ((random.random() < p_sibling_neg
                        or len(previous_edges) != 1
                        or get_implication_edges_count(edges) >= max_edges - 1)
                        and not no_negation):
                    conclusions.append({
                        "node": nid, "person": cur_person, "neg": False, "depth": depth,
                    })
                    queue.append((nid, depth,
                                  cur_person if cur_person != 0 else node_person,
                                  False))
                # (b) sibling shares target value, copy the previous edge structure
                else:
                    assert (len(previous_edges) == 1
                            and get_implication_edges_count(edges) < max_edges)
                    conclusions.append({
                        "node": nid, "person": cur_person,
                        "neg": (not value), "depth": depth,
                    })
                    key_conclusions[-1]["node"  ].append(nid)
                    key_conclusions[-1]["person"].append(cur_person)
                    key_conclusions[-1]["neg"   ].append((not value))
                    edge = copy.deepcopy(previous_edges[0])
                    for p in edge["premises"]:
                        if (p["node"] == node_id
                                and (p["person"] == node_person or p["person"] == 0)):
                            p["node"] = nid
                            p["person"] = cur_person if cur_person != 0 else node_person
                    for c in edge["conclusions"]:
                        if (c["node"] == node_id
                                and (c["person"] == node_person or c["person"] == 0)):
                            c["node"] = nid
                            c["person"] = cur_person if cur_person != 0 else node_person
                    edges.append(edge)

        edge = {"type": "implication", "premises": premises, "conclusions": conclusions}
        edges.append(edge)

        # If this is a freshly introduced ∀-edge, register it as a candidate
        # for later re-instantiation with another person id.
        if condition_person == 0:
            if int(node_id.split("node")[1]) in reusable_nodes:
                index = reusable_nodes.index(int(node_id.split("node")[1]))
                count = reusable_nodes_count[node_id]
                reusable_nodes.pop(index)
            else:
                count = reusable_nodes_count[node_id]
            person_set = set(range(1, num_persons + 1))
            person_set.discard(node_person)
            for _ in range(1, count + 1):
                person_set.discard(_)
            if person_set:
                forall_edges.append(edge)
                forall_edges_person_set.append(person_set)

    # -------------------------------------------------------------------
    # Global random negation: flip `neg` on every literal whose node id is
    # selected, so truth values are not trivially monotone.
    # -------------------------------------------------------------------
    if not no_negation:
        all_nodes = [f"node{i}" for i in range(1, next_node_id)]
        flipped_nodes = {nid for nid in all_nodes if random.random() < p_negate_node}
        if root_neg:
            flipped_nodes.add("node0")

        for conclusion in key_conclusions:
            conclusion["neg"] = [
                (not neg) if node in flipped_nodes else neg
                for node, neg in zip(conclusion["node"], conclusion["neg"])
            ]
        for edge in edges:
            for key in ("premises", "conclusions"):
                for lit in edge[key]:
                    if lit["node"] in flipped_nodes:
                        lit["neg"] = not lit["neg"]

    return {
        "edges":           edges,
        "key_conclusions": key_conclusions,
        "num_edges":       get_implication_edges_count(edges),
        "max_depth":       max_depth_used,
    }


# ---------------------------------------------------------------------------
# Batch wrapper with uniform max-depth
# ---------------------------------------------------------------------------

# Per-depth edge cap used by the batch wrapper as the upper end of the
# uniform draw on `num_edges`.  Each list is the 90th-percentile edge count
# observed in a 500-sample calibration sweep at that depth, for one combo of
# operator toggles; padded with `999` beyond depth 8.

_EDGE_LIMIT_FULL                = [0, 1, 5, 13, 21, 28, 36, 44, 51] + [999] * 100  # default: ∀ and node reuse enabled
_EDGE_LIMIT_NRNF_DEFAULT        = [0, 1, 5, 13, 32, 72, 156]        + [999] * 100  # no_reuse + p_forall≈0, no other ops disabled
_EDGE_LIMIT_NRNF_NO_DISJUNCTION = [0, 1, 3,  6, 10, 18, 28, 44, 67] + [999] * 100  # ...also no ∨
_EDGE_LIMIT_NRNF_NO_NEGATION    = [0, 1, 4,  8, 14, 24, 38, 61, 95] + [999] * 100  # ...also no ¬
_EDGE_LIMIT_IMPL_ONLY           = list(range(1000))                                # implication chains only — no cap


def _select_edge_limit(no_reuse: bool, p_forall: float,
                       no_conjunction: bool, no_disjunction: bool,
                       no_negation: bool) -> List[int]:
    """Pick the calibrated per-depth edge cap for the active operator regime."""
    if no_reuse and p_forall < 1e-6:
        if no_conjunction and no_disjunction:
            return _EDGE_LIMIT_IMPL_ONLY
        if no_disjunction:
            return _EDGE_LIMIT_NRNF_NO_DISJUNCTION
        if no_negation:
            return _EDGE_LIMIT_NRNF_NO_NEGATION
        return _EDGE_LIMIT_NRNF_DEFAULT
    return _EDGE_LIMIT_FULL


def generate_logic_graph_batch_with_uniform_depth(
    n: int,
    num_persons: int,
    max_edges: int,
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
    root_neg: bool = False,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Sample `n` graphs with `max_depth` uniformly distributed over
    ``[1, max_depth]`` and `num_edges` ≤ `max_edges`.

    For each requested depth `d` we draw an edge target uniformly from
    ``[d, min(max_edges, edge_limit[d])]`` and call
    :func:`generate_logic_graph` until the realised graph matches the
    requested `(depth, edges)` pair exactly.  ``edge_limit`` is picked by
    :func:`_select_edge_limit` from the calibration tables defined above
    this function.
    """
    if seed is not None:
        random.seed(seed)

    edge_limit = _select_edge_limit(
        no_reuse, p_forall, no_conjunction, no_disjunction, no_negation,
    )

    hist: Dict[int, int] = {d: 0 for d in range(1, max_depth + 1)}
    target_num = n // max_depth
    remainder  = n %  max_depth
    print(f"depths={max_depth}  target_per_depth={target_num}")
    print("-" * 32)

    graphs: List[Dict[str, Any]] = []
    for depth in hist.keys():
        n_this = target_num + (1 if remainder > 0 else 0)
        if remainder > 0:
            remainder -= 1
        for _ in range(n_this):
            requested_edges = random.randint(depth, min(max_edges, edge_limit[depth]))
            while True:
                g = generate_logic_graph(
                    num_persons=num_persons,
                    max_edges=requested_edges,
                    max_depth=depth,
                    max_arity=max_arity,
                    no_reuse=no_reuse,
                    no_disjunction=no_disjunction,
                    no_negation=no_negation,
                    no_conjunction=no_conjunction,
                    p_small_arity=p_small_arity,
                    p_forall=p_forall,
                    p_negate_node=p_negate_node,
                    p_sibling_neg=p_sibling_neg,
                    deep_first=True,
                    root_neg=root_neg,
                    seed=None,           # use the global RNG for diversity
                )
                if g["num_edges"] == requested_edges and g["max_depth"] == depth:
                    hist[depth] += 1
                    graphs.append(g)
                    break

    print(f"generated {len(graphs)} graphs")
    print("-" * 32)
    return graphs[:n]
