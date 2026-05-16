"""Verbalise a logic graph (the output of :pyfile:`generator.py`) into a
natural-language multiple-choice question.

The graph is shipped as facts ("Alice is xxxxx", "If A then B", "For any X,
…").  We then list `branches` candidate root statements — exactly one of
which is provable from the facts — and ask the model to copy that statement
into an ``<answer> … </answer>`` block.

The fixed random property names (5-letter strings) and the fixed person
pool are seeded per call so the output is reproducible from the (graphs,
extra_info, seed) triple alone.
"""

import random
import string
from typing import List, Dict, Any, Tuple


FINAL_ANSWER_PROMPT = """
Exactly one of the above statements can be logically derived from the given facts.
Identify that statement and return it within <answer> </answer> tags. Example: <answer> Alice is abcde </answer>.
"""


# Fixed pool of human-readable person names; one name per person id within
# a single graph.  The pool is shuffled per graph so identifiers don't leak
# information across rows.
_BASE_NAMES: Tuple[str, ...] = (
    "Alice", "Bob", "Carol", "David", "Emma",
    "Frank", "Grace", "Henry", "Irene", "Jack",
    "Kate", "Leo", "Mona", "Nick", "Olivia", "Paul",
    "Quincy", "Rachel", "Sam", "Tina", "Ulysses",
    "Victoria", "Wendy", "Xavier", "Yvonne", "Zach",
)


def graphs_to_qa(
    graphs: List[Dict[str, Any]],
    extra_info: List[Dict[str, Any]],
    seed: int = 0,
) -> Tuple[List[str], List[str]]:
    """Convert a list of bundled logic graphs into prompts and gold answers.

    Each input `graphs[i]` is the augmented graph produced by
    :func:`aug_generator.generate_uniform_depth_samples`; `extra_info[i]`
    must contain ``root_node_ids`` and ``root_node_values`` for that bundle
    (the candidate-statement roots and their gold truth values).

    Returns
    -------
    inputs : list[str]
        Natural-language prompts ready to be tokenised.
    outputs : list[str]
        The gold answer string (a verbalised literal), one per input.
    """
    if seed is not None:
        random.seed(seed)

    inputs:  List[str] = []
    outputs: List[str] = []

    for g, info in zip(graphs, extra_info):
        edges = g["edges"]
        if not edges:
            continue

        # ---- person ids → human names (random per graph) ----
        names = list(_BASE_NAMES)
        random.shuffle(names)

        def person_to_name(pid: int) -> str:
            if pid == 0:
                return "anyone"
            return names[(pid - 1) % len(names)]

        # ---- node ids → 5-letter property strings ----
        node_ids: set = set()
        for e in edges:
            for lit in e.get("premises", []):
                node_ids.add(lit["node"])
            for lit in e.get("conclusions", []):
                node_ids.add(lit["node"])

        def random_prop() -> str:
            return "".join(random.choice(string.ascii_lowercase) for _ in range(5))

        # sort node ids so the property assignment is deterministic given the seed
        node2prop: Dict[str, str] = {nid: random_prop() for nid in sorted(node_ids)}

        def literal_to_text(lit: Dict[str, Any]) -> str:
            name = person_to_name(lit["person"])
            prop = node2prop[lit["node"]]
            return (f"{name} is not {prop}" if lit["neg"] else f"{name} is {prop}").strip()

        # ---- build fact sentences in shuffled edge order ----
        shuffled_edges = edges[:]
        random.shuffle(shuffled_edges)

        sentences: List[str] = []
        for e in shuffled_edges:
            et          = e["type"]
            premises    = e.get("premises", [])
            conclusions = e.get("conclusions", [])
            if not conclusions:
                continue

            if et == "assign":
                # a fact: "Alice is xxxxx."
                sentences.append(literal_to_text(conclusions[0]).capitalize() + ".")
                continue

            if et != "implication":
                continue  # unknown edge types are silently dropped

            if premises:
                prem_str = " and ".join(literal_to_text(p) for p in premises)
                conc_str = " or ".join(literal_to_text(c) for c in conclusions)
                sent = f"If {prem_str}, then {conc_str}."
                # if any "anyone" appears, re-realise as ∀-quantified text
                if "anyone" in sent:
                    prem_str = prem_str.replace("anyone", "X")
                    conc_str = conc_str.replace("anyone", "X")
                    if random.random() < 0.5:
                        sent = f"For any X, if {prem_str}, then {conc_str}."
                    else:
                        sent = f"For any person X, if {prem_str}, then {conc_str}."
                sentences.append(sent)
            else:
                # implication without premises: treat each conclusion as a bare fact
                for c in conclusions:
                    sentences.append(literal_to_text(c) + ".")

        # ---- emit candidate statements + the prompt ----
        facts_str = " ".join(sentences)
        root_node_ids    = info.get("root_node_ids",    [])
        root_node_values = info.get("root_node_values", [])

        # randomise candidate order so the correct one isn't always first
        perm = random.sample(range(len(root_node_ids)), len(root_node_ids))
        statement_lines = ["", "Here are some candidate statements:"]
        answer = ""
        for i in perm:
            rid  = root_node_ids[i]
            rval = root_node_values[i]
            lit  = {"person": 1, "node": f"node{rid}", "neg": not rval}
            statement_lines.append(literal_to_text(lit))
            if i == 0:
                answer = literal_to_text(lit)

        full_input = (
            "Suppose we have the following facts: "
            + facts_str
            + "\n"
            + "\n".join(statement_lines)
            + "\n"
            + FINAL_ANSWER_PROMPT
        )

        inputs.append(full_input)
        outputs.append(answer)

    return inputs, outputs
