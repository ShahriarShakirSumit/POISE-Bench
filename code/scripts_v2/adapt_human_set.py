# Converts the 78 naturalistic tasks to the format the agent runner reads.
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/human_validation/test.json"
OUT = ROOT / "data/naturalistic/naturalistic78.json"
RULES = {
    "retrieval": "Exactly one answer choice is correct.",
    "spatial": "Exactly one answer choice is correct.",
    "procedural": "Exactly one answer choice is correct.",
    "abstention": "Exactly one answer choice is correct. If the evidence shows that the question cannot be "
                  "answered with the available information, answer 'abstain'.",
}

rng = np.random.default_rng(78)
items = json.load(open(SRC))
out = []
for e in items:
    nodes = e["hidden_evidence_graph"]["nodes"]
    perm = rng.permutation(len(nodes))
    rename = {nodes[j]["id"]: f"doc_{k}" for k, j in enumerate(perm)}
    mse = [rename[m] for m in e["minimal_sufficient_evidence"]]
    new_nodes = []
    for j in perm:
        n = nodes[j]
        new_nodes.append({"id": rename[n["id"]], "title": "", "content_text": n["content_text"],
                          "content_type": "text", "modality": "textual",
                          "necessity": "mandatory" if rename[n["id"]] in mse else
                          ("distractor" if n["necessity"] == "distractor" else "helpful"),
                          "information_gain": n.get("information_gain", 0.5)})
    choices = ["abstain" if a.strip().lower() == "abstain" else a for a in e["answer_space"]]
    gold = "abstain" if e["gold_answer"].strip().lower() == "abstain" else e["gold_answer"]
    c_star = len(mse)
    out.append({
        "example_id": e["example_id"], "task_family": e["task_family"], "instructions": RULES[e["task_family"]],
        "initial_observation": {"modality": "text", "text": e["initial_observation"]["text"]},
        "question": e["question"], "answer_space": choices, "gold_answer": gold,
        "gold_answer_type": "abstain" if gold == "abstain" else "exact",
        "hidden_evidence_graph": {"nodes": new_nodes,
                                  "edges": [{"source": "initial", "target": n["id"], "action": "read", "cost": 1}
                                            for n in new_nodes]},
        "minimal_sufficient_evidence": mse, "oracle_cost": c_star,
        "budget_tiers": {"easy": math.ceil(3 * c_star), "medium": math.ceil(1.5 * c_star), "hard": c_star},
        "split": "human", "source_id": e["example_id"],
    })
OUT.write_text(json.dumps(out, indent=1))
print(len(out), "items ->", OUT)
