# Checks for overlap between splits and reports statistics per split.
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/v2"
splits = {s: json.load(open(DATA / f"{s}.json")) for s in ("train", "dev", "test")}


def entities(ex):
    fam = ex["task_family"]
    out = set()
    if fam == "retrieval":
        for f in ex["_world"]["node_facts"].values():
            out.add(f[1])
            if isinstance(f[2], str):
                out.add(f[2])
    elif fam == "abstention":
        for f in ex["_world"]["visible_facts"]:
            out.add(f[1])
    return out


sets = {}
for s, items in splits.items():
    sets[s] = {
        "ids": {e["example_id"] for e in items},
        "hash": {e["contamination_metadata"]["content_hash"] for e in items},
        "question": {e["question"] for e in items if e["task_family"] != "abstention"},
        "template": {e["contamination_metadata"]["template_id"] for e in items},
        "entities": set().union(*[entities(e) for e in items]),
    }
overlap = {}
for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
    overlap[f"{a}-{b}"] = {k: len(sets[a][k] & sets[b][k]) for k in sets[a]}

stats = {}
for s, items in splits.items():
    fam = defaultdict(list)
    for e in items:
        fam[e["task_family"]].append(e)
    fam["all"] = items
    stats[s] = {}
    for f, es in fam.items():
        n_nodes = [len(e["hidden_evidence_graph"]["nodes"]) for e in es]
        m = [len(e["minimal_sufficient_evidence"]) for e in es]
        hel = [len(e["helpful_evidence"]) / len(e["hidden_evidence_graph"]["nodes"]) for e in es]
        dis = [len(e["distractor_evidence"]) / len(e["hidden_evidence_graph"]["nodes"]) for e in es]
        ql = [len(re.findall(r"\S+", e["question"])) for e in es]
        stats[s][f] = {
            "n": len(es),
            "nodes_mean": round(float(np.mean(n_nodes)), 2), "nodes_range": [int(min(n_nodes)), int(max(n_nodes))],
            "mandatory_mean": round(float(np.mean(m)), 2), "mandatory_range": [int(min(m)), int(max(m))],
            "helpful_frac": round(float(np.mean(hel)), 3), "distractor_frac": round(float(np.mean(dis)), 3),
            "question_words_mean": round(float(np.mean(ql)), 1),
            "abstain_rate": round(float(np.mean([e["gold_answer"] == "abstain" for e in es])), 3),
            "templates": sorted({e["contamination_metadata"]["template_id"] for e in es}),
        }
out = {"overlap": overlap, "stats": stats}
(ROOT / "audit/v2_split_stats.json").write_text(json.dumps(out, indent=1))
print(json.dumps(overlap, indent=1))
for s in stats:
    for f, v in stats[s].items():
        print(s, f, {k: v[k] for k in v if k != "templates"})
