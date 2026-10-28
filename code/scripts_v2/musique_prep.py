# Builds the 600-item MuSiQue sample from the official MuSiQue release.
import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/musique/data/musique_full_v1.0_dev.jsonl"
OUT = ROOT / "data/musique/transfer_600.json"
N_ANS = 300

rows = [json.loads(l) for l in open(SRC)]
pairs = collections.defaultdict(dict)
for r in rows:
    pairs[r["id"]]["ans" if r["answerable"] else "unans"] = r
ids = sorted(i for i, p in pairs.items() if "ans" in p and "unans" in p)
hop = {i: i.split("__")[0] for i in ids}
counts = collections.Counter(hop.values())
rng = np.random.default_rng(2026)
chosen = []
for h in sorted(counts):
    pool = [i for i in ids if hop[i] == h]
    k = round(N_ANS * counts[h] / len(ids))
    chosen += [pool[j] for j in sorted(rng.choice(len(pool), size=k, replace=False))]
chosen = chosen[:N_ANS]
items = []
for i in chosen:
    for kind in ("ans", "unans"):
        r = pairs[i][kind]
        items.append({
            "example_id": f"{i}::{kind}",
            "pair_id": i,
            "hop_type": hop[i],
            "hops": len(r["question_decomposition"]),
            "answerable": r["answerable"],
            "question": r["question"],
            "gold_answer": r["answer"] if r["answerable"] else "abstain",
            "answer_aliases": r.get("answer_aliases", []) if r["answerable"] else [],
            "paragraphs": [{"pid": f"p{p['idx']}", "title": p["title"], "text": p["paragraph_text"],
                            "is_supporting": bool(p["is_supporting"])} for p in r["paragraphs"]],
        })
OUT.write_text(json.dumps(items, indent=1))
print(len(items), "items;", collections.Counter(x["hop_type"] for x in items if x["answerable"]))
print("supporting per answerable item:", collections.Counter(sum(p["is_supporting"] for p in x["paragraphs"]) for x in items if x["answerable"]))
