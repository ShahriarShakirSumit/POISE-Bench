# Builds the 48 items shown to the human annotators and the answer key kept from them.
import json
import random
import string
from pathlib import Path

N = Path(__file__).resolve().parents[1]
SEED = 20261009
PER_FAMILY = 12
FAMS = ["retrieval", "spatial", "procedural", "abstention"]

test = json.load(open(N / "data/v2/test.json"))
human = json.load(open(N / "data/naturalistic/naturalistic78.json"))
rng = random.Random(SEED)
sample = []
for f in FAMS:
    pool = [e for e in test if e["task_family"] == f and e["example_id"] != "retrieval_test_00000"]
    sample += rng.sample(pool, PER_FAMILY)
rng.shuffle(sample)
human = list(human)
rng.shuffle(human)

public, key = [], {}
for part, items in (("A", sample), ("B", human)):
    for e in items:
        nodes = list(e["hidden_evidence_graph"]["nodes"])
        rng.shuffle(nodes)
        letters = string.ascii_uppercase[:len(nodes)]
        k = f"{part}-{e['example_id']}"
        public.append({
            "key": k, "part": part, "family": e["task_family"],
            "rules": (e.get("instructions") or "").strip(),
            "start": (e.get("initial_observation") or {}).get("text", ""),
            "question": e["question"], "choices": e["answer_space"],
            "records": [{"label": L, "title": (n.get("title") or "").strip(), "text": n["content_text"]}
                        for L, n in zip(letters, nodes)],
        })
        key[k] = {"gold": e["gold_answer"], "family": e["task_family"],
                  "mandatory": sorted(L for L, n in zip(letters, nodes) if n["id"] in e["minimal_sufficient_evidence"]),
                  "roles": {L: n.get("necessity") for L, n in zip(letters, nodes)},
                  "ids": {L: n["id"] for L, n in zip(letters, nodes)}}
(N / "annotation/items_public.json").write_text(json.dumps(public, ensure_ascii=False))
(N / "annotation/answer_key.json").write_text(json.dumps(key, indent=1))
print(len(public), "items:", sum(p["part"] == "A" for p in public), "test +", sum(p["part"] == "B" for p in public), "naturalistic")
print("records per item:", min(len(p["records"]) for p in public), "to", max(len(p["records"]) for p in public))
print("public json size (KB):", round(len(json.dumps(public, ensure_ascii=False)) / 1024, 1))
