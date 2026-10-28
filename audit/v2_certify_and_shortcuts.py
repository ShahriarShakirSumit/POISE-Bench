# Re-checks every released item from its text alone and looks for answer-choice shortcuts.
import itertools
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from poise2 import FAMILIES
from poise2.core import World
from poise2 import fam_spatial, fam_abstention

DATA = ROOT / "data/v2"


def world_from_text(ex):
    fam = ex["task_family"]
    mod = FAMILIES[fam]
    nodes = ex["hidden_evidence_graph"]["nodes"]
    facts = {n["id"]: mod.parse_fact(n["content_text"]) for n in nodes}
    if fam == "spatial":
        vis = [("pos", nm, int(x), int(y)) for nm, x, y in fam_spatial._VISIBLE.findall(ex["initial_observation"]["text"])]
    elif fam == "abstention":
        vis = fam_abstention.parse_table(ex["initial_observation"]["text"])
    else:
        vis = []
    q = mod.parse_question(ex["question"]) if fam != "abstention" else {"kind": "identify"}
    w = World(family=fam, variant="", split="", question=ex["question"], choices=ex["answer_space"],
              gold=ex["gold_answer"], initial_text="", visible_facts=vis, nodes=[], qstruct=q,
              template_id="", actions=[])
    return mod, w, vis, facts


def det(mod, w, vis, facts, ids):
    return mod.determine(w, list(vis) + [facts[i] for i in ids if facts[i] is not None])


def unique_minimum(mod, w, vis, facts, ids, mse):
    target = set(mse)
    for k in range(len(mse) + 1):
        for s in itertools.combinations(ids, k):
            if set(s) != target and det(mod, w, vis, facts, list(s)) is not None:
                return False
    return True


out = {}
for split in ("train", "dev", "test"):
    items = json.load(open(DATA / f"{split}.json"))
    c = defaultdict(Counter)
    pos_c = defaultdict(Counter)
    rank_c = defaultdict(Counter)
    twin = defaultdict(Counter)
    for ex in items:
        fam = ex["task_family"]
        mod, w, vis, facts = world_from_text(ex)
        allids = list(facts)
        mse = ex["minimal_sufficient_evidence"]
        cc = c[fam]
        cc["n"] += 1
        cc["T1_all_evidence_determines_gold"] += det(mod, w, vis, facts, allids) == ex["gold_answer"]
        cc["T2_no_shortcut_from_observation"] += det(mod, w, vis, facts, []) is None
        cc["T3_mandatory_set_determines_gold"] += det(mod, w, vis, facts, mse) == ex["gold_answer"]
        cc["T4_every_mandatory_node_necessary"] += all(
            det(mod, w, vis, facts, [m for m in mse if m != d]) is None for d in mse)
        cc["T5_mandatory_set_unique_minimum"] += unique_minimum(mod, w, vis, facts, allids, mse)
        cc["parse_failures"] += sum(f is None and n["title"] != "logistics"
                                   for f, n in zip(facts.values(), ex["hidden_evidence_graph"]["nodes"]))
        pos_c[fam][ex["answer_space"].index(ex["gold_answer"])] += 1
        nums = [a for a in ex["answer_space"] if a.replace(",", "").isdigit()]
        if len(nums) == 4:
            r = sorted(nums, key=lambda a: int(a.replace(",", ""))).index(ex["gold_answer"])
            rank_c[fam + ":" + ex["contamination_metadata"]["template_id"].split("_", 2)[-1]][r] += 1
        if fam == "abstention":
            table = defaultdict(dict)
            for f in vis:
                table[f[1]][f[2]] = f[3]
            cls = [a for a in ex["answer_space"] if a != "abstain"]
            near = any(sum(table[a][k] != table[b][k] for k in table[a]) <= 2
                       for i, a in enumerate(cls) for b in cls[i + 1:])
            kind = "abstain" if ex["gold_answer"] == "abstain" else "solvable"
            twin[kind]["n"] += 1
            twin[kind]["near_twin_rows_among_choices"] += near
    out[split] = {
        "certificate": {f: dict(v) for f, v in c.items()},
        "gold_position_counts": {f: dict(sorted(v.items())) for f, v in pos_c.items()},
        "gold_rank_among_numeric_choices": {f: dict(sorted(v.items())) for f, v in rank_c.items()},
        "abstention_near_twin": {k: dict(v) for k, v in twin.items()},
    }
(ROOT / "audit/v2_certification.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
