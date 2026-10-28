# Algorithmic baselines: selection policies paired with a symbolic reader.
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from . import FAMILIES, fam_abstention, fam_spatial
from .core import World

_TOK = re.compile(r"[a-z0-9]+")


def text_world(ex):
    fam = ex["task_family"]
    mod = FAMILIES[fam]
    facts = {n["id"]: mod.parse_fact(n["content_text"]) for n in ex["hidden_evidence_graph"]["nodes"]}
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


class Reader:
    def __init__(self, ex):
        self.ex = ex
        self.mod, self.w, self.vis, self.facts = text_world(ex)

    def determined(self, gathered) -> Optional[str]:
        fs = list(self.vis) + [self.facts[g] for g in gathered if self.facts.get(g) is not None]
        return self.mod.determine(self.w, fs)

    def answer(self, gathered, rng, abstain_if_unknown=False) -> str:
        a = self.determined(gathered)
        if a is not None:
            return a
        if abstain_if_unknown:
            return "abstain"
        opts = [c for c in self.ex["answer_space"] if c != "abstain"]
        return opts[int(rng.integers(len(opts)))]


def _pred(ex, gathered, answer):
    act = {e["target"]: e["action"] for e in ex["hidden_evidence_graph"]["edges"]}
    return {"answer": answer, "evidence_gathered": list(gathered),
            "actions": [f"{act[g]}:{g}" for g in gathered] + ["abstain" if answer == "abstain" else "answer_now"],
            "total_cost": len(gathered)}


def _budget(ex, tier="medium"):
    return ex["budget_tiers"][tier]


def _bm25_rank(ex) -> List[str]:
    nodes = ex["hidden_evidence_graph"]["nodes"]
    docs = [_TOK.findall(n["title"].lower()) for n in nodes]
    query = _TOK.findall((ex["question"] + " " + " ".join(ex["answer_space"])).lower())
    N = len(docs)
    avg = sum(len(d) for d in docs) / max(1, N)
    df = Counter(t for d in docs for t in set(d))
    k1, b = 1.2, 0.75
    scores = []
    for n, d in zip(nodes, docs):
        tf = Counter(d)
        s = 0.0
        for t in query:
            if t in tf:
                idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    order = sorted(range(N), key=lambda i: -scores[i])
    return [nodes[i]["id"] for i in order]


def _gather_until(reader, order, budget, stop_when_determined=True):
    got = []
    for nid in order:
        if len(got) >= budget:
            break
        if stop_when_determined and reader.determined(got) is not None:
            break
        got.append(nid)
    return got


def run_baseline(name: str, ex: Dict, rng, gains: Optional[Dict[str, float]] = None) -> Dict:
    reader = Reader(ex)
    nodes = ex["hidden_evidence_graph"]["nodes"]
    ids = [n["id"] for n in nodes]
    g = gains or {n["id"]: n["information_gain"] for n in nodes}
    B = _budget(ex)
    if name == "oracle":
        got = list(ex["minimal_sufficient_evidence"])
        return _pred(ex, got, ex["gold_answer"])
    if name == "direct":
        return _pred(ex, [], reader.answer([], rng))
    if name == "random_order":
        order = list(rng.permutation(ids))
        got = _gather_until(reader, order, B)
        return _pred(ex, got, reader.answer(got, rng))
    if name == "always_search":
        order = list(rng.permutation(ids))
        got = order[:B]
        return _pred(ex, got, reader.answer(got, rng))
    if name == "bm25_preview":
        got = _gather_until(reader, _bm25_rank(ex), B)
        return _pred(ex, got, reader.answer(got, rng))
    by_g = sorted(ids, key=lambda i: -g[i])
    if name.startswith("fixed_k"):
        k = int(name.split("_")[-1])
        got = by_g[:min(k, B)]
        return _pred(ex, got, reader.answer(got, rng))
    if name == "heuristic":
        got = [i for i in by_g if g[i] >= 0.5][:B]
        return _pred(ex, got, reader.answer(got, rng, abstain_if_unknown=True))
    if name == "greedy_g":
        got = _gather_until(reader, by_g, B)
        return _pred(ex, got, reader.answer(got, rng))
    raise ValueError(name)


NON_PRIVILEGED = ["direct", "random_order", "always_search", "bm25_preview"]
PRIVILEGED = ["fixed_k_1", "fixed_k_3", "fixed_k_5", "heuristic", "greedy_g"]
ALL = NON_PRIVILEGED + PRIVILEGED + ["oracle"]
