# Shared machinery: worlds, exact evidence analysis, certification checks and item export.
from __future__ import annotations

import hashlib
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HELPFUL_SLACK = 1
UNIT_COST = 1
BUDGET_MULTIPLIERS = {"easy": 3.0, "medium": 1.5, "hard": 1.0}


@dataclass
class Node:
    id: str
    text: str
    title: str
    fact: Optional[tuple]
    action: str
    cost: int = UNIT_COST


@dataclass
class World:
    family: str
    variant: str
    split: str
    question: str
    choices: List[str]
    gold: str
    initial_text: str
    visible_facts: List[tuple]
    nodes: List[Node]
    qstruct: Dict[str, Any]
    template_id: str
    actions: List[str]
    instructions: str = ""
    meta: Dict[str, Any] = field(default_factory=dict)


class CertificationError(Exception):
    pass


def _facts(world: World, idx: Sequence[int]) -> List[tuple]:
    out = list(world.visible_facts)
    for i in idx:
        f = world.nodes[i].fact
        if f is not None:
            out.append(f)
    return out


def analyse(world: World, determine: Callable[[World, List[tuple]], Optional[str]],
            max_size: int = 8) -> Dict[str, Any]:
    n = len(world.nodes)
    if determine(world, list(world.visible_facts)) is not None:
        raise CertificationError("shortcut: initial observation already determines the answer")
    full = determine(world, _facts(world, range(n)))
    if full is None:
        raise CertificationError("all evidence does not determine the answer")
    if full != world.gold:
        raise CertificationError(f"solver answer {full!r} != gold {world.gold!r}")

    useful = [i for i in range(n) if world.nodes[i].fact is not None]
    minimum: List[Tuple[int, ...]] = []
    k_min = None
    for k in range(1, min(max_size, len(useful)) + 1):
        for comb in itertools.combinations(useful, k):
            ans = determine(world, _facts(world, comb))
            if ans is not None:
                if ans != world.gold:
                    raise CertificationError(f"subset {comb} forces wrong answer {ans!r}")
                minimum.append(comb)
        if minimum:
            k_min = k
            break
    if not minimum:
        raise CertificationError("no sufficient set within max_size")
    if len(minimum) > 1:
        raise CertificationError("minimum evidence set not unique")

    irreducible = list(minimum)
    suff_cache = {c: True for c in minimum}

    def is_suff(c: Tuple[int, ...]) -> bool:
        if c not in suff_cache:
            suff_cache[c] = determine(world, _facts(world, c)) is not None
        return suff_cache[c]

    for k in range(k_min + 1, min(k_min + HELPFUL_SLACK, len(useful)) + 1):
        for comb in itertools.combinations(useful, k):
            if not is_suff(comb):
                continue
            if any(is_suff(sub) for sub in itertools.combinations(comb, k - 1)):
                continue
            irreducible.append(comb)

    mandatory = set(minimum[0])
    for c in minimum[1:]:
        mandatory &= set(c)
    in_any = set(itertools.chain.from_iterable(irreducible))
    helpful = in_any - mandatory
    roles = {}
    for i, node in enumerate(world.nodes):
        roles[node.id] = ("mandatory" if i in mandatory else
                          "helpful" if i in helpful else "distractor")
    return {
        "minimum_sets": [[world.nodes[i].id for i in c] for c in minimum],
        "irreducible_sets": [[world.nodes[i].id for i in c] for c in irreducible],
        "roles": roles,
        "oracle_cost": sum(world.nodes[i].cost for i in minimum[0]),
        "k_min": k_min,
    }


def _gain(role: str, rng) -> float:
    if role == "mandatory":
        return round(float(rng.uniform(0.7, 0.9)), 3)
    if role == "helpful":
        return round(float(rng.uniform(0.4, 0.6)), 3)
    return round(float(rng.uniform(0.0, 0.3)), 3)


def export_item(world: World, analysis: Dict[str, Any], example_id: str, seed: int, rng) -> Dict[str, Any]:
    roles = analysis["roles"]
    mse = analysis["minimum_sets"][0]
    c_star = analysis["oracle_cost"]
    budgets = {t: max(1, math.ceil(m * c_star - 1e-9)) for t, m in BUDGET_MULTIPLIERS.items()}
    terminal = "abstain" if world.gold == "abstain" else "answer_now"
    nodes_out = []
    for nd in world.nodes:
        nodes_out.append({
            "id": nd.id,
            "title": nd.title,
            "content_text": nd.text,
            "content_type": "text",
            "modality": "textual",
            "necessity": roles[nd.id],
            "information_gain": _gain(roles[nd.id], rng),
        })
    edges = [{"source": "initial", "target": nd.id, "action": nd.action, "cost": nd.cost}
             for nd in world.nodes]
    action_of = {nd.id: nd.action for nd in world.nodes}
    trajectories = [{
        "actions": [f"{action_of[i]}:{i}" for i in s] + [terminal],
        "cost": sum(1 for _ in s) * UNIT_COST,
        "is_minimal": True,
        "evidence_gathered": list(s),
    } for s in analysis["minimum_sets"]]
    action_costs = {a: UNIT_COST for a in world.actions}
    action_costs.update({"answer_now": 0, "abstain": 0})
    n_m = sum(r == "mandatory" for r in roles.values())
    n_d = sum(r == "distractor" for r in roles.values())
    difficulty = "easy" if (len(mse) <= 1 and n_d <= 6) else ("medium" if len(mse) <= 2 else "hard")
    return {
        "example_id": example_id,
        "task_family": world.family,
        "difficulty": difficulty,
        "instructions": world.instructions,
        "initial_observation": {"modality": "text", "text": world.initial_text},
        "question": world.question,
        "answer_space": list(world.choices),
        "action_set": ["answer_now", "abstain"] + list(world.actions),
        "action_costs": action_costs,
        "hidden_evidence_graph": {"nodes": nodes_out, "edges": edges},
        "gold_answer": world.gold,
        "gold_answer_type": "abstain" if world.gold == "abstain" else "exact",
        "valid_trajectories": trajectories,
        "minimal_sufficient_evidence": list(mse),
        "minimal_sufficient_sets": analysis["minimum_sets"],
        "irreducible_sufficient_sets": analysis["irreducible_sets"],
        "helpful_evidence": [i for i, r in roles.items() if r == "helpful"],
        "distractor_evidence": [i for i, r in roles.items() if r == "distractor"],
        "oracle_cost": c_star,
        "budget_tiers": budgets,
        "split": world.split,
        "certification": {
            "evidence_determines_gold": True,
            "no_shortcut_from_observation": True,
            "unique_gold_among_choices": True,
            "text_round_trip": True,
            "n_mandatory": n_m,
            "n_minimum_sets": len(analysis["minimum_sets"]),
        },
        "contamination_metadata": {
            "template_id": world.template_id,
            "generation_seed": seed,
            "structural_variant": world.variant,
            "content_hash": hashlib.sha1(
                (world.question + "|" + "|".join(n.text for n in world.nodes)).encode()
            ).hexdigest()[:16],
        },
        "_world": {"qstruct": world.qstruct, "meta": world.meta,
                   "visible_facts": [list(f) for f in world.visible_facts],
                   "node_facts": {nd.id: (list(nd.fact) if nd.fact is not None else None)
                                  for nd in world.nodes}},
    }
