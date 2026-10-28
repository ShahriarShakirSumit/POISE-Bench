# Procedural family: prerequisite questions over a maintenance procedure with hidden dependency cards.
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from .core import Node, World, CertificationError
from .textfmt import TemplateSet

ACTION = "open_card"
VERBS = ["Calibrate", "Flush", "Inspect", "Replace", "Mount", "Seal", "Purge", "Test", "Torque",
         "Align", "Ground", "Prime", "Vent", "Lubricate", "Reset", "Patch", "Drain", "Charge"]
OBJECTS = ["sensor", "valve", "coolant line", "relay", "pump", "gasket", "filter", "bracket",
           "manifold", "breaker", "actuator", "bearing", "hose", "panel", "regulator", "fan"]
SYSTEMS = ["chiller unit", "backup generator", "rooftop air handler", "boiler room",
           "compressor bank", "water treatment skid", "conveyor line", "server room cooling loop"]

CARD = TemplateSet({
    "deps": ["Card for '{step}': it requires {deps} to be finished first.",
             "Before '{step}' can start, these must be done: {deps}.",
             "'{step}' depends on {deps}."],
    "none": ["Card for '{step}': it has no prerequisites.",
             "'{step}' can start right away; nothing must be done before it.",
             "'{step}' depends on nothing else."],
})
QUESTION = TemplateSet({
    "prereq": ["Which of these steps must be finished before '{target}' can start?"],
    "count": ["How many other steps must be finished before '{target}' can start?"],
    "notdep": ["Which of these steps does not depend, directly or indirectly, on '{target}'?"],
})
INSTRUCTIONS = ("A step can start only after all of its prerequisites, and their prerequisites, are "
                "finished. Exactly one answer choice is correct.")


def _fmt_deps(deps: List[str]) -> str:
    q = [f"'{d}'" for d in deps]
    return q[0] if len(q) == 1 else ", ".join(q[:-1]) + " and " + q[-1]


def _parse_deps(s: str) -> List[str]:
    return re.findall(r"'([^']+)'", s)


def _ancestors(x: str, parents: Dict[str, Set[str]]) -> Set[str]:
    out, stack = set(), [x]
    while stack:
        for p in parents.get(stack.pop(), ()):
            if p not in out:
                out.add(p)
                stack.append(p)
    return out


def _known_closure(x: str, cards: Dict[str, Set[str]]):
    anc, stack, complete = set(), [x], True
    seen = {x}
    while stack:
        s = stack.pop()
        if s not in cards:
            complete = False
            continue
        for p in cards[s]:
            anc.add(p)
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return anc, complete


def determine(world: World, facts: List[tuple]) -> Optional[str]:
    q = world.qstruct
    cards = {f[1]: set(f[2]) for f in facts if f[0] == "card"}
    x = q["x"]
    if q["kind"] == "count":
        anc, complete = _known_closure(x, cards)
        if complete:
            s = str(len(anc))
            return s if s in world.choices else None
        alive = [c for c in world.choices if int(c) >= len(anc)]
        return alive[0] if len(alive) == 1 else None
    st = {}
    if q["kind"] == "prereq":
        anc_x, complete_x = _known_closure(x, cards)
        for c in world.choices:
            if c in anc_x:
                st[c] = True
            elif complete_x or x in _known_closure(c, cards)[0]:
                st[c] = False
            else:
                st[c] = None
    else:
        anc_x, _ = _known_closure(x, cards)
        for c in world.choices:
            anc_c, complete_c = _known_closure(c, cards)
            if x in anc_c:
                st[c] = False
            elif complete_c or c in anc_x:
                st[c] = True
            else:
                st[c] = None
    trues = [c for c, s in st.items() if s is True]
    if len(trues) > 1:
        raise CertificationError("more than one choice satisfies the question")
    if len(trues) == 1:
        return trues[0]
    alive = [c for c, s in st.items() if s is not False]
    return alive[0] if len(alive) == 1 else None


def parse_fact(text: str):
    p = CARD.parse(text)
    if p is None:
        return None
    k, g = p
    return ("card", g["step"], tuple(sorted(_parse_deps(g["deps"]))) if k == "deps" else tuple())


def parse_question(text: str):
    p = QUESTION.parse(text)
    if p is None:
        return None
    k, g = p
    return {"kind": k, "x": g["target"]}


def _steps(rng, n):
    out, used = [], set()
    while len(out) < n:
        v = VERBS[int(rng.integers(len(VERBS)))]
        o = OBJECTS[int(rng.integers(len(OBJECTS)))]
        nm = f"{v} {o} {int(rng.integers(1, 10))}"
        key = (v, o)
        if key not in used:
            used.add(key)
            out.append(nm)
    return out


def generate(rng, split: str, variant: str, idx: int) -> World:
    style = {"v1": 0, "v2": 1, "v3": 2}[variant]
    kind = {"v1": "prereq", "v2": "count", "v3": "notdep"}[variant]
    r_rank = int(rng.integers(4))
    for _ in range(200):
        n = int(rng.integers(7, 11))
        steps = _steps(rng, n)
        parents = {s: set() for s in steps}
        for i in range(1, n):
            k = int(rng.choice([0, 1, 1, 2, 2]))
            if k:
                for j in rng.choice(i, size=min(k, i), replace=False):
                    parents[steps[i]].add(steps[int(j)])
        anc = {s: _ancestors(s, parents) for s in steps}
        if kind == "prereq":
            xs = [s for s in steps if len(anc[s]) >= 2]
            if not xs:
                continue
            x = xs[int(rng.integers(len(xs)))]
            non = [s for s in steps if s != x and s not in anc[x]]
            if len(non) < 3:
                continue
            gold = sorted(anc[x])[int(rng.integers(len(anc[x])))]
            pick = [non[i] for i in rng.choice(len(non), size=3, replace=False)]
            choices = [gold] + pick
        elif kind == "count":
            r = r_rank
            xs = [s for s in steps if len(anc[s]) >= max(2, r) and len(anc[s]) - r + 3 <= n - 1]
            if not xs:
                continue
            x = xs[int(rng.integers(len(xs)))]
            g = len(anc[x])
            lo = g - r
            gold = str(g)
            choices = [str(v) for v in range(lo, lo + 4)]
        else:
            xs = [s for s in steps if sum(s in anc[t] for t in steps) >= 3]
            if not xs:
                continue
            x = xs[int(rng.integers(len(xs)))]
            dep = [t for t in steps if x in anc[t]]
            free = [t for t in steps if t != x and x not in anc[t]]
            if not free or len(dep) < 3:
                continue
            gold = free[int(rng.integers(len(free)))]
            pick = [dep[i] for i in rng.choice(len(dep), size=3, replace=False)]
            choices = [gold] + pick
        rng.shuffle(choices)
        break
    else:
        raise CertificationError("no procedural configuration")

    order = rng.permutation(n)
    nodes = []
    for j, oi in enumerate(order):
        s = steps[oi]
        deps = sorted(parents[s])
        text = (CARD.render("deps", style, step=s, deps=_fmt_deps(deps)) if deps
                else CARD.render("none", style, step=s))
        nodes.append(Node(id=f"card_{j}", text=text, title=f"prerequisites of '{s}'",
                          fact=("card", s, tuple(deps)), action=ACTION))
    listing = list(steps)
    rng.shuffle(listing)
    system = SYSTEMS[int(rng.integers(len(SYSTEMS)))]
    initial = (f"Runbook for servicing the {system}. Its {n} steps, in no particular order: "
               + "; ".join(f"'{s}'" for s in listing)
               + f". Each step has a dependency card (card_0 to card_{n - 1}) listing its direct prerequisites.")
    question = QUESTION.render(kind, 0, target=x)
    return World(family="procedural", variant=variant, split=split, question=question, choices=choices,
                 gold=gold, initial_text=initial, visible_facts=[], nodes=nodes,
                 qstruct={"kind": kind, "x": x}, template_id=f"procedural_{variant}_{kind}",
                 actions=[ACTION], instructions=INSTRUCTIONS,
                 meta={"order": steps, "parents": {k: sorted(v) for k, v in parents.items()}})


def round_trip(world: World) -> bool:
    for nd in world.nodes:
        f = parse_fact(nd.text)
        if f is None or (f[0], f[1], tuple(sorted(f[2]))) != (nd.fact[0], nd.fact[1], tuple(sorted(nd.fact[2]))):
            raise CertificationError(f"card does not parse back: {nd.text!r}")
    if parse_question(world.question) != world.qstruct:
        raise CertificationError(f"question does not parse back: {world.question!r}")
    return True
