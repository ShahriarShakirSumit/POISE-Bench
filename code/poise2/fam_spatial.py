# Spatial family: compass and distance questions about landmarks hidden on a town map.
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .core import Node, World, CertificationError
from .textfmt import TemplateSet

ACTION = "reveal_region"
LANDMARKS = [
    "fountain", "bakery", "clock tower", "bus stop", "library", "bike rack", "cafe", "statue",
    "kiosk", "pharmacy", "post office", "bench", "flower stall", "bookshop", "museum",
    "playground", "taxi stand", "newsstand", "tram stop", "ticket booth", "gallery",
    "fish market", "chapel", "bandstand", "water tower", "rose garden", "duck pond",
    "footbridge", "hotel", "tea house", "barber", "laundry", "cinema", "gym", "school", "bank",
    "fire station", "pottery shop", "toy store", "bicycle shop", "noodle bar", "market hall",
    "observatory", "windmill", "greenhouse", "chess tables", "fruit stand", "town hall",
]
DIRS = {"north": (1, +1), "south": (1, -1), "east": (0, +1), "west": (0, -1)}

NODE_TEXT = TemplateSet({
    "pos": ["Region {i} uncovered: the {name} is at ({x}, {y}).",
            "Under region {i} is the {name}, located at ({x}, {y}).",
            "Uncovering region {i} shows the {name} at position ({x}, {y})."],
})
QUESTION = TemplateSet({
    "one": ["Which of these landmarks is {d1} of the {a1}?"],
    "and": ["Which of these landmarks is {d1} of the {a1} and {d2} of the {a2}?"],
    "not": ["Which of these landmarks is {d1} of the {a1} but not {d2} of the {a2}?"],
    "closest": ["Which of these landmarks is closest to the {a1}?"],
})
INSTRUCTIONS = ("Coordinates are (x, y): x grows to the east and y grows to the north. 'North of' means "
                "a larger y, 'south of' a smaller y, 'east of' a larger x, 'west of' a smaller x. "
                "Distance is the number of blocks walked, |x1 - x2| + |y1 - y2|. "
                "Exactly one answer choice is correct.")
_VISIBLE = re.compile(r"the ([a-z ]+?) at \((\d+), (\d+)\)")


def _holds(d, p, q) -> bool:
    axis, sign = DIRS[d]
    return (p[axis] - q[axis]) * sign > 0


def _truth(q, c, pos) -> Optional[bool]:
    pc = pos.get(c)
    if q["kind"] == "one":
        pa = pos.get(q["a1"])
        return None if pc is None or pa is None else _holds(q["d1"], pc, pa)
    t1 = None if pc is None or pos.get(q["a1"]) is None else _holds(q["d1"], pc, pos[q["a1"]])
    t2 = None if pc is None or pos.get(q["a2"]) is None else _holds(q["d2"], pc, pos[q["a2"]])
    if q["kind"] == "and":
        if t1 is False or t2 is False:
            return False
        return True if (t1 and t2) else None
    if t1 is False or t2 is True:
        return False
    return True if (t1 is True and t2 is False) else None


def determine(world: World, facts: List[tuple]) -> Optional[str]:
    q = world.qstruct
    pos = {f[1]: (f[2], f[3]) for f in facts if f[0] == "pos"}
    if q["kind"] == "closest":
        pa = pos.get(q["a1"])
        if pa is None or any(c not in pos for c in world.choices):
            return None
        dist = {c: abs(pos[c][0] - pa[0]) + abs(pos[c][1] - pa[1]) for c in world.choices}
        best = min(dist.values())
        win = [c for c, v in dist.items() if v == best]
        return win[0] if len(win) == 1 else None
    st = {c: _truth(q, c, pos) for c in world.choices}
    trues = [c for c, s in st.items() if s is True]
    if len(trues) > 1:
        raise CertificationError("more than one choice satisfies the question")
    if len(trues) == 1:
        return trues[0]
    alive = [c for c, s in st.items() if s is not False]
    return alive[0] if len(alive) == 1 else None


def parse_fact(text: str):
    p = NODE_TEXT.parse(text)
    if p is None:
        return None
    _k, g = p
    return ("pos", g["name"], int(g["x"]), int(g["y"]))


def parse_question(text: str):
    p = QUESTION.parse(text)
    if p is None:
        return None
    k, g = p
    out = {"kind": k, "a1": g["a1"]}
    if "d1" in g:
        out["d1"] = g["d1"]
    if "a2" in g:
        out["a2"], out["d2"] = g["a2"], g["d2"]
    return out


def generate(rng, split: str, variant: str, idx: int) -> World:
    n = int(rng.integers(7, 10))
    names = [LANDMARKS[i] for i in rng.choice(len(LANDMARKS), size=n, replace=False)]
    xs = rng.choice(10, size=n, replace=False)
    ys = rng.choice(10, size=n, replace=False)
    pos = {nm: (int(x), int(y)) for nm, x, y in zip(names, xs, ys)}
    style = {"v1": 0, "v2": 1, "v3": 2}[variant]

    if variant == "v1":
        kind = "one"
    elif variant == "v2":
        kind = "and"
    else:
        kind = "not" if rng.random() < 0.6 else "closest"

    dirs = list(DIRS)
    for _ in range(200):
        a1, a2 = [names[i] for i in rng.choice(n, size=2, replace=False)]
        q = {"kind": kind, "a1": a1}
        if kind != "closest":
            q["d1"] = dirs[int(rng.integers(4))]
        if kind in ("and", "not"):
            q["a2"] = a2
            q["d2"] = dirs[int(rng.integers(4))]
            if q["d2"] == q["d1"] and kind == "not":
                continue
        anchors = {a1} | ({a2} if "a2" in q else set())
        cands = [nm for nm in names if nm not in anchors]
        if kind == "closest":
            d = {c: abs(pos[c][0] - pos[a1][0]) + abs(pos[c][1] - pos[a1][1]) for c in cands}
            order = sorted(cands, key=lambda c: d[c])
            if len(order) < 4 or d[order[0]] == d[order[1]]:
                continue
            gold = order[0]
            others = order[1:]
        else:
            sat = [c for c in cands if _truth(q, c, pos) is True]
            others = [c for c in cands if _truth(q, c, pos) is False]
            if len(sat) < 1 or len(others) < 3:
                continue
            gold = sat[int(rng.integers(len(sat)))]
        pick = [others[i] for i in rng.choice(len(others), size=3, replace=False)]
        choices = [gold] + pick
        rng.shuffle(choices)
        if kind != "closest" and sum(_truth(q, c, pos) is True for c in choices) != 1:
            continue
        break
    else:
        raise CertificationError("no spatial configuration")

    must_mask = set(anchors) | {gold}
    vis_pool = [nm for nm in names if nm not in must_mask]
    n_vis = int(rng.integers(1, 3))
    visible = [vis_pool[i] for i in rng.choice(len(vis_pool), size=min(n_vis, len(vis_pool)), replace=False)]
    masked = [nm for nm in names if nm not in visible]
    order = rng.permutation(len(masked))
    nodes = []
    for j, oi in enumerate(order):
        nm = masked[oi]
        x, y = pos[nm]
        nodes.append(Node(id=f"region_{j}", text=NODE_TEXT.render("pos", style, i=str(j), name=nm, x=str(x), y=str(y)),
                          title=f"covers the {nm}", fact=("pos", nm, x, y), action=ACTION))
    vis_txt = "; ".join(f"the {nm} at ({pos[nm][0]}, {pos[nm][1]})" for nm in visible)
    initial = (f"A map of a town square. Visible landmarks: {vis_txt}. "
               f"{len(nodes)} more landmarks are hidden under masked regions region_0 to region_{len(nodes) - 1}.")
    fields = {"a1": q["a1"]}
    if "d1" in q:
        fields["d1"] = q["d1"]
    if "a2" in q:
        fields["a2"], fields["d2"] = q["a2"], q["d2"]
    question = QUESTION.render(kind, 0, **fields)
    return World(family="spatial", variant=variant, split=split, question=question, choices=choices,
                 gold=gold, initial_text=initial,
                 visible_facts=[("pos", nm, pos[nm][0], pos[nm][1]) for nm in visible],
                 nodes=nodes, qstruct=q, template_id=f"spatial_{variant}_{kind}", actions=[ACTION],
                 instructions=INSTRUCTIONS, meta={"all_positions": {k: list(v) for k, v in pos.items()}})


def round_trip(world: World) -> bool:
    for nd in world.nodes:
        if parse_fact(nd.text) != nd.fact:
            raise CertificationError(f"node text does not parse back: {nd.text!r}")
        m = re.search(r"region (\d+)", nd.text, re.IGNORECASE)
        if m is None or f"region_{m.group(1)}" != nd.id:
            raise CertificationError(f"region index mismatch: {nd.text!r}")
    vis = {("pos", nm, int(x), int(y)) for nm, x, y in _VISIBLE.findall(world.initial_text)}
    if vis != set(world.visible_facts):
        raise CertificationError("initial observation does not parse back")
    if parse_question(world.question) != world.qstruct:
        raise CertificationError(f"question does not parse back: {world.question!r}")
    return True
