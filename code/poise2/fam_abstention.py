# Abstention family: name an item from a reference table, or abstain when the notes cannot decide.
from __future__ import annotations

import re
from typing import Dict, List, Optional

from .core import Node, World, CertificationError
from .names import NamePool
from .textfmt import TemplateSet

ACTION = "read_note"
WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
DOMAINS = {
    "mineral": {"obj": "specimen", "kind": "mineral", "suffix": "ite",
                "attrs": {"luster": ["glassy", "metallic", "waxy", "dull"],
                          "hardness": ["soft", "medium", "hard"],
                          "color": ["green", "red", "black", "white", "blue"],
                          "streak": ["white", "gray", "brown", "colorless"],
                          "density": ["low", "medium", "high"]}},
    "fault": {"obj": "pump", "kind": "fault", "suffix": " fault",
              "attrs": {"noise": ["grinding", "humming", "clicking", "silent"],
                        "temperature": ["normal", "warm", "hot"],
                        "vibration": ["none", "mild", "strong"],
                        "pressure": ["low", "normal", "high"],
                        "odor": ["none", "burnt", "oily"]}},
    "plant": {"obj": "plant", "kind": "species", "suffix": "ia",
              "attrs": {"leaf shape": ["round", "oval", "needle", "lobed"],
                        "flower color": ["white", "yellow", "purple", "none"],
                        "height": ["short", "medium", "tall"],
                        "habitat": ["wetland", "forest", "meadow", "rocky"],
                        "bark": ["smooth", "ridged", "peeling"]}},
    "incident": {"obj": "outage", "kind": "incident type", "suffix": " pattern",
                 "attrs": {"latency": ["normal", "elevated", "severe"],
                           "packet loss": ["none", "low", "high"],
                           "scope": ["single host", "subnet", "region"],
                           "error rate": ["low", "medium", "high"],
                           "timing": ["constant", "periodic", "bursty"]}},
}
NOTE = TemplateSet({
    "obs": ["Lab note: the {obj}'s {attr} is {value}.",
            "Observation: the {attr} of the {obj} is {value}.",
            "Inspection found the {attr} of the {obj} to be {value}."],
    "na": ["Lab note: the {obj}'s {attr} could not be determined.",
           "Observation: the {attr} of the {obj} could not be measured.",
           "Inspection could not establish the {attr} of the {obj}."],
    "noise": ["Lab note: the {obj} was logged by {person} on a {day}.",
              "Observation: the {obj} file was opened by {person} on a {day}.",
              "Inspection was scheduled by {person} for a {day}."],
})
INSTRUCTIONS = ("Use the reference table and the notes about the item. Answer with the single table "
                "entry that matches every observed value. If the entries that still match differ only "
                "in attributes the notes say could not be determined, answer 'abstain'. "
                "Exactly one answer choice is correct.")
_ROW = re.compile(r"^- (?P<cls>[^:]+): (?P<vals>.+)$")


def determine(world: World, facts: List[tuple]) -> Optional[str]:
    table: Dict[str, Dict[str, str]] = {}
    obs, unavailable = {}, set()
    for f in facts:
        if f[0] == "row":
            table.setdefault(f[1], {})[f[2]] = f[3]
        elif f[0] == "obs":
            obs[f[1]] = f[2]
        elif f[0] == "na":
            unavailable.add(f[1])
    cons = [c for c, row in table.items() if all(row.get(a) == v for a, v in obs.items())]
    if len(cons) == 1:
        return cons[0] if cons[0] in world.choices else None
    if len(cons) >= 2:
        attrs = {a for c in cons for a in table[c]}
        diff = {a for a in attrs if len({table[c][a] for c in cons}) > 1}
        if diff and diff <= unavailable:
            return "abstain"
    return None


def parse_fact(text: str):
    p = NOTE.parse(text)
    if p is None:
        return None
    k, g = p
    if k == "obs":
        return ("obs", g["attr"], g["value"])
    if k == "na":
        return ("na", g["attr"])
    return None


def parse_table(text: str):
    rows = []
    for line in text.splitlines():
        m = _ROW.match(line)
        if m:
            for part in m.group("vals").split("; "):
                a, v = part.split(" = ")
                rows.append(("row", m.group("cls"), a, v))
    return rows


def generate(rng, split: str, variant: str, idx: int) -> World:
    style = {"v1": 0, "v2": 1, "v3": 2}[variant]
    n_cls, n_att = {"v1": (3, 3), "v2": (4, 4), "v3": (5, 5)}[variant]
    n_noise = {"v1": 1, "v2": 1, "v3": 2}[variant]
    dname = list(DOMAINS)[int(rng.integers(len(DOMAINS)))]
    dom = DOMAINS[dname]
    names = NamePool(split, rng)
    abstain = rng.random() < 0.5
    for _ in range(300):
        attrs = [list(dom["attrs"])[i] for i in sorted(rng.choice(len(dom["attrs"]), size=n_att, replace=False))]
        used = set()
        classes = [names.word(used) + dom["suffix"] for _ in range(n_cls)]
        table = {c: {a: dom["attrs"][a][int(rng.integers(len(dom["attrs"][a])))] for a in attrs} for c in classes}
        t = classes[0]
        unavailable = set()
        if abstain:
            twin = classes[1]
            k = int(rng.integers(1, 3)) if n_att >= 4 else 1
            D = set(rng.choice(attrs, size=k, replace=False).tolist())
            table[twin] = dict(table[t])
            for a in sorted(D):
                alts = [v for v in dom["attrs"][a] if v != table[t][a]]
                table[twin][a] = alts[int(rng.integers(len(alts)))]
            unavailable = set(D)
        else:
            if n_att >= 4 and rng.random() < 0.5:
                unavailable = {attrs[int(rng.integers(n_att))]}
            if True:
                avail0 = [a for a in attrs if a not in unavailable]
                k = int(rng.integers(1, 3)) if len(avail0) >= 3 else 1
                D = set(rng.choice(avail0, size=k, replace=False).tolist())
                table[classes[1]] = dict(table[t])
                for a in sorted(D):
                    alts = [v for v in dom["attrs"][a] if v != table[t][a]]
                    table[classes[1]][a] = alts[int(rng.integers(len(alts)))]
        avail = [a for a in attrs if a not in unavailable]
        rows = [tuple(table[c][a] for a in attrs) for c in classes]
        if len(set(rows)) != n_cls:
            continue
        others = [c for c in classes if c != t and not (abstain and c == classes[1])]
        if not all(any(table[c][a] != table[t][a] for a in avail) for c in others):
            continue
        break
    else:
        raise CertificationError("no abstention configuration")

    order = list(classes)
    rng.shuffle(order)
    gold = "abstain" if abstain else t
    if abstain:
        rest = [c for c in classes if c not in (t, classes[1])]
        choices = [t, classes[1], rest[int(rng.integers(len(rest)))], "abstain"]
    else:
        rest = [c for c in classes if c not in (t, classes[1])]
        choices = [t, classes[1], rest[int(rng.integers(len(rest)))], "abstain"]
    rng.shuffle(choices)

    notes = []
    for a in attrs:
        if a in unavailable:
            notes.append(("na", a))
        else:
            notes.append(("obs", a, table[t][a]))
    person_pool = NamePool(split, rng)
    for _ in range(n_noise):
        notes.append(("noise", person_pool.person(set()), WEEKDAYS[int(rng.integers(5))]))
    perm = rng.permutation(len(notes))
    nodes = []
    for j, oi in enumerate(perm):
        f = notes[oi]
        if f[0] == "obs":
            text = NOTE.render("obs", style, obj=dom["obj"], attr=f[1], value=f[2])
            title, fact = f[1], f
        elif f[0] == "na":
            text = NOTE.render("na", style, obj=dom["obj"], attr=f[1])
            title, fact = f[1], f
        else:
            text = NOTE.render("noise", style, obj=dom["obj"], person=f[1], day=f[2])
            title, fact = "logistics", None
        nodes.append(Node(id=f"note_{j}", text=text, title=title, fact=fact, action=ACTION))

    lines = [f"Reference table of {dom['kind']}s:"]
    vis = []
    for c in order:
        lines.append(f"- {c}: " + "; ".join(f"{a} = {table[c][a]}" for a in attrs))
        vis += [("row", c, a, table[c][a]) for a in attrs]
    lines.append(f"Hidden notes note_0 to note_{len(nodes) - 1} describe the {dom['obj']} under examination.")
    question = (f"Which {dom['kind']} is the {dom['obj']}? If the notes cannot single out one entry of the "
                f"table, answer 'abstain'.")
    return World(family="abstention", variant=variant, split=split, question=question, choices=choices,
                 gold=gold, initial_text="\n".join(lines), visible_facts=vis, nodes=nodes,
                 qstruct={"kind": "identify", "domain": dname},
                 template_id=f"abstention_{variant}_{dname}_{'abstain' if abstain else 'solvable'}",
                 actions=[ACTION], instructions=INSTRUCTIONS,
                 meta={"true_class": t, "twin": classes[1] if abstain else None,
                       "unavailable": sorted(unavailable)})


def round_trip(world: World) -> bool:
    for nd in world.nodes:
        if parse_fact(nd.text) != nd.fact:
            raise CertificationError(f"note does not parse back: {nd.text!r}")
        if nd.fact is None and NOTE.parse(nd.text) is None:
            raise CertificationError(f"noise note does not parse: {nd.text!r}")
    if set(parse_table(world.initial_text)) != set(world.visible_facts):
        raise CertificationError("reference table does not parse back")
    return True
