# Retrieval family: multi-hop questions over invented organisations, people and cities.
from __future__ import annotations

from typing import Dict, List, Optional

from .core import Node, World, CertificationError
from .names import NamePool
from .textfmt import TemplateSet

ACTION = "read_record"

FACT_TEXT = TemplateSet({
    "founded": ["{org} was founded in {year}.",
                "Records show that {org} opened its doors in {year}.",
                "{org} has been operating since its founding in {year}."],
    "hq": ["{org} is headquartered in {city}.",
           "The head office of {org} is located in {city}.",
           "{org} runs its main office out of {city}."],
    "employer": ["{person} works at {org}.",
                 "{person} is currently employed by {org}.",
                 "{person} holds a full-time position at {org}."],
    "born_in": ["{person} was born in {city}.",
                "{person} grew up in {city}, where they were born.",
                "The birthplace of {person} is {city}."],
    "population": ["{city} has a population of about {pop}.",
                   "Roughly {pop} people live in {city}.",
                   "The most recent census counted about {pop} residents in {city}."],
})
TITLE = {"founded": "{org}: history", "hq": "{org}: offices", "employer": "{person}: career",
         "born_in": "{person}: early life", "population": "{city}: demographics"}

QUESTION = TemplateSet({
    "c:founded": ["In what year was {org} founded?"],
    "c:hq": ["In which city is {org} headquartered?"],
    "c:employer": ["Which organization does {person} work for?"],
    "c:born_in": ["In which city was {person} born?"],
    "c:population": ["What is the population of {city}?"],
    "c:employer,hq": ["In which city is the employer of {person} headquartered?"],
    "c:employer,founded": ["In what year was the employer of {person} founded?"],
    "c:hq,population": ["What is the population of the city where {org} is headquartered?"],
    "c:born_in,population": ["What is the population of the birthplace of {person}?"],
    "c:employer,hq,population": ["How many people live in the city that hosts the head office of the organization employing {person}?"],
    "t:employer,hq": ["{person} works for an organization; in which city does that organization keep its main office?"],
    "t:employer,founded": ["{person}'s employer opened in which year?"],
    "n:negconj": ["Which of these organizations was founded before {year} but is not headquartered in {city}?"],
})
START_TYPE = {"founded": "org", "hq": "org", "employer": "person", "born_in": "person", "population": "city"}
VALUE_TYPE = {"founded": "year", "hq": "city", "employer": "org", "born_in": "city", "population": "pop"}

V1_PATHS = [["founded"], ["hq"], ["employer"], ["born_in"], ["population"]]
V2_PATHS = [["employer", "hq"], ["employer", "founded"], ["hq", "population"], ["born_in", "population"]]
INSTRUCTIONS = ("The records describe invented organizations, people and places, so outside knowledge "
                "will not help. Exactly one answer choice is correct.")


def _fmt_pop(n: int) -> str:
    return f"{n:,}"


def _render_fact(f, style: int) -> str:
    kind = f[0]
    if kind == "founded":
        return FACT_TEXT.render(kind, style, org=f[1], year=str(f[2]))
    if kind == "hq":
        return FACT_TEXT.render(kind, style, org=f[1], city=f[2])
    if kind == "employer":
        return FACT_TEXT.render(kind, style, person=f[1], org=f[2])
    if kind == "born_in":
        return FACT_TEXT.render(kind, style, person=f[1], city=f[2])
    if kind == "population":
        return FACT_TEXT.render(kind, style, city=f[1], pop=_fmt_pop(f[2]))
    raise ValueError(kind)


def parse_fact(text: str):
    p = FACT_TEXT.parse(text)
    if p is None:
        return None
    kind, g = p
    if kind == "founded":
        return ("founded", g["org"], int(g["year"]))
    if kind == "hq":
        return ("hq", g["org"], g["city"])
    if kind == "employer":
        return ("employer", g["person"], g["org"])
    if kind == "born_in":
        return ("born_in", g["person"], g["city"])
    if kind == "population":
        return ("population", g["city"], int(g["pop"].replace(",", "")))
    return None


def parse_question(text: str):
    p = QUESTION.parse(text)
    if p is None:
        return None
    key, g = p
    if key.startswith("n:"):
        return {"kind": "negconj", "year": int(g["year"]), "city": g["city"]}
    path = key.split(":", 1)[1].split(",")
    start = g.get(START_TYPE[path[0]])
    return {"kind": "chain", "start": start, "path": path}


def _value_str(v) -> str:
    return _fmt_pop(v) if isinstance(v, int) and v >= 10000 else str(v)


def determine(world: World, facts: List[tuple]) -> Optional[str]:
    q = world.qstruct
    rel: Dict[tuple, object] = {}
    for f in facts:
        rel[(f[0], f[1])] = f[2]
    if q["kind"] == "chain":
        cur = q["start"]
        for r in q["path"]:
            key = (r, cur)
            if key not in rel:
                return None
            cur = rel[key]
        ans = _value_str(cur)
        return ans if ans in world.choices else None
    status = {}
    for c in world.choices:
        fy, hq = rel.get(("founded", c)), rel.get(("hq", c))
        if (fy is not None and fy >= q["year"]) or (hq is not None and hq == q["city"]):
            status[c] = False
        elif fy is not None and hq is not None:
            status[c] = True
        else:
            status[c] = None
    trues = [c for c, s in status.items() if s is True]
    if len(trues) == 1:
        return trues[0]
    alive = [c for c, s in status.items() if s is not False]
    return alive[0] if len(alive) == 1 else None


def _world(rng, names: NamePool, variant: str):
    used = set()
    n_org, n_person, n_city = int(rng.integers(4, 7)), int(rng.integers(5, 8)), int(rng.integers(4, 7))
    orgs = [names.org(used) for _ in range(n_org)]
    people = [names.person(used) for _ in range(n_person)]
    cities = [names.city(used) for _ in range(n_city)]
    years = rng.choice(range(1950, 2020), size=n_org, replace=False)
    pops = rng.choice(range(40, 990), size=n_city, replace=False) * 1000
    facts = {}
    for o, y in zip(orgs, years):
        facts[("founded", o)] = int(y)
        facts[("hq", o)] = cities[int(rng.integers(n_city))]
    for p in people:
        facts[("employer", p)] = orgs[int(rng.integers(n_org))]
        facts[("born_in", p)] = cities[int(rng.integers(n_city))]
    for c, n in zip(cities, pops):
        facts[("population", c)] = int(n)
    return orgs, people, cities, facts


def _numeric_choices(gold: int, pool, step: int, lo_bound: int, hi_bound: int, rng, fmt):
    r = int(rng.integers(4))
    below = sorted({v for v in pool if lo_bound <= v < gold})
    above = sorted({v for v in pool if gold < v <= hi_bound})
    k = 1
    while len(below) < r:
        cand = gold - step * k * 3
        if cand >= lo_bound and cand not in below:
            below.append(cand)
        k += 1
        if k > 200:
            raise CertificationError("cannot place numeric distractors")
    k = 1
    while len(above) < 3 - r:
        cand = gold + step * k * 3
        if cand <= hi_bound and cand not in above:
            above.append(cand)
        k += 1
        if k > 200:
            raise CertificationError("cannot place numeric distractors")
    pick_b = [below[i] for i in rng.choice(len(below), size=r, replace=False)] if r else []
    pick_a = [above[i] for i in rng.choice(len(above), size=3 - r, replace=False)] if 3 - r else []
    ch = [fmt(v) for v in [gold] + pick_b + pick_a]
    rng.shuffle(ch)
    return ch


def _choices_for(value_type, gold, orgs, people, cities, facts, rng):
    if value_type == "year":
        pool = [v for (k, _), v in facts.items() if k == "founded"]
        return _numeric_choices(int(gold), pool, 1, 1930, 2025, rng, str)
    if value_type == "pop":
        pool = [v for (k, _), v in facts.items() if k == "population"]
        return _numeric_choices(int(gold.replace(",", "")), pool, 1000, 10000, 1500000, rng, _fmt_pop)
    pool = [c for c in cities if c != gold] if value_type == "city" else [o for o in orgs if o != gold]
    pool = list(dict.fromkeys(pool))
    if len(pool) < 3:
        raise CertificationError("not enough distinct distractor choices")
    pick = [pool[i] for i in rng.choice(len(pool), size=3, replace=False)]
    ch = [gold] + pick
    rng.shuffle(ch)
    return ch


def generate(rng, split: str, variant: str, idx: int) -> World:
    names = NamePool(split, rng)
    orgs, people, cities, facts = _world(rng, names, variant)
    style = {"v1": 0, "v2": 1, "v3": 2}[variant]
    n_nodes = {"v1": 8, "v2": 10, "v3": 12}[variant]

    if variant == "v1":
        kind, path = "chain", V1_PATHS[int(rng.integers(len(V1_PATHS)))]
        qkey = "c:" + ",".join(path)
    elif variant == "v2":
        kind, path = "chain", V2_PATHS[int(rng.integers(len(V2_PATHS)))]
        qkey = "c:" + ",".join(path)
    else:
        r = rng.random()
        if r < 0.4:
            kind, path, qkey = "chain", ["employer", "hq", "population"], "c:employer,hq,population"
        elif r < 0.7:
            path = [["employer", "hq"], ["employer", "founded"]][int(rng.integers(2))]
            kind, qkey = "chain", "t:" + ",".join(path)
        else:
            kind, path, qkey = "negconj", None, "n:negconj"

    needed = []
    if kind == "chain":
        st = START_TYPE[path[0]]
        start = {"org": orgs, "person": people, "city": cities}[st]
        start = start[int(rng.integers(len(start)))]
        cur = start
        for r in path:
            needed.append((r, cur, facts[(r, cur)]))
            cur = facts[(r, cur)]
        gold = _value_str(cur)
        choices = _choices_for(VALUE_TYPE[path[-1]], gold, orgs, people, cities, facts, rng)
        fields = {st: start}
        question = QUESTION.render(qkey, 0, **fields)
        qstruct = {"kind": "chain", "start": start, "path": path}
    else:
        for _ in range(50):
            year = int(rng.integers(1965, 2010))
            city = cities[int(rng.integers(len(cities)))]
            sat = [o for o in orgs if facts[("founded", o)] < year and facts[("hq", o)] != city]
            uns = [o for o in orgs if o not in sat]
            if sat and len(uns) >= 3:
                break
        else:
            raise CertificationError("no negconj configuration")
        gold = sat[int(rng.integers(len(sat)))]
        pick = [uns[i] for i in rng.choice(len(uns), size=3, replace=False)]
        choices = [gold] + pick
        rng.shuffle(choices)
        needed = [("founded", gold, facts[("founded", gold)]), ("hq", gold, facts[("hq", gold)])]
        for o in pick:
            if facts[("founded", o)] >= year:
                needed.append(("founded", o, facts[("founded", o)]))
            else:
                needed.append(("hq", o, facts[("hq", o)]))
        question = QUESTION.render(qkey, 0, year=str(year), city=city)
        qstruct = {"kind": "negconj", "year": year, "city": city}

    needed_keys = {(f[0], f[1]) for f in needed}
    ents = {f[1] for f in needed} | {f[2] for f in needed if isinstance(f[2], str)}
    rels = {f[0] for f in needed}
    others = [(k, s, v) for (k, s), v in facts.items() if (k, s) not in needed_keys]
    near = [f for f in others if f[1] in ents or (isinstance(f[2], str) and f[2] in ents) or f[0] in rels]
    far = [f for f in others if f not in near]
    rng.shuffle(near)
    rng.shuffle(far)
    pool = list(needed) + (near + far)[: max(0, n_nodes - len(needed))]
    order = rng.permutation(len(pool))
    nodes = []
    for j, oi in enumerate(order):
        f = pool[oi]
        title_fields = {"org": f[1] if START_TYPE[f[0]] == "org" else None,
                        "person": f[1] if START_TYPE[f[0]] == "person" else None,
                        "city": f[1] if START_TYPE[f[0]] == "city" else None}
        nodes.append(Node(id=f"rec_{j}", text=_render_fact(f, style),
                          title=TITLE[f[0]].format(**{k: v for k, v in title_fields.items() if v}),
                          fact=tuple(f), action=ACTION))

    region = names.word(set(ents))
    initial = (f"An archive of {len(nodes)} short records about organizations, people and places "
               f"in the {region} district. Record titles name the subject of each record.")
    return World(family="retrieval", variant=variant, split=split, question=question,
                 choices=choices, gold=gold, initial_text=initial, visible_facts=[], nodes=nodes,
                 qstruct=qstruct, template_id=f"retrieval_{variant}_{qkey}", actions=[ACTION],
                 instructions=INSTRUCTIONS)


def round_trip(world: World) -> bool:
    for nd in world.nodes:
        if parse_fact(nd.text) != nd.fact:
            raise CertificationError(f"node text does not parse back: {nd.text!r}")
    if parse_question(world.question) != world.qstruct:
        raise CertificationError(f"question does not parse back: {world.question!r}")
    return True
