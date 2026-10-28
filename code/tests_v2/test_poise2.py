# Unit tests for the generator, certifier, metrics and baselines.
import json
import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE))

from poise2 import FAMILIES
from poise2.baselines import Reader, run_baseline
from poise2.build import BASE_SEED, FAM_OFFSET, SPLITS
from poise2.core import CertificationError, analyse, export_item
from poise2.metrics import evaluate, per_item

DATA = CODE.parent / "data/v2"


def _items(fam, split, n, seed=7):
    mod = FAMILIES[fam]
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        try:
            w = mod.generate(rng, split, SPLITS[split], len(out))
            mod.round_trip(w)
            a = analyse(w, mod.determine)
        except CertificationError:
            continue
        out.append((w, a))
    return out


@pytest.mark.parametrize("fam", list(FAMILIES))
@pytest.mark.parametrize("split", ["train", "dev", "test"])
def test_generation_is_deterministic(fam, split):
    a = [w.question + "|" + "|".join(n.text for n in w.nodes) for w, _ in _items(fam, split, 20, seed=3)]
    b = [w.question + "|" + "|".join(n.text for n in w.nodes) for w, _ in _items(fam, split, 20, seed=3)]
    assert a == b


@pytest.mark.parametrize("fam", list(FAMILIES))
def test_certificate_invariants(fam):
    mod = FAMILIES[fam]
    for w, a in _items(fam, "test", 40):
        facts = {nd.id: nd.fact for nd in w.nodes}
        M = a["minimum_sets"][0]
        vis = list(w.visible_facts)
        assert len(a["minimum_sets"]) == 1
        assert mod.determine(w, vis) is None
        assert mod.determine(w, vis + [f for f in facts.values() if f]) == w.gold
        assert mod.determine(w, vis + [facts[m] for m in M]) == w.gold
        for d in M:
            assert mod.determine(w, vis + [facts[m] for m in M if m != d]) is None
        assert w.gold in w.choices and len(set(w.choices)) == 4


@pytest.mark.parametrize("fam", list(FAMILIES))
def test_text_round_trip(fam):
    mod = FAMILIES[fam]
    for w, _ in _items(fam, "dev", 30):
        for nd in w.nodes:
            assert mod.parse_fact(nd.text) == (nd.fact if fam != "procedural" else nd.fact) or \
                (fam == "procedural" and sorted(mod.parse_fact(nd.text)[2]) == sorted(nd.fact[2]))


def test_metrics_reference_points():
    exs = []
    rng = np.random.default_rng(0)
    for fam in FAMILIES:
        for i, (w, a) in enumerate(_items(fam, "test", 5)):
            exs.append(export_item(w, a, f"{fam}_{i}", 0, rng))
    oracle = [run_baseline("oracle", e, rng) for e in exs]
    direct = [{"answer": None, "evidence_gathered": [], "total_cost": 0} for _ in exs]
    mo, md = evaluate(oracle, exs), evaluate(direct, exs)
    assert mo["poise"] == pytest.approx(1.0)
    assert md["process"] == 0.0 and md["budget"] == 1.0 and md["acc"] == 0.0
    e = exs[0]
    over = per_item({"answer": e["gold_answer"], "evidence_gathered": [n["id"] for n in e["hidden_evidence_graph"]["nodes"]],
                     "total_cost": len(e["hidden_evidence_graph"]["nodes"])}, e)
    assert over["budget"] < 1.0


def test_reader_uses_text_only():
    exs = json.load(open(DATA / "test.json"))[:40] if (DATA / "test.json").exists() else []
    for e in exs:
        r = Reader(e)
        assert r.determined(e["minimal_sufficient_evidence"]) == e["gold_answer"]
        assert r.determined([]) is None


def test_released_splits_seeds_are_distinct():
    seeds = {BASE_SEED[s] * 100 + FAM_OFFSET[f] for s in SPLITS for f in FAMILIES}
    assert len(seeds) == len(SPLITS) * len(FAMILIES)
