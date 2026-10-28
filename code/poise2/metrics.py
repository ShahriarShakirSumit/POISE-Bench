# Answer, process and budget scores and the combined POISE score.
from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import numpy as np

WEIGHTS = (0.4, 0.3, 0.3)


def norm(a) -> str:
    return "" if a is None else str(a).strip().lower()


def per_item(pred: Dict, ex: Dict) -> Dict[str, float]:
    G = list(dict.fromkeys(pred.get("evidence_gathered", [])))
    M = set(ex["minimal_sufficient_evidence"])
    inter = len(set(G) & M)
    c_star = ex["oracle_cost"]
    cost = pred.get("total_cost", len(G))
    a = float(norm(pred.get("answer")) == norm(ex["gold_answer"]))
    p = (2 * inter / (len(G) + len(M))) if (G or M) else 1.0
    b = min(1.0, max(0.0, 1.0 - (cost - c_star) / c_star)) if c_star > 0 else float(cost == 0)
    return {
        "acc": a, "process": p, "budget": b,
        "ep": inter / len(G) if G else 0.0,
        "er": inter / len(M) if M else 1.0,
        "regret": (cost - c_star) / c_star if c_star > 0 else 0.0,
        "cost": float(cost),
        "over_search": float(bool(set(G) - M)),
        "under_search": float(bool(M - set(G))),
        "invalid": float(bool(pred.get("invalid", False))),
        "pred_abstain": float(norm(pred.get("answer")) == "abstain"),
        "gold_abstain": float(norm(ex["gold_answer"]) == "abstain"),
    }


def poise_from_means(acc, proc, bud) -> float:
    return WEIGHTS[0] * acc + WEIGHTS[1] * proc + WEIGHTS[2] * bud


def evaluate(preds: List[Dict], examples: List[Dict]) -> Dict:
    assert len(preds) == len(examples)
    rows = [per_item(p, e) for p, e in zip(preds, examples)]
    out = summarize(rows)
    fam = defaultdict(list)
    for r, e in zip(rows, examples):
        fam[e["task_family"]].append(r)
    out["per_family"] = {f: summarize(v) for f, v in sorted(fam.items())}
    return out


def summarize(rows: List[Dict]) -> Dict:
    m = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    tp = sum(r["pred_abstain"] * r["gold_abstain"] for r in rows)
    fp = sum(r["pred_abstain"] * (1 - r["gold_abstain"]) for r in rows)
    fn = sum((1 - r["pred_abstain"]) * r["gold_abstain"] for r in rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    m["abst_precision"], m["abst_recall"] = prec, rec
    m["abst_f1"] = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    m["poise"] = poise_from_means(m["acc"], m["process"], m["budget"])
    m["n"] = len(rows)
    return m


def item_scores(preds: List[Dict], examples: List[Dict]) -> np.ndarray:
    rows = [per_item(p, e) for p, e in zip(preds, examples)]
    return np.array([poise_from_means(r["acc"], r["process"], r["budget"]) for r in rows])
