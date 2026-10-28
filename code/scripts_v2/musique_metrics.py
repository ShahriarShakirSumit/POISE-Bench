# Scores for the MuSiQue study, fixed before any model was run on it.
import re
import string

WEIGHTS = (0.4, 0.3, 0.3)


def normalize(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    common = sum(min(p.count(t), g.count(t)) for t in set(p))
    if not p or not g or common == 0:
        return 0.0
    pr, rc = common / len(p), common / len(g)
    return 2 * pr * rc / (pr + rc)


def score_item(pred, item):
    ans = pred.get("answer")
    abst = ans is not None and normalize(ans) == "abstain"
    golds = [item["gold_answer"]] + item.get("answer_aliases", [])
    if item["answerable"]:
        em = float(ans is not None and not abst and any(normalize(ans) == normalize(g) for g in golds))
        tf1 = max(f1(ans, g) for g in golds) if (ans is not None and not abst) else 0.0
        sup = {p["pid"] for p in item["paragraphs"] if p["is_supporting"]}
        got = set(pred.get("returned", []))
        inter = len(got & sup)
        proc = 2 * inter / (len(got) + len(sup)) if (got or sup) else 1.0
        recall = inter / len(sup)
    else:
        em, tf1, proc, recall = float(abst), float(abst), None, None
    h = item["hops"]
    c = pred.get("n_searches", 0)
    bud = min(1.0, max(0.0, 1 - (c - h) / h))
    return {"answer": em, "f1": tf1, "process": proc, "recall": recall, "budget": bud, "searches": c,
            "abstained": float(abst), "answerable": item["answerable"], "invalid": float(bool(pred.get("invalid")))}


def evaluate(preds, items):
    by = {i["example_id"]: i for i in items}
    rows = [score_item(p, by[p["example_id"]]) for p in preds]
    ans = [r for r in rows if r["answerable"]]
    un = [r for r in rows if not r["answerable"]]
    mean = lambda xs: sum(xs) / len(xs) if xs else float("nan")
    out = {
        "n": len(rows),
        "answer": mean([r["answer"] for r in rows]),
        "process": mean([r["process"] for r in ans]),
        "budget": mean([r["budget"] for r in rows]),
        "em_answerable": mean([r["answer"] for r in ans]),
        "f1_answerable": mean([r["f1"] for r in ans]),
        "abstain_rate_unanswerable": mean([r["abstained"] for r in un]),
        "abstain_rate_answerable": mean([r["abstained"] for r in ans]),
        "support_recall": mean([r["recall"] for r in ans]),
        "searches": mean([r["searches"] for r in rows]),
        "invalid": mean([r["invalid"] for r in rows]),
    }
    out["musique_poise"] = WEIGHTS[0] * out["answer"] + WEIGHTS[1] * out["process"] + WEIGHTS[2] * out["budget"]
    return out
