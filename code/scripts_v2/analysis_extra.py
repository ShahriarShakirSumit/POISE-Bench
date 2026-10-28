# Further analyses, called by make_authoritative_results.py.
from __future__ import annotations

import json
from itertools import combinations

import numpy as np
from scipy import stats

FAMS = ["retrieval", "spatial", "procedural", "abstention"]
CONDS = ["closed", "blind", "preview", "full", "gold"]
B = 10000


def _boot_diff(x, y, seed=7):
    rng = np.random.default_rng(seed)
    d = np.asarray(x, float) - np.asarray(y, float)
    idx = rng.integers(0, len(d), size=(B, len(d)))
    m = d[idx].mean(axis=1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _perm(x, y, seed=42):
    rng = np.random.default_rng(seed)
    d = np.asarray(x, float) - np.asarray(y, float)
    obs = abs(d.mean())
    null = np.abs((rng.choice([-1.0, 1.0], size=(B, len(d))) * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (B + 1))


def _holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (len(items) - i) * p))
        out[k] = running
    return out


def unusable_cause(pred, ex, tier="medium"):
    if not pred.get("invalid"):
        return None
    turns = pred.get("turns") or []
    msg = str(turns[-1]["parsed"][1]) if turns and turns[-1].get("parsed") else ""
    spent = len(pred.get("evidence_gathered", [])) >= ex["budget_tiers"][tier]
    if "already revealed" in msg:
        return "rerequest"
    if "not one of the hidden" in msg:
        return "after_budget" if spent else "bad_id"
    if "not one of the answer" in msg:
        return "not_choice"
    return "format"


def compute(res, exs, by_id, load, models, alg_preds, per_item):
    X = {}
    acc_items, poise_items = {}, {}
    for t in models:
        for c in CONDS:
            P = load(t, c)
            if P is None:
                continue
            rows = [per_item(p, by_id[p["example_id"]]) for p in P]
            acc_items[(t, c)] = np.array([r["acc"] for r in rows])
            poise_items[(t, c)] = np.array([0.4 * r["acc"] + 0.3 * r["process"] + 0.3 * r["budget"] for r in rows])

    gap, title, distr = {}, {}, {}
    for t in models:
        if (t, "gold") in acc_items and (t, "preview") in acc_items:
            gap[t] = _boot_diff(acc_items[(t, "gold")], acc_items[(t, "preview")])
        if (t, "preview") in acc_items and (t, "blind") in acc_items:
            title[t] = _boot_diff(acc_items[(t, "preview")], acc_items[(t, "blind")])
        if (t, "gold") in acc_items and (t, "full") in acc_items:
            distr[t] = _boot_diff(acc_items[(t, "gold")], acc_items[(t, "full")])
    X["seeking_gap_ci"], X["title_value_ci"], X["distraction_ci"] = gap, title, distr
    fam_gap = {}
    for f in FAMS:
        vals = [res["llm"][t]["gold"]["per_family"][f]["acc"] - res["llm"][t]["preview"]["per_family"][f]["acc"]
                for t in models if "gold" in res["llm"].get(t, {}) and "preview" in res["llm"].get(t, {})]
        if vals:
            fam_gap[f] = (float(np.mean(vals)), float(min(vals)), float(max(vals)))
    X["seeking_gap_by_family"] = fam_gap

    scale = {}
    for a, b in (("qwen3-32b", "qwen3-8b"), ("qwen3-14b", "qwen3-8b"), ("qwen3-32b", "qwen3-14b")):
        for c in ("preview", "blind", "gold"):
            if (a, c) in acc_items and (b, c) in acc_items:
                scale[f"{a} vs {b} {c} acc"] = {"diff": float(acc_items[(a, c)].mean() - acc_items[(b, c)].mean()),
                                                "p": _perm(acc_items[(a, c)], acc_items[(b, c)])}
    sh = _holm({k: v["p"] for k, v in scale.items()})
    for k in scale:
        scale[k]["p_holm"] = sh[k]
    X["scale_tests"] = scale

    bm = alg_preds.get("bm25_preview")
    vs_bm = {}
    if bm is not None:
        bmi = np.array([0.4 * r["acc"] + 0.3 * r["process"] + 0.3 * r["budget"]
                        for r in (per_item(p, e) for p, e in zip(bm, exs))])
        raw = {}
        for t in models:
            for c in ("preview", "blind"):
                if (t, c) in poise_items:
                    raw[f"{t}/{c}"] = _perm(poise_items[(t, c)], bmi)
        main_holm = res.get("paired_tests_holm", {})
        hol = {k: main_holm.get(f"{k} vs bm25_preview", _holm(raw)[k]) for k in raw}
        for k, p in raw.items():
            t, c = k.split("/")
            d = float(poise_items[(t, c)].mean() - bmi.mean())
            vs_bm[k] = {"diff": d, "p": p, "p_holm": hol[k],
                        "verdict": "above" if d > 0 and hol[k] < 0.05 else "below" if d < 0 and hol[k] < 0.05 else "tie"}
    X["vs_bm25"] = vs_bm

    causes = {}
    for t in models:
        for c in ("blind", "preview"):
            P = load(t, c)
            if P is None:
                continue
            cnt = {"after_budget": 0, "rerequest": 0, "not_choice": 0, "bad_id": 0, "format": 0}
            for p in P:
                k = unusable_cause(p, by_id[p["example_id"]])
                if k:
                    cnt[k] += 1
            n_un = sum(cnt.values())
            causes[f"{t}/{c}"] = {"n": len(P), "unusable": n_un / len(P), **{k: v / len(P) for k, v in cnt.items()},
                                  "share_after_budget": cnt["after_budget"] / n_un if n_un else 0.0}
    X["unusable_causes"] = causes

    agents = {}
    for k, m in res["algorithmic"].items():
        if k != "oracle":
            agents[k] = m["poise"]
    for t in models:
        for c in ("blind", "preview"):
            m = res["llm"].get(t, {}).get(c)
            if m and m.get("complete"):
                agents[f"{t}/{c}"] = m["poise"]
    pairs = list(combinations(agents.values(), 2))
    X["certified_pairs"] = {str(eps): float(np.mean([abs(a - b) > 3 * eps for a, b in pairs])) for eps in (0.01, 0.02, 0.05)}
    X["n_agent_pairs"] = len(pairs)

    la = [(res["llm"][t][c]["acc"], res["llm"][t][c]["poise"]) for t in models for c in ("blind", "preview")
          if res["llm"].get(t, {}).get(c, {}).get("complete")]
    if len(la) >= 4:
        a, p = zip(*la)
        X["llm_acc_poise_kendall"] = float(stats.kendalltau(a, p)[0])
        X["n_llm_agents"] = len(la)

    abl = {}
    for k, v in res.get("ablations", {}).items():
        run, cond = k.split("/")
        tag = run.split("_", 1)[0]
        P = load(run, cond, root="ablations")
        R = load(tag, cond)
        if P is None or R is None:
            continue
        rp = {p["example_id"]: p for p in R}
        rows_a = [per_item(p, by_id[p["example_id"]]) for p in P]
        rows_r = [per_item(rp[p["example_id"]], by_id[p["example_id"]]) for p in P]
        ia = np.array([r["acc"] for r in rows_a])
        ir = np.array([r["acc"] for r in rows_r])
        pa = np.array([0.4 * r["acc"] + 0.3 * r["process"] + 0.3 * r["budget"] for r in rows_a])
        pr = np.array([0.4 * r["acc"] + 0.3 * r["process"] + 0.3 * r["budget"] for r in rows_r])
        proc_a = float(np.mean([r["process"] for r in rows_a]))
        proc_r = float(np.mean([r["process"] for r in rows_r]))
        bud_a = float(np.mean([r["budget"] for r in rows_a]))
        bud_r = float(np.mean([r["budget"] for r in rows_r]))
        abl[k] = {"acc": float(ia.mean()), "acc_ref": float(ir.mean()), "p_acc": _perm(ia, ir),
                  "poise": float(pa.mean()), "poise_ref": float(pr.mean()), "p_poise": _perm(pa, pr),
                  "process": proc_a, "process_ref": proc_r, "budget": bud_a, "budget_ref": bud_r,
                  "cost": float(np.mean([r["cost"] for r in rows_a])), "cost_ref": float(np.mean([r["cost"] for r in rows_r])),
                  "n": len(P)}
    ha = _holm({k: v["p_acc"] for k, v in abl.items()})
    for k in abl:
        abl[k]["p_acc_holm"] = ha[k]
    X["ablation_tests"] = abl
    X["n_main_tests"] = len(res.get("paired_tests", {}))
    X["transfer_verdicts"] = {k: bool(v["rho"] >= 0.643 and v["p_one_sided_exact"] < 0.05)
                              for k, v in res.get("transfer", {}).items() if k in ("H1", "H2", "H3", "H4")}

    X["privilege"] = res.get("privilege_ablation", {})

    st = res.get("strata_by_M_preview_acc", {})
    if st:
        ks = sorted({k for v in st.values() for k in v})
        X["strata_mean"] = {k: float(np.mean([v[k][0] for v in st.values() if k in v])) for k in ks}
        X["strata_n"] = {k: int(next(iter(v[k][1] for v in st.values() if k in v))) for k in ks}

    hum = res.get("human78", {})
    nat = {}
    for c in ("closed", "blind", "gold", "full"):
        vals = [hum[t][c]["acc"] for t in models if isinstance(hum.get(t), dict) and c in hum[t]]
        if vals:
            nat[c] = (float(min(vals)), float(max(vals)), float(np.mean(vals)))
    X["naturalistic_ranges"] = nat
    mq = res.get("musique", {})
    mqr = {}
    for key in ("abstain_rate_unanswerable", "abstain_rate_answerable", "support_recall", "searches", "em_answerable",
                "answer", "musique_poise"):
        vals = [mq[t][key] for t in models if isinstance(mq.get(t), dict) and mq[t].get("complete")]
        if vals:
            mqr[key] = (float(min(vals)), float(max(vals)), float(np.mean(vals)))
    X["musique_ranges"] = mqr
    return X


import model_registry


def md(res):
    X = res.get("extra", {})
    L = ["", "## 13. Extra analyses (analysis_extra.py)", ""]
    for t, g in X.get("seeking_gap_ci", {}).items():
        L.append(f"- seeking gap {t}: {g[0]:.3f} [{g[1]:.3f}, {g[2]:.3f}]")
    for t, g in X.get("title_value_ci", {}).items():
        L.append(f"- title value {t}: {g[0]:.3f} [{g[1]:.3f}, {g[2]:.3f}]")
    L.append("- seeking gap by family (mean, min, max over models): " + "; ".join(
        f"{f} {v[0]:.2f} ({v[1]:.2f}..{v[2]:.2f})" for f, v in X.get("seeking_gap_by_family", {}).items()))
    for k, v in X.get("vs_bm25", {}).items():
        L.append(f"- vs BM25 {k}: diff {v['diff']:+.3f}, p {v['p']:.4f}, Holm {v['p_holm']:.4f}, {v['verdict']}")
    for k, v in X.get("scale_tests", {}).items():
        L.append(f"- scale {k}: diff {v['diff']:+.3f}, p {v['p']:.4f}, Holm {v['p_holm']:.4f}")
    for k, v in X.get("unusable_causes", {}).items():
        L.append(f"- unusable {k}: {v['unusable']:.3f} (after budget {v['after_budget']:.3f}, re-request {v['rerequest']:.3f}, "
                 f"not a choice {v['not_choice']:.3f}, unlisted id {v['bad_id']:.3f}, format {v['format']:.3f})")
    for k, v in X.get("ablation_tests", {}).items():
        L.append(f"- ablation {k} (n={v['n']}): acc {v['acc']:.3f} vs {v['acc_ref']:.3f} (p {v['p_acc']:.4f}, Holm {v['p_acc_holm']:.4f}); POISE {v['poise']:.3f} "
                 f"vs {v['poise_ref']:.3f} (p {v['p_poise']:.4f}); process {v['process']:.3f} vs {v['process_ref']:.3f}; "
                 f"budget {v['budget']:.3f} vs {v['budget_ref']:.3f}; reveals {v['cost']:.2f} vs {v['cost_ref']:.2f}")
    L.append(f"- certified agent pairs (|dPOISE| > 3 eps): {X.get('certified_pairs')} of {X.get('n_agent_pairs')} pairs")
    if "llm_acc_poise_kendall" in X:
        L.append(f"- Kendall tau, LLM agents ranked by accuracy vs POISE: {X['llm_acc_poise_kendall']:.3f} (n={X['n_llm_agents']})")
    L.append(f"- strata by |M| (mean preview acc over models): {X.get('strata_mean')}")
    L.append(f"- naturalistic ranges (min, max, mean acc): {X.get('naturalistic_ranges')}")
    L.append(f"- MuSiQue ranges (min, max, mean): {X.get('musique_ranges')}")
    return L
