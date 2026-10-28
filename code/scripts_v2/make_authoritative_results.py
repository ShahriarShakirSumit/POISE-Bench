# Turns the saved model outputs into results/authoritative.json and a short summary in results/summary.md.
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from poise2.metrics import evaluate, per_item, poise_from_means
sys.path.insert(0, str(ROOT / "code/scripts_v2"))
import itertools
from musique_metrics import evaluate as mq_evaluate
import analysis_extra

TEST = ROOT / "data/v2/test.json"
ALG = ROOT / "results/algorithmic_test"
import os
_RES = ROOT / "results"
LLM = _RES / "llm"
HUMAN = ROOT / "data/naturalistic/naturalistic78.json"
HUM_RES = _RES / "human78"
MQ = ROOT / "data/musique/transfer_600.json"
MQ_RES = _RES / "musique"
ABL = _RES / "ablations"
OUT_JSON = _RES / "authoritative.json"
OUT_MD = _RES / "summary.md"
FAMS = ["retrieval", "spatial", "procedural", "abstention"]
import model_registry
MODEL_ORDER = model_registry.model_order()
PREREG = model_registry.ORIGINAL
MODEL_NAME = model_registry.NAME
ALG_ORDER = ["direct", "random_order", "always_search", "bm25_preview", "fixed_k_1", "fixed_k_3",
             "fixed_k_5", "heuristic", "oracle"]
ALG_NAME = {"direct": "Direct answer", "random_order": "Random order", "always_search": "Always search",
            "bm25_preview": "BM25 on previews", "fixed_k_1": "Fixed-$K$ (1)$^\\dagger$",
            "fixed_k_3": "Fixed-$K$ (3)$^\\dagger$", "fixed_k_5": "Fixed-$K$ (5)$^\\dagger$",
            "heuristic": "Heuristic$^\\dagger$", "oracle": "Oracle"}
AGENTIC = ["blind", "preview"]
SINGLE = ["closed", "gold", "full"]
B = 10000


def sha(p):
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()[:12]


def item_poise(preds, exs):
    return np.array([poise_from_means(r["acc"], r["process"], r["budget"])
                     for r in (per_item(p, e) for p, e in zip(preds, exs))])


def item_acc(preds, exs):
    return np.array([per_item(p, e)["acc"] for p, e in zip(preds, exs)])


def boot_ci(x, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def perm_test(a, b, seed=42):
    rng = np.random.default_rng(seed)
    d = a - b
    obs = abs(d.mean())
    flips = rng.choice([-1.0, 1.0], size=(B, len(d)))
    null = np.abs((flips * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-12) + 1) / (B + 1))


def main():
    exs = json.load(open(TEST))
    by_id = {e["example_id"]: e for e in exs}
    status, res = [], {"data_sha": sha(TEST), "algorithmic": {}, "llm": {}}

    alg_items = {}
    for name in ALG_ORDER:
        f = ALG / f"predictions_{name}.json"
        if not f.exists():
            status.append(f"missing baseline {name}")
            continue
        preds = json.load(open(f))
        res["algorithmic"][name] = evaluate(preds, exs)
        res["algorithmic"][name]["ci_poise"] = boot_ci(item_poise(preds, exs))
        alg_items[name] = item_poise(preds, exs)
    abl = json.load(open(ALG / "metrics.json"))["privilege_ablation"] if (ALG / "metrics.json").exists() else {}
    res["privilege_ablation"] = abl

    llm_items = {}
    for tag in MODEL_ORDER:
        d = LLM / tag
        res["llm"][tag] = {}
        for cond in AGENTIC + SINGLE:
            f = d / f"predictions_{cond}.json"
            if not f.exists():
                status.append(f"missing run {tag}/{cond}")
                continue
            blob = json.load(open(f))
            preds = blob["predictions"]
            ex_sub = [by_id[p["example_id"]] for p in preds]
            n = len(preds)
            full = n == len(exs)
            if not full:
                status.append(f"incomplete run {tag}/{cond}: {n}/{len(exs)} items")
            m = evaluate(preds, ex_sub)
            m["complete"] = full
            if cond in AGENTIC:
                m["ci_poise"] = boot_ci(item_poise(preds, ex_sub))
                if full:
                    llm_items[(tag, cond)] = item_poise(preds, ex_sub)
            m["ci_acc"] = boot_ci(item_acc(preds, ex_sub))
            m["invalid_rate"] = float(np.mean([bool(p.get("invalid")) for p in preds]))
            res["llm"][tag][cond] = m

    ready = [t for t in MODEL_ORDER if all(res["llm"][t].get(c, {}).get("complete") for c in AGENTIC + SINGLE)]
    status += [f"skipped {t}: not every condition is complete" for t in MODEL_ORDER if t not in ready]
    MODEL_ORDER[:] = ready
    llm_items = {k: v for k, v in llm_items.items() if k[0] in ready}

    tests = {}
    if "bm25_preview" in alg_items:
        for (tag, cond), v in llm_items.items():
            tests[f"{tag}/{cond} vs bm25_preview"] = perm_test(v, alg_items["bm25_preview"])
    for tag in MODEL_ORDER:
        if (tag, "blind") in llm_items and (tag, "preview") in llm_items:
            tests[f"{tag} preview vs blind"] = perm_test(llm_items[(tag, "preview")], llm_items[(tag, "blind")])
    res["paired_tests"] = tests

    pts = [(m["acc"], m["poise"]) for m in res["algorithmic"].values() if m["n"] == len(exs)]
    pts += [(res["llm"][t][c]["acc"], res["llm"][t][c]["poise"]) for t in MODEL_ORDER for c in AGENTIC
            if c in res["llm"][t] and res["llm"][t][c]["complete"]]
    if len(pts) >= 4:
        a, p = zip(*pts)
        res["acc_vs_poise"] = {"pearson": float(stats.pearsonr(a, p)[0]),
                               "spearman": float(stats.spearmanr(a, p)[0]), "n_agents": len(pts)}

    hum = json.load(open(HUMAN))
    hby = {e["example_id"]: e for e in hum}
    res["human78"] = {"data_sha": sha(HUMAN)}
    for tag in MODEL_ORDER:
        for cond in ("blind", "closed", "gold", "full"):
            f = HUM_RES / tag / f"predictions_{cond}.json"
            if not f.exists():
                status.append(f"missing naturalistic run {tag}/{cond}")
                continue
            preds = json.load(open(f))["predictions"]
            m = evaluate(preds, [hby[p["example_id"]] for p in preds])
            m["complete"] = len(preds) == len(hum)
            res["human78"].setdefault(tag, {})[cond] = m

    mq = json.load(open(MQ))
    res["musique"] = {"data_sha": sha(MQ)}
    for tag in MODEL_ORDER:
        f = MQ_RES / tag / "predictions_musique.json"
        if not f.exists():
            status.append(f"missing MuSiQue run {tag}")
            continue
        preds = json.load(open(f))["predictions"]
        m = mq_evaluate(preds, mq)
        m["complete"] = len(preds) == len(mq)
        res["musique"][tag] = m

    def complete(t, c):
        return c in res["llm"].get(t, {}) and res["llm"][t][c]["complete"]

    transfer = {}
    mq_all = [t for t in MODEL_ORDER if t in res["musique"] and res["musique"][t]["complete"]]
    mq_ok = [t for t in mq_all if t in PREREG]
    for name, cond, key_p, key_m in [("H1", "preview", "poise", "musique_poise"), ("H2", "blind", "poise", "musique_poise"),
                                     ("H3", "preview", "acc", "answer"), ("H4", "preview", "process", "process")]:
        ts = [t for t in mq_ok if complete(t, cond)]
        if len(ts) >= 4:
            x = [res["llm"][t][cond][key_p] for t in ts]
            y = [res["musique"][t][key_m] for t in ts]
            transfer[name] = {"models": ts, "x": x, "y": y, **spearman_exact(x, y)}
    ext = [t for t in mq_all if complete(t, "preview")]
    if len(ext) > len(mq_ok) and len(ext) <= 10:
        x = [res["llm"][t]["preview"]["poise"] for t in ext]
        y = [res["musique"][t]["musique_poise"] for t in ext]
        transfer["H1_all_models"] = {"models": ext, "x": x, "y": y, **spearman_exact(x, y)}
    elif len(ext) > 10:
        x = [res["llm"][t]["preview"]["poise"] for t in ext]
        y = [res["musique"][t]["musique_poise"] for t in ext]
        transfer["H1_all_models"] = {"models": ext, "x": x, "y": y, **spearman_mc(x, y)}
    hum_ok = [t for t in MODEL_ORDER if complete(t, "blind") and res["human78"].get(t, {}).get("blind", {}).get("complete")]
    if len(hum_ok) >= 4:
        x = [res["llm"][t]["blind"]["poise"] for t in hum_ok]
        y = [res["human78"][t]["blind"]["poise"] for t in hum_ok]
        transfer["human78_blind_poise"] = {"models": hum_ok, "x": x, "y": y,
                                           **(spearman_exact(x, y) if len(x) <= 10 else spearman_mc(x, y))}
    res["transfer"] = transfer

    agents = {}
    for k in ALG_ORDER:
        if k in res["algorithmic"] and k != "oracle":
            m = res["algorithmic"][k]
            agents[k] = (m["acc"], m["process"], m["budget"])
    for t in MODEL_ORDER:
        for c in AGENTIC:
            m = res["llm"].get(t, {}).get(c)
            if m and m["complete"]:
                agents[f"{t}/{c}"] = (m["acc"], m["process"], m["budget"])
    if len(agents) >= 4:
        names = list(agents)
        base = [poise_from_means(*agents[n]) for n in names]
        taus, grid = [], []
        for a in np.arange(0.1, 0.81, 0.05):
            for b in np.arange(0.1, 0.81, 0.05):
                g = 1 - a - b
                if g < 0.1 - 1e-9:
                    continue
                sc = [a * x + b * y + g * z for x, y, z in (agents[n] for n in names)]
                taus.append(float(stats.kendalltau(base, sc)[0]))
                grid.append((round(float(a), 2), round(float(b), 2), round(float(g), 2)))
        res["weight_sensitivity"] = {"n_agents": len(names), "n_weightings": len(taus),
                                     "kendall_tau_min": min(taus), "kendall_tau_median": float(np.median(taus)),
                                     "frac_tau_ge_0.9": float(np.mean([t >= 0.9 for t in taus])),
                                     "grid": grid, "taus": taus}
    if res["paired_tests"]:
        items = sorted(res["paired_tests"].items(), key=lambda kv: kv[1])
        m_ = len(items)
        holm, running = {}, 0.0
        for i, (k, pv) in enumerate(items):
            running = max(running, min(1.0, (m_ - i) * pv))
            holm[k] = running
        res["paired_tests_holm"] = holm
    strata = {}
    for t in MODEL_ORDER:
        f = LLM / t / "predictions_preview.json"
        if f.exists():
            preds = json.load(open(f))["predictions"]
            if len(preds) == len(exs):
                byk = {}
                for p_, e_ in zip(preds, [by_id[p["example_id"]] for p in preds]):
                    k = min(len(e_["minimal_sufficient_evidence"]), 4)
                    byk.setdefault(k, []).append(per_item(p_, e_)["acc"])
                strata[t] = {str(k): (float(np.mean(v)), len(v)) for k, v in sorted(byk.items())}
    res["strata_by_M_preview_acc"] = strata
    abl = {}
    if ABL.exists():
        for d in sorted(ABL.iterdir()):
            for f in sorted(d.glob("predictions_*.json")):
                if ".shard" in f.name:
                    continue
                blob = json.load(open(f))
                preds = blob["predictions"]
                ex_sub = [by_id[p["example_id"]] for p in preds]
                m = evaluate(preds, ex_sub)
                abl[f"{d.name}/{f.stem.replace('predictions_', '')}"] = {
                    "n": len(preds), "poise": m["poise"], "acc": m["acc"], "process": m["process"],
                    "budget": m["budget"], "cost": m["cost"], "invalid": float(np.mean([bool(p.get("invalid")) for p in preds])),
                    "ids": None}
                tag, variant = d.name.split("_", 1)
                cond = f.stem.replace("predictions_", "")
                ref = LLM / tag / f"predictions_{cond}.json"
                if ref.exists():
                    rp = {p["example_id"]: p for p in json.load(open(ref))["predictions"]}
                    if all(p["example_id"] in rp for p in preds):
                        mr = evaluate([rp[p["example_id"]] for p in preds], ex_sub)
                        abl[f"{d.name}/{cond}"]["reference_same_items"] = {"poise": mr["poise"], "acc": mr["acc"],
                                                                            "process": mr["process"], "cost": mr["cost"]}
    res["ablations"] = abl

    decomp = {}
    for t in MODEL_ORDER:
        r = res["llm"].get(t, {})
        if all(c in r and r[c]["complete"] for c in AGENTIC + SINGLE):
            acc = {c: r[c]["acc"] for c in AGENTIC + SINGLE}
            decomp[t] = {**acc, "seeking_gap": acc["gold"] - acc["preview"], "title_value": acc["preview"] - acc["blind"],
                         "distraction": acc["gold"] - acc["full"],
                         "per_family": {f: {c: r[c]["per_family"][f]["acc"] for c in AGENTIC + SINGLE} for f in FAMS}}
    res["decomposition"] = decomp
    overab = {}
    for t in MODEL_ORDER:
        for cond in AGENTIC:
            f = LLM / t / f"predictions_{cond}.json"
            if not f.exists():
                continue
            preds = json.load(open(f))["predictions"]
            if len(preds) != len(exs):
                continue
            fam_n, fam_a = {}, {}
            for p_ in preds:
                e_ = by_id[p_["example_id"]]
                if e_["gold_answer"] == "abstain":
                    continue
                fam_n[e_["task_family"]] = fam_n.get(e_["task_family"], 0) + 1
                fam_a[e_["task_family"]] = fam_a.get(e_["task_family"], 0) + (p_.get("answer") == "abstain")
            overab.setdefault(t, {})[cond] = {f: fam_a[f] / fam_n[f] for f in fam_n}
    res["over_abstention"] = overab
    cache = {}

    def load(tag, cond, root="llm"):
        key = (root, tag, cond)
        if key not in cache:
            f = (LLM if root == "llm" else ABL) / tag / f"predictions_{cond}.json"
            P = json.load(open(f))["predictions"] if f.exists() else None
            if root == "llm" and P is not None and len(P) != len(exs):
                P = None
            cache[key] = P
        return cache[key]
    alg_preds = {k: json.load(open(ALG / f"predictions_{k}.json")) for k in ("bm25_preview",)
                 if (ALG / f"predictions_{k}.json").exists()}
    res["extra"] = analysis_extra.compute(res, exs, by_id, load, MODEL_ORDER, alg_preds, per_item)
    cache.clear()
    OUT_JSON.write_text(json.dumps(res, indent=1))
    write_md(res, status, exs)


def spearman_mc(x, y, R=200000, seed=11):
    rho = float(stats.spearmanr(x, y)[0])
    rng = np.random.default_rng(seed)
    ry = np.array(stats.rankdata(y))
    rx = np.array(stats.rankdata(x))
    rxc = rx - rx.mean()
    den = np.sqrt((rxc ** 2).sum() * ((ry - ry.mean()) ** 2).sum())
    perms = np.array([rng.permutation(ry) for _ in range(R)])
    null = ((perms - ry.mean()) @ rxc) / den
    return {"rho": rho, "p_one_sided_exact": float((np.sum(null >= rho - 1e-12) + 1) / (R + 1)), "n": len(x),
            "method": "monte carlo"}


def spearman_exact(x, y):
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rho = float(stats.spearmanr(x, y)[0])
    n = len(x)
    count = total = 0
    for perm in itertools.permutations(range(n)):
        r = stats.spearmanr(rx, [ry[i] for i in perm])[0]
        total += 1
        count += r >= rho - 1e-12
    return {"rho": rho, "p_one_sided_exact": count / total, "n": n}


def f3(x):
    return f"{x:.3f}"


def write_md(res, status, exs):
    L = ["# POISE-Bench results",
         f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} by "
         "`code/scripts_v2/make_authoritative_results.py`.", "",
         "## 0. Computed status", f"- test data sha1: `{res['data_sha']}` ({len(exs)} items)"]
    L += [f"- {s}" for s in status] or ["- all inputs complete"]
    L += ["", "## 1. Algorithmic baselines (test, medium budget)", "",
          "| baseline | POISE [95% CI] | acc | EF1 | budget | regret | cost |", "|---|---|---|---|---|---|---|"]
    for k in ALG_ORDER:
        if k in res["algorithmic"]:
            m = res["algorithmic"][k]
            L.append(f"| {k} | {f3(m['poise'])} [{f3(m['ci_poise'][0])}, {f3(m['ci_poise'][1])}] | {f3(m['acc'])} | "
                     f"{f3(m['process'])} | {f3(m['budget'])} | {m['regret']:+.3f} | {m['cost']:.2f} |")
    L += ["", "Privilege ablation (POISE): " + "; ".join(
        f"{k}: true {f3(v['true'])}, shuffled {f3(v['shuffled'])}, uniform {f3(v['uniform'])}"
        for k, v in res["privilege_ablation"].items())]
    L += ["", "## 2. LLM agents", "",
          "| model | cond | complete | POISE [95% CI] | acc [95% CI] | EF1 | budget | cost | invalid |",
          "|---|---|---|---|---|---|---|---|---|"]
    for t in MODEL_ORDER:
        for c in AGENTIC + SINGLE:
            m = res["llm"].get(t, {}).get(c)
            if not m:
                continue
            pci = f"{f3(m['poise'])} [{f3(m['ci_poise'][0])}, {f3(m['ci_poise'][1])}]" if c in AGENTIC else "n/a"
            L.append(f"| {t} | {c} | {m['complete']} | {pci} | {f3(m['acc'])} [{f3(m['ci_acc'][0])}, "
                     f"{f3(m['ci_acc'][1])}] | {f3(m['process'])} | {f3(m['budget'])} | {m['cost']:.2f} | "
                     f"{f3(m['invalid_rate'])} |")
    L += ["", "### Per-family accuracy (preview / blind / gold)", "",
          "| model | " + " | ".join(f"{f} prev/blind/gold" for f in FAMS) + " |",
          "|---|" + "---|" * len(FAMS)]
    for t in MODEL_ORDER:
        r = res["llm"].get(t, {})
        if not r:
            continue
        cells = []
        for f in FAMS:
            vals = [r[c]["per_family"][f]["acc"] if c in r and f in r[c]["per_family"] else float("nan")
                    for c in ("preview", "blind", "gold")]
            cells.append(" / ".join("--" if np.isnan(v) else f"{v:.2f}" for v in vals))
        L.append(f"| {t} | " + " | ".join(cells) + " |")
    L += ["", "## 3. Paired permutation tests (per-item POISE, 10k sign flips)", ""]
    L += [f"- {k}: p = {v:.4f}" for k, v in sorted(res["paired_tests"].items())]
    if "acc_vs_poise" in res:
        a = res["acc_vs_poise"]
        L += ["", f"## 4. Accuracy vs POISE across agents: Pearson {a['pearson']:.3f}, "
                  f"Spearman {a['spearman']:.3f} (n = {a['n_agents']})"]
    L += ["", "## 5. Human-curated naturalistic set (78 tasks)", f"- data sha1 `{res['human78']['data_sha']}`", "",
          "| model | cond | complete | POISE | acc | EF1 | budget | cost |", "|---|---|---|---|---|---|---|---|"]
    for t in MODEL_ORDER:
        for c, m in res["human78"].get(t, {}).items() if isinstance(res["human78"].get(t), dict) else []:
            L.append(f"| {t} | {c} | {m['complete']} | {f3(m['poise'])} | {f3(m['acc'])} | {f3(m['process'])} | "
                     f"{f3(m['budget'])} | {m['cost']:.2f} |")
    L += ["", "## 6. MuSiQue transfer (pre-registered)", f"- data sha1 `{res['musique']['data_sha']}`", "",
          "| model | complete | MuSiQue-POISE | answer | EM ans. | F1 ans. | abstain (unans.) | abstain (ans.) | support recall | searches | invalid |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for t in MODEL_ORDER:
        m = res["musique"].get(t)
        if isinstance(m, dict) and "musique_poise" in m:
            L.append(f"| {t} | {m['complete']} | {f3(m['musique_poise'])} | {f3(m['answer'])} | {f3(m['em_answerable'])} | "
                     f"{f3(m['f1_answerable'])} | {f3(m['abstain_rate_unanswerable'])} | {f3(m['abstain_rate_answerable'])} | "
                     f"{f3(m['support_recall'])} | {m['searches']:.2f} | {f3(m['invalid'])} |")
    L += ["", "### Transfer tests (Spearman, exact one-sided permutation p)", ""]
    for k, v in res.get("transfer", {}).items():
        L.append(f"- {k}: rho = {v['rho']:.3f}, p = {v['p_one_sided_exact']:.4f}, n = {v['n']} ({', '.join(v['models'])})")
    if "weight_sensitivity" in res:
        w = res["weight_sensitivity"]
        L += ["", f"## 7. Weight sensitivity: {w['n_weightings']} weightings, {w['n_agents']} agents; "
                  f"Kendall tau min {w['kendall_tau_min']:.3f}, median {w['kendall_tau_median']:.3f}, "
                  f"share >= 0.9: {w['frac_tau_ge_0.9']:.3f}"]
    if res.get("paired_tests_holm"):
        L += ["", "## 8. Holm-corrected paired tests", ""] + [f"- {k}: p_holm = {v:.4f}" for k, v in res["paired_tests_holm"].items()]
    if res.get("ablations"):
        L += ["", "## 9. Ablations (subset)", "", "| run | n | POISE | acc | proc | cost | invalid | default on same items (POISE / acc) |", "|---|---|---|---|---|---|---|---|"]
        for k, v in res["ablations"].items():
            r = v.get("reference_same_items")
            rs = f"{r['poise']:.3f} / {r['acc']:.3f}" if r else "n/a"
            L.append(f"| {k} | {v['n']} | {v['poise']:.3f} | {v['acc']:.3f} | {v['process']:.3f} | {v['cost']:.2f} | {v['invalid']:.3f} | {rs} |")
    if res.get("strata_by_M_preview_acc"):
        L += ["", "## 10. Preview accuracy by |M| (4 = 4 or more)", ""]
        for t, v in res["strata_by_M_preview_acc"].items():
            L.append(f"- {t}: " + ", ".join(f"|M|={k}: {a:.3f} (n={n})" for k, (a, n) in v.items()))
    if res.get("decomposition"):
        L += ["", "## 11. Seeking vs reading (accuracy)", "",
              "| model | closed | blind | preview | full | gold | seeking gap (gold-preview) | title value (preview-blind) | distraction (gold-full) |",
              "|---|---|---|---|---|---|---|---|---|"]
        for t, d in res["decomposition"].items():
            L.append(f"| {t} | {d['closed']:.3f} | {d['blind']:.3f} | {d['preview']:.3f} | {d['full']:.3f} | {d['gold']:.3f} | "
                     f"{d['seeking_gap']:+.3f} | {d['title_value']:+.3f} | {d['distraction']:+.3f} |")
        L += ["", "### per family (closed / blind / preview / full / gold)", ""]
        for t, d in res["decomposition"].items():
            L.append(f"- {t}: " + "; ".join(f + " " + "/".join(f"{d['per_family'][f][c]:.2f}" for c in ["closed", "blind", "preview", "full", "gold"]) for f in FAMS))
    if res.get("over_abstention"):
        L += ["", "## 12. Over-abstention (abstain when the gold is an answer), agentic conditions", ""]
        for t, v in res["over_abstention"].items():
            for cond, fam in v.items():
                L.append(f"- {t}/{cond}: " + ", ".join(f"{f} {x:.2f}" for f, x in sorted(fam.items())))
    L += analysis_extra.md(res)
    OUT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
