# Scores the annotators' sheets against the computed evidence.
import itertools
import json
import sys
from pathlib import Path

N = Path(__file__).resolve().parents[1]
KEY = json.load(open(N / "annotation/answer_key.json"))
ITEMS = {it["key"]: it for it in json.load(open(N / "annotation/items_public.json"))}
WORDS = {2: "two", 3: "three", 4: "four", 5: "five"}


def load_sheets():
    sheets, unnamed = {}, 0
    for f in sorted((N / "annotation/submissions").rglob("*.json")):
        d = json.load(open(f))
        if "answers" not in d:
            continue
        code = (d.get("code") or "").strip().upper()
        if not code:
            unnamed += 1
            code = f"ANNOTATOR{unnamed}"
        if code in sheets:
            for k, v in d["answers"].items():
                if k not in sheets[code] or (v.get("t") or "") > (sheets[code][k].get("t") or ""):
                    sheets[code][k] = v
        else:
            sheets[code] = dict(d["answers"])
    return sheets


def complete(k, a):
    return bool(a and a.get("answer") and a.get("needed") and (k.startswith("A-") or a.get("sense")))


def kappa(x, y):
    n = len(x)
    if n == 0:
        return float("nan")
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = sum(x) / n, sum(y) / n
    pe = px * py + (1 - px) * (1 - py)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def record_labels(keys, sheet):
    out = []
    for k in keys:
        needed = set(sheet[k]["needed"]) if isinstance(sheet, dict) and k in sheet else set(KEY[k]["mandatory"])
        out += [int(r["label"] in needed) for r in ITEMS[k]["records"]]
    return out


def main():
    min_items = int(sys.argv[sys.argv.index("--min-items") + 1]) if "--min-items" in sys.argv else 1
    sheets = {c: s for c, s in load_sheets().items() if sum(complete(k, v) for k, v in s.items()) >= min_items}
    codes = sorted(sheets)
    lines = ["# Human evidence check", "",
             f"Annotators: {len(codes)} ({', '.join(codes) or 'none yet'})", ""]
    M = {}
    if not codes:
        (N / "annotation/RESULTS.md").write_text("\n".join(lines) + "\n")
        print("\n".join(lines))
        return
    partA = [k for k in ITEMS if k.startswith("A-")]
    partB = [k for k in ITEMS if k.startswith("B-")]
    per = {}
    for c in codes:
        s = sheets[c]
        doneA = [k for k in partA if complete(k, s.get(k))]
        doneB = [k for k in partB if complete(k, s.get(k))]
        exact = sum(set(s[k]["needed"]) == set(KEY[k]["mandatory"]) for k in doneA)
        tp = sum(len(set(s[k]["needed"]) & set(KEY[k]["mandatory"])) for k in doneA)
        npred = sum(len(s[k]["needed"]) for k in doneA)
        nm = sum(len(KEY[k]["mandatory"]) for k in doneA)
        accA = sum(s[k]["answer"] == KEY[k]["gold"] for k in doneA)
        accB = sum(s[k]["answer"] == KEY[k]["gold"] for k in doneB)
        sense = sum(s[k].get("sense") == "yes" for k in doneB)
        exactB = sum(set(s[k]["needed"]) == set(KEY[k]["mandatory"]) for k in doneB)
        kM = kappa(record_labels(doneA, s), record_labels(doneA, None))
        per[c] = dict(nA=len(doneA), nB=len(doneB), exact=exact / max(1, len(doneA)), prec=tp / max(1, npred),
                      rec=tp / max(1, nm), accA=accA / max(1, len(doneA)), accB=accB / max(1, len(doneB)),
                      sense=sense / max(1, len(doneB)), exactB=exactB / max(1, len(doneB)), kappaM=kM)
        lines.append(f"- {c}: part A {len(doneA)}/48, part B {len(doneB)}/78; A exact match with M {per[c]['exact']:.2f}, "
                     f"precision {per[c]['prec']:.2f}, recall {per[c]['rec']:.2f}, kappa vs M {kM:.2f}, answer acc {per[c]['accA']:.2f}; "
                     f"B makes sense {per[c]['sense']:.2f}, answer acc {per[c]['accB']:.2f}, exact match {per[c]['exactB']:.2f}")
    pairs = []
    for a, b in itertools.combinations(codes, 2):
        common = [k for k in partA if complete(k, sheets[a].get(k)) and complete(k, sheets[b].get(k))]
        if common:
            pairs.append((a, b, kappa(record_labels(common, sheets[a]), record_labels(common, sheets[b])),
                          sum(sheets[a][k]["answer"] == sheets[b][k]["answer"] for k in common) / len(common), len(common)))
            lines.append(f"- {a} vs {b}: record-level kappa {pairs[-1][2]:.2f}, same answer {pairs[-1][3]:.2f} ({len(common)} items)")
    mean = lambda key: sum(per[c][key] for c in codes) / len(codes)
    M.update({"AnnN": str(len(codes)), "AnnNWord": WORDS.get(len(codes), str(len(codes))),
              "AnnNA": str(min(per[c]["nA"] for c in codes)), "AnnNB": str(min(per[c]["nB"] for c in codes)),
              "AnnExactM": f"{mean('exact'):.2f}", "AnnPrecM": f"{mean('prec'):.2f}", "AnnRecM": f"{mean('rec'):.2f}",
              "AnnKappaM": f"{mean('kappaM'):.2f}", "AnnAccA": f"{mean('accA'):.2f}", "AnnAccB": f"{mean('accB'):.2f}",
              "AnnSenseB": f"{100 * mean('sense'):.0f}", "AnnExactB": f"{mean('exactB'):.2f}"})
    if pairs:
        M["AnnKappaPair"] = f"{sum(p[2] for p in pairs) / len(pairs):.2f}"
        M["AnnSameAnswer"] = f"{sum(p[3] for p in pairs) / len(pairs):.2f}"
    ORD = {1: "One", 2: "Two", 3: "Three"}
    for i, c in enumerate(codes, 1):
        s, tag = sheets[c], ORD.get(i, str(i))
        doneA = [k for k in partA if complete(k, s.get(k))]
        M[f"AnnExact{tag}"] = str(sum(set(s[k]["needed"]) == set(KEY[k]["mandatory"]) for k in doneA))
        M[f"AnnKappa{tag}"] = f"{per[c]['kappaM']:.2f}"
        for fam, ff in (("retrieval", "Retr"), ("spatial", "Spat"), ("procedural", "Proc"), ("abstention", "Abst")):
            ks = [k for k in doneA if KEY[k]["family"] == fam]
            M[f"AnnExact{tag}{ff}"] = str(sum(set(s[k]["needed"]) == set(KEY[k]["mandatory"]) for k in ks))
            M[f"AnnSub{tag}{ff}"] = str(sum(set(s[k]["needed"]) < set(KEY[k]["mandatory"]) for k in ks))
            M[f"AnnSup{tag}{ff}"] = str(sum(set(s[k]["needed"]) > set(KEY[k]["mandatory"]) for k in ks))
        M[f"AnnSup{tag}ProcAbst"] = str(int(M[f"AnnSup{tag}Proc"]) + int(M[f"AnnSup{tag}Abst"]))
    M["AnnAllCorrect"] = "all" if all(per[c]["accA"] == 1.0 for c in codes) else "most"
    json.dump({"per": per, "summary": M}, open(N / "annotation/RESULTS.json", "w"), indent=1)
    (N / "annotation/RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
