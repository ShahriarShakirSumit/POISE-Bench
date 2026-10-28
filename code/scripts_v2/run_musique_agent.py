# Runs a language model as a free-text search agent on the MuSiQue sample.
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_llm_agent import LocalLM, clean

SYSTEM = (
    "You answer a question by searching a small collection of Wikipedia paragraphs. Each SEARCH returns "
    "the 2 best-matching paragraphs you have not seen yet and costs 1 unit of budget; answering or "
    "abstaining is free. Spend as little budget as you can. If the paragraphs you can find do not contain "
    "the information needed to answer, choose ABSTAIN.\n"
    "Reply in exactly this format:\n"
    "Thought: <one or two short sentences>\n"
    "Action: <exactly one of: SEARCH <query> | ANSWER <short answer> | ABSTAIN>"
)
ACT = re.compile(r"^\s*\**\s*Action\s*\**\s*:\s*\**\s*(SEARCH|ANSWER|ABSTAIN)\b\s*:?\s*(.*)$", re.I | re.M)
LOOSE = re.compile(r"\b(SEARCH|ANSWER|ABSTAIN)\b\s*:?\s*([^\n]*)", re.I)
TOK = re.compile(r"[a-z0-9]+")
TOP_K, BUDGET = 2, 6


class BM25:
    def __init__(self, paras, k1=1.2, b=0.75):
        self.paras = paras
        self.docs = [TOK.findall((p["title"] + " " + p["text"]).lower()) for p in paras]
        self.avg = sum(map(len, self.docs)) / len(self.docs)
        self.df = Counter(t for d in self.docs for t in set(d))
        self.tf = [Counter(d) for d in self.docs]
        self.k1, self.b, self.N = k1, b, len(self.docs)

    def rank(self, query):
        q = TOK.findall(query.lower())
        sc = []
        for d, tf in zip(self.docs, self.tf):
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (self.N - self.df[t] + 0.5) / (self.df[t] + 0.5))
                    s += idf * tf[t] * (self.k1 + 1) / (tf[t] + self.k1 * (1 - self.b + self.b * len(d) / self.avg))
            sc.append(s)
        return sorted(range(self.N), key=lambda i: (-sc[i], i))


def parse(text):
    m = list(ACT.finditer(text)) or list(LOOSE.finditer(text))
    if not m:
        return ("invalid", "no Action line")
    k, arg = m[0].group(1).upper(), clean(m[0].group(2))
    if k == "ABSTAIN":
        return ("abstain", None)
    if not arg:
        return ("invalid", f"{k} needs an argument")
    return (k.lower(), arg)


def prompt(item, st, forced):
    L = [f"Question: {item['question']}", "", "Paragraphs found so far:"]
    if st["seen"]:
        for pid in st["seen"]:
            p = st["by_pid"][pid]
            L.append(f"[{p['title']}] {p['text']}")
    else:
        L.append("none")
    L.append("")
    if forced or st["budget"] <= 0:
        L.append("You cannot search any more. Choose ANSWER or ABSTAIN now.")
    else:
        L.append(f"Budget remaining: {st['budget']} searches")
        if st["queries"]:
            L.append("Your previous queries: " + " | ".join(st["queries"]))
    if st.get("feedback"):
        L.append(f"Note: your previous reply could not be used ({st['feedback']}).")
    L.append("What is your next action?")
    return "\n".join(L)


def run(lm, items, batch, log):
    stop = re.compile(r"(?m)^\s*\**\s*Action\s*\**\s*:[^\n]*\n", re.I)
    S = []
    for it in items:
        S.append({"seen": [], "queries": [], "budget": BUDGET, "done": False, "answer": None, "invalid": False,
                  "streak": 0, "forced": False, "feedback": None, "turns": [], "bm25": BM25(it["paragraphs"]),
                  "by_pid": {p["pid"]: p for p in it["paragraphs"]}})
    t = 0
    while True:
        live = [i for i, s in enumerate(S) if not s["done"]]
        if not live:
            break
        t += 1
        t0 = time.time()
        reps = lm.generate(SYSTEM, [prompt(items[i], S[i], S[i]["forced"]) for i in live], stop, batch)
        log(f"  musique turn {t}: {len(live)} live, {time.time() - t0:.0f}s")
        for i, rep in zip(live, reps):
            st, it = S[i], items[i]
            k, v = parse(rep)
            st["turns"].append({"reply": rep, "parsed": [k, v]})
            st["feedback"] = None
            if k == "search" and not st["forced"] and st["budget"] > 0:
                st["queries"].append(v)
                st["budget"] -= 1
                new = [it["paragraphs"][j]["pid"] for j in st["bm25"].rank(v)
                       if it["paragraphs"][j]["pid"] not in st["seen"]][:TOP_K]
                st["seen"] += new
                st["streak"] = 0
                if st["budget"] <= 0:
                    st["forced"] = True
            elif k == "answer":
                st["answer"], st["done"] = v, True
            elif k == "abstain":
                st["answer"], st["done"] = "abstain", True
            else:
                st["streak"] += 1
                st["feedback"] = v if k == "invalid" else "no more searches allowed"
                if st["forced"] and st["streak"] >= 2:
                    st["done"], st["invalid"] = True, True
                elif st["streak"] >= 2:
                    st["forced"] = True
            if len(st["turns"]) > BUDGET + 5 and not st["done"]:
                st["done"], st["invalid"] = True, True
    return [{"example_id": it["example_id"], "answer": s["answer"], "invalid": s["invalid"],
             "queries": s["queries"], "returned": s["seen"], "n_searches": len(s["queries"]),
             "turns": s["turns"]} for it, s in zip(items, S)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data/musique/transfer_600.json"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    a = ap.parse_args()
    random.seed(0)
    items = json.load(open(a.data))
    if a.limit:
        items = items[:a.limit]
    if a.nshards > 1:
        items = items[a.shard::a.nshards]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fn = out / ("predictions_musique.json" if a.nshards == 1 else f"predictions_musique.shard{a.shard}of{a.nshards}.json")
    if fn.exists() or (out / "predictions_musique.json").exists():
        print("exists, skipping")
        return
    lf = open(out / "run.log", "a")

    def log(m):
        print(m, flush=True)
        lf.write(m + "\n")
        lf.flush()

    lm = LocalLM(a.model, a.cache, a.max_new, mock=a.mock, dtype=a.dtype)
    if a.mock:
        lm._mock = lambda u: random.choice(["Thought: x\nAction: SEARCH " + u.split("\n")[0][10:60] + "\n",
                                            "Thought: x\nAction: ANSWER Paris\n", "Thought: x\nAction: ABSTAIN\n"])
    log(f"model {a.model}: {len(items)} items, batch {a.batch}")
    t0 = time.time()
    preds = run(lm, items, a.batch, log)
    fn.write_text(json.dumps({"model": a.model, "n": len(preds), "predictions": preds}))
    log(f"done musique: {len(preds)} items, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
