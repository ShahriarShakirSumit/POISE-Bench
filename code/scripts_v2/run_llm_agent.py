# Runs an open-weight language model as an agent on POISE-Bench in all five conditions.
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path

SYSTEM_AGENT = (
    "You are an information-seeking agent. You answer a multiple-choice question by revealing hidden "
    "evidence nodes one at a time. Each reveal costs 1 unit of budget; answering or abstaining is free. "
    "Spend as little budget as you can, but make sure the evidence supports your answer.\n"
    "Reply in exactly this format:\n"
    "Thought: <one or two short sentences>\n"
    "Action: <exactly one of: GATHER <node id> | ANSWER <one answer choice, copied exactly> | ABSTAIN>"
)
SYSTEM_AGENT_TERSE = (
    "You are an information-seeking agent. You answer a multiple-choice question by revealing hidden "
    "evidence nodes one at a time. Each reveal costs 1 unit of budget; answering or abstaining is free. "
    "Spend as little budget as you can, but make sure the evidence supports your answer.\n"
    "Reply with exactly one line:\n"
    "Action: <exactly one of: GATHER <node id> | ANSWER <one answer choice, copied exactly> | ABSTAIN>"
)
SYSTEM_SINGLE_CLOSED = (
    "Answer the multiple-choice question using only the information given. If the information does not "
    "determine the answer, still choose the most likely option.\n"
    "Reply in exactly this format:\n"
    "Thought: <short reasoning>\n"
    "Answer: <one answer choice, copied exactly>"
)
SYSTEM_SINGLE = (
    "Answer the multiple-choice question using only the information given.\n"
    "Reply in exactly this format:\n"
    "Thought: <short reasoning>\n"
    "Answer: <one answer choice, copied exactly>"
)
ACTION_RX = re.compile(r"^\s*\**\s*Action\s*\**\s*:\s*\**\s*(GATHER|ANSWER|ABSTAIN)\b\s*:?\s*(.*)$", re.I | re.M)
LOOSE_RX = re.compile(r"\b(GATHER|ANSWER|ABSTAIN)\b\s*:?\s*([^\n]*)", re.I)
ANSWER_RX = re.compile(r"^\s*\**\s*(?:Final\s+)?Answer\s*\**\s*:(?!\s*choices)\s*\**\s*(.+?)\s*$", re.I | re.M)


def clean(s: str) -> str:
    s = s.strip().strip("*").strip()
    s = re.sub(r"^[\"'`\[\(<]+|[\"'`\]\)>.,;:!]+$", "", s.strip())
    return s.strip()


def match_choice(text: str, choices):
    t = clean(text).lower()
    for c in choices:
        if t == c.lower():
            return c
    hits = [c for c in choices if c.lower() in t]
    if len(hits) == 1:
        return hits[0]
    hits = [c for c in choices if t and t in c.lower()]
    return hits[0] if len(hits) == 1 else None


def parse_agent(text: str, choices, available, already=()):
    m = list(ACTION_RX.finditer(text)) or list(LOOSE_RX.finditer(text))
    if not m:
        return ("invalid", "no Action line")
    kind, arg = m[0].group(1).upper(), m[0].group(2)
    if kind == "ABSTAIN":
        return ("abstain", None)
    if kind == "GATHER":
        a = clean(arg)
        first = a.split()[0] if a else ""
        mm = re.match(r"([a-z]+)[ _-]?(\d+)", a.lower())
        cands = [first] + ([f"{mm.group(1)}_{mm.group(2)}"] if mm else [])
        for c in cands:
            if c in available:
                return ("gather", c)
        for c in cands:
            if c in already:
                return ("invalid", f"'{c}' was already revealed; its text is listed under 'Evidence revealed so far'")
        return ("invalid", f"'{arg.strip()}' is not one of the hidden nodes you can reveal")
    c = match_choice(arg, choices)
    return ("answer", c) if c is not None else ("invalid", f"'{arg.strip()}' is not one of the answer choices")


ANSWER_INLINE_RX = re.compile(r"(?:Final\s+)?Answer\s*\**\s*:(?!\s*choices)\s*\**\s*([^\n]+)", re.I)


def parse_single(text: str, choices):
    for rx in (ANSWER_RX, ANSWER_INLINE_RX):
        for mm in reversed(list(rx.finditer(text))):
            c = match_choice(mm.group(1), choices)
            if c is not None:
                return c
    return None


def fmt_choices(ex):
    return " | ".join(ex["answer_space"])


def agent_prompt(ex, st, condition, forced=False):
    nodes = ex["hidden_evidence_graph"]["nodes"]
    lines = [f"Task rules: {ex['instructions']}", f"Question: {ex['question']}",
             f"Answer choices: {fmt_choices(ex)}", "Initial observation:", ex["initial_observation"]["text"], ""]
    lines.append("Evidence revealed so far:")
    if st["gathered"]:
        txt = {n["id"]: n["content_text"] for n in nodes}
        lines += [f"[{g}] {txt[g]}" for g in st["gathered"]]
    else:
        lines.append("none")
    lines.append("")
    avail = [n for n in nodes if n["id"] not in st["gathered"]]
    if forced or st["budget"] <= 0 or not avail:
        lines.append("You cannot reveal more evidence. Choose ANSWER or ABSTAIN now.")
    else:
        lines.append(f"Hidden nodes you can reveal ({len(avail)} left, cost 1 each):")
        if condition == "preview":
            lines += [f"- {n['id']}: {n['title']}" for n in avail]
        else:
            lines += [f"- {n['id']}" for n in avail]
        lines.append(f"Budget remaining: {st['budget']}")
    if st.get("feedback"):
        lines.append(f"Note: your previous reply could not be used ({st['feedback']}).")
    lines.append("What is your next action?")
    return "\n".join(lines)


def single_prompt(ex, condition, rng):
    nodes = ex["hidden_evidence_graph"]["nodes"]
    lines = [f"Task rules: {ex['instructions']}", f"Question: {ex['question']}",
             f"Answer choices: {fmt_choices(ex)}", "Information:", ex["initial_observation"]["text"]]
    if condition in ("gold", "full"):
        ids = set(ex["minimal_sufficient_evidence"]) if condition == "gold" else {n["id"] for n in nodes}
        ev = [n["content_text"] for n in nodes if n["id"] in ids]
        rng.shuffle(ev)
        lines += ["Evidence:"] + [f"- {t}" for t in ev]
    return "\n".join(lines)


class LocalLM:
    def __init__(self, model, cache, max_new, mock=False, thinking=False, max_new_single=512, dtype="fp16"):
        self.mock, self.max_new, self.thinking = mock, max_new, thinking
        self.max_new_single = max_new_single
        if mock:
            self.name = "mock"
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model, cache_dir=cache, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1")
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        kw = dict(cache_dir=cache, local_files_only=os.environ.get("HF_HUB_OFFLINE") == "1", device_map="auto",
                  torch_dtype={"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[dtype])
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model, **kw)
        except ValueError:
            from transformers import AutoModelForImageTextToText
            self.model = AutoModelForImageTextToText.from_pretrained(model, **kw)
        self.model.eval()
        self.name = model

        tok = self.tok

        class StopOnLine(StoppingCriteria):
            def __init__(self, rx, start):
                self.rx, self.start = rx, start

            def __call__(self, input_ids, scores, **kw):
                tails = tok.batch_decode(input_ids[:, self.start:], skip_special_tokens=True)
                return torch.tensor([bool(self.rx.search(t)) for t in tails], device=input_ids.device)

        self.StopOnLine = StopOnLine

    def chat(self, system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kw = {"tokenize": False, "add_generation_prompt": True, "enable_thinking": self.thinking}
        try:
            return self.tok.apply_chat_template(msgs, **kw)
        except Exception:
            return self.tok.apply_chat_template([{"role": "user", "content": system + "\n\n" + user}], **kw)

    def generate(self, system, users, stop_rx, batch, max_new=None):
        if self.mock:
            return [self._mock(u) for u in users]
        torch = self.torch
        texts = [self.chat(system, u) for u in users]
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        out = [None] * len(texts)
        for s in range(0, len(order), batch):
            for i, d in self._generate_batch(system, texts, order[s:s + batch], stop_rx, max_new):
                out[i] = d
        return out

    def _generate_batch(self, system, texts, idx, stop_rx, max_new):
        torch, oom = self.torch, False
        try:
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True,
                           add_special_tokens=False).to(self.model.device)
            start = enc["input_ids"].shape[1]
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=max_new or self.max_new, do_sample=False,
                                          temperature=None, top_p=None, top_k=None,
                                          pad_token_id=self.tok.pad_token_id,
                                          stopping_criteria=[self.StopOnLine(stop_rx, start)])
        except torch.cuda.OutOfMemoryError:
            if len(idx) == 1:
                raise
            oom = True
        if oom:
            torch.cuda.empty_cache()
            h = len(idx) // 2
            return (self._generate_batch(system, texts, idx[:h], stop_rx, max_new)
                    + self._generate_batch(system, texts, idx[h:], stop_rx, max_new))
        dec = self.tok.batch_decode(gen[:, start:], skip_special_tokens=True)
        return list(zip(idx, dec))

    def _mock(self, user):
        ids = re.findall(r"^- (\w+_\d+)", user, re.M)
        ch = re.search(r"Answer choices: (.*)", user).group(1).split(" | ")
        if ids and random.random() < 0.6:
            return f"Thought: look.\nAction: GATHER {random.choice(ids)}\n"
        if random.random() < 0.05:
            return "Thought: hmm\nAction: GATHER nonsense\n"
        return f"Thought: done.\nAction: ANSWER {random.choice(ch)}\nAnswer: {random.choice(ch)}\n"


def run_agentic(lm, examples, condition, batch, log, tier="medium", system=SYSTEM_AGENT):
    stop = re.compile(r"(?m)^\s*\**\s*Action\s*\**\s*:[^\n]*\n", re.I)
    states = []
    for ex in examples:
        states.append({"gathered": [], "budget": ex["budget_tiers"][tier], "done": False,
                       "answer": None, "invalid": False, "n_invalid": 0, "streak": 0,
                       "feedback": None, "forced": False, "turns": []})
    turn = 0
    while True:
        live = [i for i, s in enumerate(states) if not s["done"]]
        if not live:
            break
        turn += 1
        prompts = [agent_prompt(examples[i], states[i], condition, states[i]["forced"]) for i in live]
        t0 = time.time()
        replies = lm.generate(system, prompts, stop, batch)
        log(f"  {condition} turn {turn}: {len(live)} live, {time.time() - t0:.0f}s")
        for i, rep in zip(live, replies):
            ex, st = examples[i], states[i]
            nodes = ex["hidden_evidence_graph"]["nodes"]
            avail = [n["id"] for n in nodes if n["id"] not in st["gathered"]]
            can_gather = st["budget"] > 0 and avail and not st["forced"]
            kind, val = parse_agent(rep, ex["answer_space"], avail if can_gather else [], already=st["gathered"])
            st["turns"].append({"reply": rep, "parsed": [kind, val]})
            st["feedback"] = None
            if kind == "gather":
                st["gathered"].append(val)
                st["budget"] -= 1
                st["streak"] = 0
            elif kind == "answer":
                st["answer"], st["done"] = val, True
            elif kind == "abstain":
                st["answer"], st["done"] = "abstain", True
            else:
                st["n_invalid"] += 1
                st["streak"] += 1
                st["feedback"] = val
                if st["forced"] or st["streak"] >= 3:
                    st["done"], st["invalid"] = True, True
                elif st["streak"] >= 2 or not can_gather:
                    st["forced"] = True
            if not st["done"] and (st["budget"] <= 0 or not [n for n in avail if n not in st["gathered"]]):
                st["forced"] = True
            if len(st["turns"]) > ex["budget_tiers"][tier] + 6 and not st["done"]:
                st["done"], st["invalid"] = True, True
    preds = []
    act = None
    for ex, st in zip(examples, states):
        act = {e["target"]: e["action"] for e in ex["hidden_evidence_graph"]["edges"]}
        preds.append({"example_id": ex["example_id"], "answer": st["answer"],
                      "evidence_gathered": st["gathered"], "total_cost": len(st["gathered"]),
                      "actions": [f"{act[g]}:{g}" for g in st["gathered"]] + [
                          "abstain" if st["answer"] == "abstain" else "answer_now"],
                      "invalid": st["invalid"], "n_invalid_actions": st["n_invalid"], "turns": st["turns"]})
    return preds


def run_single(lm, examples, condition, batch, log):
    stop = re.compile(r"(?m)^\s*\**\s*(?:Final\s+)?Answer\s*\**\s*:(?!\s*choices)[^\n]*\n", re.I)
    rng = random.Random(0)
    prompts = [single_prompt(ex, condition, rng) for ex in examples]
    t0 = time.time()
    system = SYSTEM_SINGLE_CLOSED if condition == "closed" else SYSTEM_SINGLE
    replies = lm.generate(system, prompts, stop, batch, max_new=lm.max_new_single)
    log(f"  {condition}: {len(prompts)} prompts, {time.time() - t0:.0f}s")
    preds = []
    for ex, rep in zip(examples, replies):
        ans = parse_single(rep, ex["answer_space"])
        gathered = (list(ex["minimal_sufficient_evidence"]) if condition == "gold" else
                    [n["id"] for n in ex["hidden_evidence_graph"]["nodes"]] if condition == "full" else [])
        preds.append({"example_id": ex["example_id"], "answer": ans, "evidence_gathered": gathered,
                      "total_cost": len(gathered), "invalid": ans is None, "turns": [{"reply": rep}]})
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--conditions", default="blind,preview,closed,gold,full")
    ap.add_argument("--split", default="test")
    ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data/v2"))
    ap.add_argument("--examples", default=None, help="explicit JSON file of examples (e.g. the naturalistic set)")
    ap.add_argument("--limit", type=int, default=0, help="stratified subset: limit/4 items per family")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max_new", type=int, default=256, help="token cap per agentic turn")
    ap.add_argument("--max_new_single", type=int, default=512, help="token cap for single-turn answers")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--tier", default="medium", choices=["easy", "medium", "hard"])
    ap.add_argument("--prompt", default="default", choices=["default", "terse"])
    ap.add_argument("--thinking", action="store_true", help="Qwen3 thinking mode (default off)")
    ap.add_argument("--shard", type=int, default=0, help="shard index (items shard, shard+nshards, ...)")
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    args = ap.parse_args()
    random.seed(0)
    path = Path(args.examples) if args.examples else Path(args.data) / f"{args.split}.json"
    examples = json.load(open(path))
    if args.limit:
        per = args.limit // 4
        fam = {}
        for ex in examples:
            fam.setdefault(ex["task_family"], [])
            if len(fam[ex["task_family"]]) < per:
                fam[ex["task_family"]].append(ex)
        examples = [e for v in fam.values() for e in v]
    if args.nshards > 1:
        examples = examples[args.shard::args.nshards]
    suffix = "" if args.nshards == 1 else f".shard{args.shard}of{args.nshards}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    logf = open(out / "run.log", "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    lm = LocalLM(args.model, args.cache, args.max_new, mock=args.mock, thinking=args.thinking,
                 max_new_single=max(args.max_new_single, args.max_new), dtype=args.dtype)
    system = SYSTEM_AGENT_TERSE if args.prompt == "terse" else SYSTEM_AGENT
    log(f"model {args.model} loaded in {time.time() - t0:.0f}s; {len(examples)} examples from {path.name}; "
        f"new-token caps {lm.max_new} per agentic turn, {lm.max_new_single} per single-turn answer; batch {args.batch}")
    run_conditions(lm, args.model, examples, path, args.conditions.split(","), out, suffix, args.batch, log,
                   tier=args.tier, prompt=args.prompt, thinking=args.thinking, system=system)


def run_conditions(lm, model, examples, path, conds, out, suffix, batch, log, tier="medium", prompt="default",
                   thinking=False, system=SYSTEM_AGENT):
    for cond in conds:
        fn = out / f"predictions_{cond}{suffix}.json"
        if fn.exists() or (suffix and (out / f"predictions_{cond}.json").exists()):
            log(f"skip {cond} (exists)")
            continue
        t1 = time.time()
        if cond in ("blind", "preview"):
            preds = run_agentic(lm, examples, cond, batch, log, tier=tier, system=system)
        else:
            preds = run_single(lm, examples, cond, batch, log)
        fn.write_text(json.dumps({"model": model, "condition": cond, "source": str(path),
                                  "tier": tier, "prompt": prompt, "thinking": thinking,
                                  "n": len(preds), "predictions": preds}))
        acc = sum(p["answer"] is not None and p["answer"].lower() == e["gold_answer"].lower()
                  for p, e in zip(preds, examples)) / len(examples)
        log(f"done {cond}: acc {acc:.3f}, invalid {sum(p['invalid'] for p in preds)}, {time.time() - t1:.0f}s")


if __name__ == "__main__":
    main()
