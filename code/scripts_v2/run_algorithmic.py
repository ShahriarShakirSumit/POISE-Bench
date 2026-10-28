# Runs every algorithmic baseline on a split.
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from poise2.baselines import ALL, PRIVILEGED, run_baseline
from poise2.metrics import evaluate

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="test")
ap.add_argument("--data", default=str(Path(__file__).resolve().parents[2] / "data/v2"))
ap.add_argument("--out", required=True)
args = ap.parse_args()
examples = json.load(open(Path(args.data) / f"{args.split}.json"))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)

metrics = {}
for name in ALL:
    rng = np.random.default_rng(42)
    preds = [run_baseline(name, ex, rng) for ex in examples]
    (out / f"predictions_{name}.json").write_text(json.dumps(preds))
    metrics[name] = evaluate(preds, examples)
    m = metrics[name]
    print(f"{name:15s} POISE {m['poise']:.3f}  acc {m['acc']:.3f}  EF1 {m['process']:.3f}  "
          f"budget {m['budget']:.3f}  regret {m['regret']:+.3f}  cost {m['cost']:.2f}", flush=True)

ablation = {}
for name in PRIVILEGED:
    res = {}
    for mode in ("shuffled", "uniform"):
        vals = []
        for seed in (range(5) if mode == "shuffled" else [0]):
            rng = np.random.default_rng(100 + seed)
            preds = []
            for ex in examples:
                nodes = ex["hidden_evidence_graph"]["nodes"]
                if mode == "uniform":
                    g = {n["id"]: 0.5 for n in nodes}
                else:
                    vals_g = [n["information_gain"] for n in nodes]
                    perm = rng.permutation(len(vals_g))
                    g = {n["id"]: vals_g[j] for n, j in zip(nodes, perm)}
                preds.append(run_baseline(name, ex, rng, gains=g))
            vals.append(evaluate(preds, examples)["poise"])
        res[mode] = float(np.mean(vals))
    res["true"] = metrics[name]["poise"]
    ablation[name] = res
    print(f"ablation {name:12s} true {res['true']:.3f}  shuffled {res['shuffled']:.3f}  uniform {res['uniform']:.3f}")

(out / "metrics.json").write_text(json.dumps({"split": args.split, "metrics": metrics,
                                              "privilege_ablation": ablation}, indent=1))
