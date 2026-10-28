# Builds the train, dev and test splits and reports how many generated worlds were kept.
from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path

import numpy as np

from . import FAMILIES
from .core import CertificationError, analyse, export_item

SPLITS = {"train": "v1", "dev": "v2", "test": "v3"}
BASE_SEED = {"train": 1001, "dev": 2002, "test": 3003}
FAM_OFFSET = {"retrieval": 11, "spatial": 22, "procedural": 33, "abstention": 44}


def build_split(split: str, per_family: int):
    variant = SPLITS[split]
    items, report = [], {}
    for fam, mod in FAMILIES.items():
        seed = BASE_SEED[split] * 100 + FAM_OFFSET[fam]
        rng = np.random.default_rng(seed)
        reasons = collections.Counter()
        kept, draws = 0, 0
        while kept < per_family:
            draws += 1
            if draws > per_family * 50:
                raise RuntimeError(f"{fam}/{split}: yield too low")
            try:
                w = mod.generate(rng, split, variant, kept)
                mod.round_trip(w)
                a = analyse(w, mod.determine)
            except CertificationError as e:
                reasons[str(e).split(":")[0][:50]] += 1
                continue
            ex = export_item(w, a, f"{fam}_{split}_{kept:05d}", seed, rng)
            items.append(ex)
            kept += 1
        report[fam] = {"kept": kept, "draws": draws, "yield": round(kept / draws, 3),
                       "rejections": dict(reasons.most_common())}
    return items, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--dev", type=int, default=400)
    ap.add_argument("--test", type=int, default=500)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    full_report = {}
    for split in ("train", "dev", "test"):
        t = time.time()
        items, rep = build_split(split, getattr(args, split))
        (out / f"{split}.json").write_text(json.dumps(items, indent=1))
        full_report[split] = rep
        print(f"{split}: {len(items)} items in {time.time() - t:.1f}s  " +
              ", ".join(f"{f} yield {r['yield']}" for f, r in rep.items()), flush=True)
    (out / "certification_report.json").write_text(json.dumps(full_report, indent=2))


if __name__ == "__main__":
    main()
