#!/usr/bin/env python3
"""Paired per-clip comparison of command-level conditions (18 test clips): mean difference, bootstrap 95% CI, wins."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def per_clip(run: Path, metric: str) -> dict[str, float]:
    src = run / ("render_compare.json" if metric.startswith(("iou", "obs_l1")) else "eval_vs_demo.json")
    return {r["clip"]: r[metric] for r in json.loads(src.read_text())["clips"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--pairs", nargs="+", required=True, help="A:B pairs; reports B - A")
    p.add_argument("--metric", default="ate_vs_demo_m")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    rng = np.random.default_rng(0)
    res = []
    for pair in args.pairs:
        a, b = pair.split(":")
        da, db = per_clip(args.dir / a, args.metric), per_clip(args.dir / b, args.metric)
        clips = sorted(set(da) & set(db))
        d = np.array([db[c] - da[c] for c in clips])
        boot = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)])
        row = {"a": a, "b": b, "metric": args.metric, "n": len(d), "mean_a": float(np.mean([da[c] for c in clips])),
               "mean_b": float(np.mean([db[c] for c in clips])), "diff": float(d.mean()),
               "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
               "b_better": int(np.sum(d < 0) if not args.metric.startswith("iou") else np.sum(d > 0))}
        res.append(row)
        print(f"{b} - {a}: {row['diff']:+.4f}  CI95 [{row['ci95'][0]:+.4f}, {row['ci95'][1]:+.4f}]  "
              f"B better on {row['b_better']}/{row['n']}")
    if args.out:
        prev = json.loads(args.out.read_text()) if args.out.exists() else []
        args.out.write_text(json.dumps(prev + res, indent=2))


if __name__ == "__main__":
    main()
