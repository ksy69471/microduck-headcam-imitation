#!/usr/bin/env python3
"""Short-horizon grid search over Tracking RL reward scales (CPU; ~few minutes per cell).

Isolates reward design from V2M error by training on GT-joint demos (POL-WALK-EGO) with
``--root-cond`` and ``--no-onnx-teacher``. Each cell trains for ``--time-budget-s`` seconds,
then is ranked by travel × success with a joint-MAE penalty.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
OUT_ROOT = REPO / "work/local/runs/ego/tracking_grid"
MOTIONS = REPO / "datasets/motions/POL-WALK-EGO"


def cells() -> list[dict]:
    """Compact grid focused on the standing-local-optimum hypothesis."""
    grid = []
    for joint_sigma, travel, alive, upright in itertools.product(
        (0.5, 0.05, 0.02),
        (0.0, 1.5),
        (0.1, 0.0),
        (0.8, 0.4),
    ):
        grid.append({
            "joint_sigma": joint_sigma,
            "joint": 1.2 if joint_sigma >= 0.05 else 2.0,
            "travel": travel,
            "alive": alive,
            "upright": upright,
            "root_xy": 0.6,
            "heading": 0.4,
            "xy_sigma": 0.15,
        })
    # Extra: tighter joint + strong travel (most promising a-priori)
    grid.append({
        "joint_sigma": 0.02, "joint": 2.0, "travel": 2.0, "alive": 0.0, "upright": 0.3,
        "root_xy": 0.9, "heading": 0.4, "xy_sigma": 0.15,
    })
    # Ego baseline used in the report
    grid.append({
        "joint_sigma": 0.5, "joint": 1.2, "travel": 0.0, "alive": 0.1, "upright": 0.8,
        "root_xy": 0.6, "heading": 0.4, "xy_sigma": 0.15, "tag": "ego_baseline",
    })
    # Deduplicate
    seen, out = set(), []
    for c in grid:
        key = tuple(sorted((k, round(float(v), 4)) for k, v in c.items() if k != "tag"))
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def cell_name(c: dict) -> str:
    if c.get("tag"):
        return c["tag"]
    return (
        f"js{c['joint_sigma']:g}_j{c['joint']:g}_tr{c['travel']:g}"
        f"_al{c['alive']:g}_up{c['upright']:g}_xy{c['root_xy']:g}"
    )


def last_eval(metrics: Path) -> dict | None:
    if not metrics.exists():
        return None
    best = None
    for line in metrics.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "eval" in row:
            best = {"update": row["update"], "elapsed_s": row["elapsed_s"], **row["eval"]}
    return best


def run_cell(spec: dict) -> dict:
    name = cell_name(spec["cell"])
    out = Path(spec["out_root"]) / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "cell.json").write_text(json.dumps(spec["cell"], indent=2))
    c = spec["cell"]
    cmd = [
        PY, "training/train_tracking_policy.py",
        "--motions", str(MOTIONS),
        "--clip-names", str(MOTIONS / "clips_train.json"),
        "--eval-clip-names", str(MOTIONS / "clips_test.json"),
        "--out", str(out),
        "--device", "cpu",
        "--no-onnx-teacher",
        "--root-cond",
        "--num-envs", str(spec["num_envs"]),
        "--rollout-steps", "512",
        "--updates", "200000",
        "--time-budget-s", str(spec["time_budget_s"]),
        "--eval-every", str(spec["eval_every"]),
        "--ckpt-every", "99999",
        "--hidden", "256",
        "--best-metric", "travel_x_succ",
        "--joint-weight", str(c["joint"]),
        "--joint-sigma", str(c["joint_sigma"]),
        "--travel-weight", str(c["travel"]),
        "--alive-weight", str(c["alive"]),
        "--upright-weight", str(c["upright"]),
        "--root-xy-weight", str(c["root_xy"]),
        "--heading-weight", str(c["heading"]),
        "--xy-sigma", str(c["xy_sigma"]),
        "--seed", "0",
    ]
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "MUJOCO_GL": "egl"}
    t0 = time.time()
    log = out / "train.log"
    with log.open("w") as f:
        proc = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    ev = last_eval(out / "metrics.jsonl") or {}
    row = {
        "name": name,
        **{k: c[k] for k in ("joint_sigma", "joint", "travel", "alive", "upright", "root_xy", "heading", "xy_sigma")},
        "exit": proc.returncode,
        "wall_s": round(time.time() - t0, 1),
        "update": ev.get("update"),
        "success": ev.get("success_rate"),
        "fall": ev.get("fall_rate"),
        "joint_mae": ev.get("joint_mae"),
        "travel_m": ev.get("travel_xy_m"),
        "mean_len": ev.get("mean_len"),
        "xy_err": ev.get("xy_err_m"),
    }
    # Prefer configs that travel while staying upright and tracking joints.
    succ = float(row["success"] or 0.0)
    travel = float(row["travel_m"] or 0.0)
    mae = float(row["joint_mae"] or 1.0)
    row["score"] = round(travel * succ - 0.5 * mae, 4)
    (out / "result.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row), flush=True)
    return row


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=OUT_ROOT)
    p.add_argument("--time-budget-s", type=float, default=210.0, help="~3.5 min per cell")
    p.add_argument("--workers", type=int, default=3)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--limit", type=int, default=0, help="run only first N cells (0=all)")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    grid = cells()[: args.limit] if args.limit else cells()
    (args.out / "grid.json").write_text(json.dumps(grid, indent=2))
    specs = [{"cell": c, "out_root": str(args.out), "time_budget_s": args.time_budget_s,
              "num_envs": args.num_envs, "eval_every": args.eval_every} for c in grid]
    print(f"grid size={len(specs)} workers={args.workers} budget={args.time_budget_s}s", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_cell, s) for s in specs]
        for fut in as_completed(futs):
            rows.append(fut.result())
    rows.sort(key=lambda r: (-(r.get("score") or -1e9), -(r.get("travel_m") or 0), r.get("joint_mae") or 9))
    (args.out / "summary.json").write_text(json.dumps(rows, indent=2))
    with (args.out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("\n=== top 8 ===", flush=True)
    for r in rows[:8]:
        print(
            f"{r['name']:40s} score={r['score']:+.3f} travel={r['travel_m']} "
            f"mae={r['joint_mae']} succ={r['success']} fall={r['fall']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
