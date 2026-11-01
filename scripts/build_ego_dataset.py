#!/usr/bin/env python3
"""Build POL-WALK-EGO: MuJoCo-rendered walking clips + Condition C observations per visual variant.

Layout per sequence::

    sequence_XXXX/
      front_<variant>.mp4, camera_<variant>.npz        (teacher video + true camera path)
      observations_<variant>/robot_camera_obs.{npz,mp4} (Condition C: screen seen by head camera)
      joint_positions.npz, root_poses.npz, contacts.npz, commands.npz, metadata.json

A sequence-level split (train/val/test) is written once to ``split.json`` so every
V2M variant and every tracking evaluation uses the same held-out motions.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
VARIANTS = ("follow_floor", "hand_floor", "hand_plain")


def build_one(seq: Path, seed: int, variants: tuple[str, ...] = VARIANTS) -> str:
    env = {**os.environ, "MUJOCO_GL": "egl"}
    if not (seq / "metadata.json").exists():
        subprocess.run(
            [PY, "simulation/mujoco/capture_ego.py", "--out", str(seq), "--seed", str(seed)],
            cwd=REPO, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    for v in variants:
        obs_dir = seq / f"observations_{v}"
        if (obs_dir / "robot_camera_obs.npz").exists():
            continue
        subprocess.run(
            [PY, "observation/virtual_screen.py", "--seq", str(seq), "--video", str(seq / f"front_{v}.mp4"),
             "--out", str(obs_dir), "--no-world"],
            cwd=REPO, env=env, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    return seq.name


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO / "datasets/motions/POL-WALK-EGO")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--seed-base", type=int, default=1000)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--split-seed", type=int, default=0)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--extend", type=Path, default=None,
                   help="train-only extension: link every sequence of this dataset, add --n new train sequences "
                        "(numbered after it), keep its val/test split unchanged")
    args = p.parse_args()
    if args.extend is not None:
        extend(args)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    seqs = [args.out / f"sequence_{i + 1:04d}" for i in range(args.n)]
    with ThreadPoolExecutor(args.workers) as ex:
        for i, name in enumerate(ex.map(build_one, seqs, [args.seed_base + i for i in range(args.n)])):
            if (i + 1) % 10 == 0:
                print(f"built {i + 1}/{args.n} ({name})", flush=True)

    split_path = args.out / "split.json"
    if not split_path.exists():
        rng = np.random.default_rng(args.split_seed)
        idx = rng.permutation(args.n)
        n_tr, n_va = int(0.7 * args.n), int(0.15 * args.n)
        split = {
            "train": sorted(seqs[i].name for i in idx[:n_tr]),
            "val": sorted(seqs[i].name for i in idx[n_tr : n_tr + n_va]),
            "test": sorted(seqs[i].name for i in idx[n_tr + n_va :]),
            "note": "sequence-level split; all visual variants of one motion share a split",
        }
        split_path.write_text(json.dumps(split, indent=2))

    metas = [json.loads((s / "metadata.json").read_text()) for s in seqs]
    stats = {
        "n": len(metas),
        "travel_m_mean": float(np.mean([m["travel_m"] for m in metas])),
        "path_len_m_mean": float(np.mean([m["path_len_m"] for m in metas])),
        "max_tilt_deg_max": float(np.max([m["max_tilt_deg"] for m in metas])),
        "segments_mean": float(np.mean([len(m["schedule"]) for m in metas])),
        "variants": list(VARIANTS),
    }
    (args.out / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def extend(args) -> None:
    base = json.loads((args.extend / "split.json").read_text())
    old = sorted(args.extend.glob("sequence_*"))
    args.out.mkdir(parents=True, exist_ok=True)
    for s in old:
        link = args.out / s.name
        if not link.exists():
            link.symlink_to(s.resolve())
    first = int(old[-1].name.split("_")[1]) + 1
    seqs = [args.out / f"sequence_{first + i:04d}" for i in range(args.n)]
    variants = tuple(args.variants)
    with ThreadPoolExecutor(args.workers) as ex:
        for i, name in enumerate(ex.map(lambda s, sd: build_one(s, sd, variants), seqs,
                                        [args.seed_base + i for i in range(args.n)])):
            if (i + 1) % 20 == 0:
                print(f"built {i + 1}/{args.n} ({name})", flush=True)
    split = {"train": sorted(base["train"] + [s.name for s in seqs]), "val": base["val"], "test": base["test"],
             "note": f"train-only extension of {args.extend.name}: +{args.n} new train sequences ({','.join(variants)})"}
    (args.out / "split.json").write_text(json.dumps(split, indent=2))
    print(json.dumps({k: len(v) for k, v in split.items() if k != "note"}))


if __name__ == "__main__":
    main()
