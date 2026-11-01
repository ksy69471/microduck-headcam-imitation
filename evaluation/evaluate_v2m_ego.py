#!/usr/bin/env python3
"""Evaluate V2M-Ego runs on the held-out test split; store per-sequence metrics and predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.ego_dataset import EgoDataset, split_sequences  # noqa: E402
from training.train_root_from_joints import JointsToRoot  # noqa: E402
from training.train_v2m_ego import aggregate, build_model, predict, sequence_metrics  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", type=Path, nargs="+", required=True)
    p.add_argument("--data", type=Path, default=REPO / "datasets/motions/POL-WALK-EGO")
    p.add_argument("--split", default="test")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument(
        "--joints-model", type=Path, default=REPO / "work/local/runs/ego/v2m/joints_only/model.pt",
        help="joints->root control model; also reports root from predicted joints (two-stage)",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seqs = split_sequences(args.data)[args.split]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    j2r = None
    if args.joints_model.exists():
        jk = torch.load(args.joints_model, map_location="cpu", weights_only=False)
        j2r = JointsToRoot(jk["n_in"])
        j2r.load_state_dict(jk["model"])
        j2r.eval()
    table = {}
    for run in args.runs:
        ck = torch.load(run / "best.pt", map_location=device, weights_only=False)
        model = build_model(ck).to(device)
        model.load_state_dict(ck["model"])
        ds = EgoDataset(seqs, ck["variant"], ck["source"], ck["image_size"])
        preds = predict(model, ds, device)
        mean_vel = np.asarray(ck["mean_train_root_vel"])
        rows = [sequence_metrics(pr, tg, ds.fps, mean_vel) | {"seq": s.name} for pr, tg, s in zip(preds, ds.targets, seqs)]
        if j2r is not None:
            for pr, tg, row in zip(preds, ds.targets, rows):
                with torch.no_grad():
                    vel = j2r(torch.as_tensor(pr["joint"], dtype=torch.float32)[None])[0].numpy()
                two = sequence_metrics(pr | {"root_vel": vel}, tg, ds.fps, mean_vel)
                ens = sequence_metrics(pr | {"root_vel": 0.5 * (vel + pr["root_vel"])}, tg, ds.fps, mean_vel)
                pr["root_vel_twostage"] = vel
                row |= {"ate_twostage_m": two["ate_m"], "final_pos_err_twostage_m": two["final_pos_err_m"],
                        "final_yaw_err_twostage_deg": two["final_yaw_err_deg"], "ate_ens_m": ens["ate_m"],
                        "final_pos_err_ens_m": ens["final_pos_err_m"], "final_yaw_err_ens_deg": ens["final_yaw_err_deg"]}
        summ = aggregate([{k: v for k, v in r.items() if k != "seq"} for r in rows])
        table[run.name] = {"variant": ck["variant"], "input": ck["source"], "epoch": ck["epoch"], "summary": summ, "seqs": rows}
        np.savez_compressed(
            args.out.parent / f"v2m_preds_{run.name}.npz",
            **{f"{s.name}__{k}": v for s, pr in zip(seqs, preds) for k, v in pr.items()},
        )
        print(run.name, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items()}), flush=True)
    args.out.write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
