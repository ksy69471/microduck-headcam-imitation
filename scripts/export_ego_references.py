#!/usr/bin/env python3
"""Export V2M-Ego predictions as a tracking reference library.

--root est   : root xy / yaw integrated from predicted per-frame root motion (video only), z predicted
--root gt    : privileged GT root (upper bound; joints still from V2M)
--root none  : no root information (origin, identity heading); z predicted

The world frame of every clip is the robot's pose at the first observed frame, so
integration starts from the identity — no GT is needed for the start pose.
"""

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

from simulation.mujoco.image_centroid import camera_from_view  # noqa: E402
from training.ego_dataset import EgoDataset, integrate, se2_inv, se2_mul  # noqa: E402
from training.train_v2m_ego import build_model, predict, sequence_metrics  # noqa: E402


def yaw_quat(yaw: np.ndarray) -> np.ndarray:
    return np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], -1)


def write_image_target(path: Path, mode: str, pred: dict, tg: dict, ref_pos0, ref_quat0, n: int, fovy: float) -> None:
    """Demo camera path + demo silhouette, with the camera placed relative to the reference start pose.

    Camera path = start pose from the relative-pose head (C_0 = P_0 ∘ rel_0^{-1}) then integrated camera
    motion, so it never uses the estimated (or GT) root trajectory; height / elevation are per frame.
    """
    if mode == "est":
        view = pred["view"][:n].astype(np.float64)
        rel0 = pred["rel"][0].astype(np.float64)
        cam_traj = integrate(pred["cam_vel"][:n][:, [0, 1, 3]].astype(np.float64), tg["fps"])
    else:
        view = tg["view"][:n].astype(np.float64)
        rel0 = tg["rel"][0].astype(np.float64)
        cam_traj = se2_mul(se2_inv(tg["cam_pose"][:1]), tg["cam_pose"][:n]).astype(np.float64)
    p0 = np.array([ref_pos0[0], ref_pos0[1], float(np.arctan2(2 * (ref_quat0[0] * ref_quat0[3] + ref_quat0[1] * ref_quat0[2]),
                                                                1 - 2 * (ref_quat0[2] ** 2 + ref_quat0[3] ** 2)))])
    c0 = se2_mul(p0, se2_inv(rel0))
    cam_pose = se2_mul(np.repeat(c0[None], n, 0), cam_traj)
    lookat, aed = camera_from_view(cam_pose, view[:, 3], view[:, 4])
    np.savez_compressed(path, lookat=lookat, aed=aed, uv=view[:, :2], area=np.exp(view[:, 2]) - 1e-4,
                        fovy_deg=np.array([fovy]), mode=np.array(mode))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=Path, required=True)
    p.add_argument("--data", type=Path, default=REPO / "datasets/motions/POL-WALK-EGO")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--root", choices=["est", "gt", "none"], required=True)
    p.add_argument(
        "--img-target", choices=["none", "est", "gt"], default="none",
        help="also write image_target.npz (demo camera path + silhouette) in the reference frame: "
             "est = from the head-camera observation only (needs a --with-view checkpoint), gt = privileged",
    )
    p.add_argument("--fovy-deg", type=float, default=45.0, help="demo camera field of view (intrinsics assumed known)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = build_model(ck).to(device)
    model.load_state_dict(ck["model"])
    seqs = sorted(s for s in args.data.glob("sequence_*") if s.is_dir())
    ds = EgoDataset(seqs, ck["variant"], ck["source"], ck["image_size"])
    preds = predict(model, ds, device)
    mean_vel = np.asarray(ck["mean_train_root_vel"])
    split = json.loads((args.data / "split.json").read_text())
    names = np.load(seqs[0] / "joint_positions.npz")["joint_names"]

    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    for seq, pred, tg in zip(seqs, preds, ds.targets):
        n = min(len(pred["joint"]), len(tg["joint"]))
        fps = ds.fps
        z = pred["root_z"][:n].astype(np.float64)
        if args.root == "gt":
            roots = np.load(seq / "root_poses.npz")
            root_pos, root_quat = roots["root_pos"][:n], roots["root_quat_wxyz"][:n]
        elif args.root == "est":
            pose = integrate(pred["root_vel"][:n].astype(np.float64), fps)
            root_pos = np.stack([pose[:, 0], pose[:, 1], z], -1)
            root_quat = yaw_quat(pose[:, 2])
        else:
            root_pos = np.stack([np.zeros(n), np.zeros(n), z], -1)
            root_quat = yaw_quat(np.zeros(n))
        dst = args.out / seq.name
        dst.mkdir(exist_ok=True)
        t = np.arange(n) / fps
        np.savez_compressed(dst / "joint_positions.npz", times=t, joint_positions=pred["joint"][:n].astype(np.float64),
                            joint_names=names, fps=np.array([fps]))
        np.savez_compressed(dst / "root_poses.npz", times=t, root_pos=root_pos, root_quat_wxyz=root_quat)
        np.savez_compressed(dst / "contacts_pred.npz", contact_prob=1 / (1 + np.exp(-pred["contact_logits"][:n])))
        if args.img_target != "none":
            write_image_target(dst / "image_target.npz", args.img_target, pred, tg, root_pos[0], root_quat[0], n, args.fovy_deg)
        (dst / "metadata.json").write_text(json.dumps({
            "motion_id": args.out.name, "source": "v2m_ego_predicted", "fps": fps, "n_frames": int(n),
            "root_mode": args.root, "v2m_ckpt": str(args.ckpt), "variant": ck["variant"], "input": ck["source"],
            "demo_sequence": str(seq),
        }, indent=2))
        m = sequence_metrics(pred, tg, fps, mean_vel)
        m["seq"] = seq.name
        m["split"] = next(k for k in ("train", "val", "test") if seq.name in split[k])
        report.append(m)

    def mean_of(rows, k):
        return float(np.mean([r[k] for r in rows])) if rows else float("nan")

    summary = {
        "ckpt": str(args.ckpt), "root": args.root, "variant": ck["variant"], "input": ck["source"],
        **{f"{sp}_{k}": mean_of([r for r in report if r["split"] == sp], k)
           for sp in ("train", "test") for k in ("joint_mae", "ate_m", "final_pos_err_m", "final_yaw_err_deg", "ate_baseline_m")},
        "seqs": report,
    }
    (args.out / "export_report.json").write_text(json.dumps(summary, indent=2))
    for name in ("train", "val", "test"):
        (args.out / f"clips_{name}.json").write_text(json.dumps(split[name]))
    print(json.dumps({k: v for k, v in summary.items() if k != "seqs"}, indent=2))


if __name__ == "__main__":
    main()
