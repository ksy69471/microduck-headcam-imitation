#!/usr/bin/env python3
"""Run a tracking policy on held-out reference clips and compare the executed walk with the demo.

The key metric is not "did the robot follow its reference" but "did the robot walk the
path shown in the demo video": executed trunk trajectory vs the demo's GT root trajectory.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from simulation.mujoco.tracking_env import MotionTrackingEnv, _wrap, _yaw_from_quat, load_motion_clip  # noqa: E402
from training.ppo import ActorCritic  # noqa: E402


def run_clip(env: MotionTrackingEnv, model: ActorCritic, device) -> dict:
    obs = env.reset(clip_idx=0, start_frac=0.0)
    log = {"t": [], "xy": [], "yaw": [], "q": [], "qpos": []}
    done, info = False, {}
    while not done:
        with torch.no_grad():
            a = model.dist(torch.as_tensor(obs, dtype=torch.float32, device=device)[None]).mean[0].cpu().numpy()
        obs, _r, done, info = env.step(a)
        log["t"].append(env.phase_t)
        log["xy"].append(info["root_xy"])
        log["yaw"].append(info["yaw"])
        log["q"].append(env.data.qpos[env.qpos_idx].copy())
        log["qpos"].append(env.data.qpos.copy())
    out = {k: np.asarray(v) for k, v in log.items()}
    out["success"] = bool(info["success"])
    out["fell"] = bool(info["fell"])
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True, help="tracking run dir (config.json + best.pt)")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--refs", type=Path, default=None, help="reference library (default: run config motions)")
    p.add_argument("--demo", type=Path, default=REPO / "datasets/motions/POL-WALK-EGO")
    p.add_argument("--clips", type=Path, default=REPO / "datasets/motions/POL-WALK-EGO/clips_test.json")
    p.add_argument("--label", default=None)
    args = p.parse_args()

    cfg = json.loads((args.run / "config.json").read_text())
    refs = args.refs or Path(cfg["motions"])
    device = torch.device("cpu")
    model = ActorCritic(cfg["obs_dim"], cfg["act_dim"], hidden=512)
    model.load_state_dict(torch.load(args.run / args.ckpt, map_location=device, weights_only=False)["model"])
    model.eval()

    rows, paths, rollouts = [], [], {}
    for name in json.loads(args.clips.read_text()):
        clip = load_motion_clip(refs / name)
        clip.teacher_actions = None
        env = MotionTrackingEnv([clip], root_cond=cfg.get("root_cond", False), seed=0)
        res = run_clip(env, model, device)
        demo = np.load(args.demo / name / "root_poses.npz")
        dq = np.load(args.demo / name / "joint_positions.npz")["joint_positions"]
        fps = clip.fps
        tt = np.arange(len(demo["root_pos"])) / fps
        gx = np.interp(res["t"], tt, demo["root_pos"][:, 0])
        gy = np.interp(res["t"], tt, demo["root_pos"][:, 1])
        gyaw = np.interp(res["t"], tt, np.unwrap(_yaw_from_quat(demo["root_quat_wxyz"])))
        err = np.hypot(res["xy"][:, 0] - gx, res["xy"][:, 1] - gy)
        rx = np.interp(res["t"], tt[: clip.n_frames], clip.root_pos[:, 0])
        ry = np.interp(res["t"], tt[: clip.n_frames], clip.root_pos[:, 1])
        idx = np.clip(np.round(res["t"] * fps).astype(int), 0, len(dq) - 1)
        q_demo = np.stack([dq[i] for i in idx])
        rows.append({
            "clip": name,
            "success": res["success"],
            "fell": res["fell"],
            "duration_s": float(res["t"][-1]),
            "ate_vs_demo_m": float(np.mean(err)),
            "final_err_vs_demo_m": float(err[-1]),
            "final_yaw_err_deg": float(np.degrees(abs(_wrap(res["yaw"][-1] - gyaw[-1])))),
            "ate_vs_ref_m": float(np.mean(np.hypot(res["xy"][:, 0] - rx, res["xy"][:, 1] - ry))),
            "joint_mae_vs_demo": float(np.mean(np.abs(res["q"] - q_demo))),
            "demo_travel_m": float(np.hypot(gx[-1] - gx[0], gy[-1] - gy[0])),
        })
        paths.append({"clip": name, "exec": res["xy"].tolist(), "demo": np.stack([gx, gy], -1).tolist(),
                      "ref": clip.root_pos[:, :2].tolist()})
        frame_t = np.arange(len(tt)) / fps
        pick = np.clip(np.searchsorted(res["t"], frame_t), 0, len(res["t"]) - 1)
        rollouts[name] = res["qpos"][pick]
        rollouts[name + "__valid"] = frame_t <= res["t"][-1] + 0.5 / fps

    keys = [k for k in rows[0] if k != "clip"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    summary["n"] = len(rows)
    label = args.label or args.run.name
    out = {"label": label, "run": str(args.run), "refs": str(refs), "root_cond": cfg.get("root_cond", False),
           "summary": summary, "clips": rows}
    (args.run / "eval_vs_demo.json").write_text(json.dumps(out, indent=2))
    (args.run / "eval_paths.json").write_text(json.dumps(paths))
    np.savez_compressed(args.run / "eval_rollouts.npz", **rollouts)

    k = min(6, len(paths))
    fig, axes = plt.subplots(1, k, figsize=(3 * k, 3.2))
    for ax, pth in zip(np.atleast_1d(axes), paths[:k]):
        d, e, r = np.asarray(pth["demo"]), np.asarray(pth["exec"]), np.asarray(pth["ref"])
        ax.plot(d[:, 0], d[:, 1], "k-", lw=2, label="demo GT")
        ax.plot(r[:, 0], r[:, 1], "--", color="tab:orange", lw=1.2, label="reference")
        ax.plot(e[:, 0], e[:, 1], "-", color="tab:blue", lw=1.5, label="executed")
        ax.set_title(pth["clip"].replace("sequence_", "#"), fontsize=9)
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(fontsize=7)
    fig.suptitle(label)
    fig.tight_layout()
    fig.savefig(args.run / "eval_paths.png", dpi=130)
    print(json.dumps({"label": label, **summary}, indent=2))


if __name__ == "__main__":
    main()
