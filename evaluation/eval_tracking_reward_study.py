#!/usr/bin/env python3
"""Evaluate tracking checkpoints against the demo's GT trajectory on a fixed clip split.

For each checkpoint and clip, the deterministic policy is rolled out for the full clip and the
executed trunk path is compared with the demo's GT root path (not with the reward's reference).
Results are cached per (run, checkpoint, split) in <run>/study_eval/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from simulation.mujoco.tracking_env import MotionTrackingEnv, load_motion_clip  # noqa: E402
from training.ppo import ActorCritic  # noqa: E402

DEMO = REPO / "datasets/motions/POL-WALK-EGO"


def load_model(ckpt: Path, obs_dim: int) -> tuple[ActorCritic, int]:
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    hidden = blob["model"]["trunk.0.weight"].shape[0]
    model = ActorCritic(obs_dim, MotionTrackingEnv.ACT_DIM, hidden=hidden)
    model.load_state_dict(blob["model"])
    model.eval()
    return model, int(blob.get("update", -1))


def rollout(env: MotionTrackingEnv, model: ActorCritic, ci: int, demo_xy: np.ndarray, fps: float) -> dict:
    obs = env.reset(clip_idx=ci, start_frac=0.0)
    ts, xy, maes, imgs = [], [], [], []
    done, info = False, {}
    while not done:
        with torch.no_grad():
            a = model.dist(torch.as_tensor(obs, dtype=torch.float32)[None]).mean[0].numpy()
        obs, _r, done, info = env.step(a)
        ts.append(env.phase_t)
        xy.append(info["root_xy"])
        maes.append(info["joint_mae"])
        if np.isfinite(info.get("img_centroid_err", np.nan)):
            imgs.append(info["img_centroid_err"])
    ts, xy = np.asarray(ts), np.asarray(xy)
    tt = np.arange(len(demo_xy)) / fps
    g = np.stack([np.interp(ts, tt, demo_xy[:, 0]), np.interp(ts, tt, demo_xy[:, 1])], -1)
    err = np.linalg.norm(xy - g, axis=1)
    demo_disp = float(np.linalg.norm(demo_xy[-1] - demo_xy[0]))
    exec_disp = float(np.linalg.norm(xy[-1] - demo_xy[0]))
    # Fallen episodes are scored as if the robot stayed where it fell for the rest of the clip.
    t_full = tt[tt > ts[-1]]
    if len(t_full):
        tail = np.linalg.norm(demo_xy[len(tt) - len(t_full):] - xy[-1], axis=1)
        err_full = np.concatenate([err, tail])
    else:
        err_full = err
    return {
        "success": bool(info["success"]),
        "fell": bool(info["fell"]),
        "dur_s": float(ts[-1]),
        "ate_demo_m": float(np.mean(err_full)),
        "final_err_m": float(err_full[-1]),
        "joint_mae": float(np.mean(maes)),
        "demo_disp_m": demo_disp,
        "exec_disp_m": exec_disp,
        "disp_ratio": exec_disp / max(demo_disp, 1e-6),
        "img_err": float(np.mean(imgs)) if imgs else float("nan"),
        "xy": xy[:: 2].tolist(),
        "demo_xy": g[:: 2].tolist(),
    }


def evaluate(run: Path, ckpt: Path, split: str, with_img: bool) -> dict:
    tag = f"{ckpt.stem}_{split}{'_img' if with_img else ''}"
    cache = run / "study_eval" / f"{tag}.json"
    if cache.exists() and cache.stat().st_mtime >= ckpt.stat().st_mtime:
        return json.loads(cache.read_text())
    cfg = json.loads((run / "config.json").read_text())
    names = json.loads((DEMO / f"clips_{split}.json").read_text())
    clips = [load_motion_clip(DEMO / n) for n in names]
    for c in clips:
        c.teacher_actions = None
    rw = dict(cfg["reward_weights"])
    rw["img"] = max(float(rw.get("img", 0.0)), 1e-6) if with_img else 0.0
    env = MotionTrackingEnv(clips, seed=0, root_cond=cfg.get("root_cond", True), xy_sigma=cfg.get("xy_sigma", 0.15),
                            joint_sigma=cfg.get("joint_sigma", 0.5), reward_weights=rw,
                            img_sigma=cfg.get("img_sigma", 0.15), img_every=1)
    model, upd = load_model(ckpt, cfg["obs_dim"])
    rows = []
    for ci, n in enumerate(names):
        demo_xy = np.load(DEMO / n / "root_poses.npz")["root_pos"][:, :2]
        r = rollout(env, model, ci, demo_xy, clips[ci].fps)
        r["clip"] = n
        rows.append(r)
    keys = ["success", "fell", "ate_demo_m", "final_err_m", "joint_mae", "disp_ratio", "img_err"]
    summ = {k: float(np.nanmean([float(r[k]) for r in rows])) for k in keys}
    out = {"run": run.name, "ckpt": ckpt.name, "update": upd, "split": split, "summary": summ, "clips": rows}
    cache.parent.mkdir(exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument("--split", default="val")
    p.add_argument("--ckpts", default="all", help="all | best | last | comma list of file names")
    p.add_argument("--img", action="store_true", help="also measure image-centroid error (slower)")
    args = p.parse_args()
    for run in args.runs:
        if args.ckpts == "all":
            cks = sorted((run / "checkpoints").glob("update_*.pt"))
        elif args.ckpts in ("best", "last"):
            cks = [run / f"{args.ckpts}.pt"]
        else:
            cks = [run / "checkpoints" / c if (run / "checkpoints" / c).exists() else run / c for c in args.ckpts.split(",")]
        for ck in cks:
            r = evaluate(run, ck, args.split, args.img)
            s = r["summary"]
            print(f"{run.name:28s} {ck.name:18s} upd={r['update']:5d} succ={s['success']:.2f} "
                  f"ate={s['ate_demo_m']:.3f} fin={s['final_err_m']:.3f} mae={s['joint_mae']:.3f} "
                  f"disp={s['disp_ratio']:.2f} img={s['img_err']:.3f}", flush=True)


if __name__ == "__main__":
    main()
