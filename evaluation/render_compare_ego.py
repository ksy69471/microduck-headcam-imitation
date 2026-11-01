#!/usr/bin/env python3
"""Render-and-compare: does the executed walk *look like* the demo from the same camera?

For each held-out clip the executed rollout (eval_rollouts.npz from evaluate_ego_tracking.py)
is re-rendered with the demo's scene (floor style, landmarks) and camera path, then

  * silhouette IoU of the robot (direct render, segmentation)
  * observation L1: both videos are shown on the virtual screen and seen by the head
    camera (Condition C), so screen / lens effects cancel out

A "stand still" rollout (first demo pose held) gives the scale of a non-imitating robot.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from observation.virtual_screen import observe_video_on_screen  # noqa: E402
from simulation.mujoco.capture_ego import IMG, LANDMARK_GROUP, build_render_model  # noqa: E402


def render_clip(model, qposes: np.ndarray, cam_npz, seg: bool) -> np.ndarray:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=IMG, width=IMG)
    if seg:
        renderer.enable_segmentation_rendering()
    opt = mujoco.MjvOption()
    mujoco.mjv_defaultOption(opt)
    opt.geomgroup[LANDMARK_GROUP] = 1
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    aed, lookat = cam_npz["azimuth_elevation_distance"], cam_npz["lookat"]
    out = []
    for t, qpos in enumerate(qposes):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        cam.lookat[:] = lookat[t]
        cam.azimuth, cam.elevation, cam.distance = aed[t]
        renderer.update_scene(data, camera=cam, scene_option=opt)
        img = renderer.render().copy()
        if seg:
            gid, typ = img[..., 0], img[..., 1]
            body = np.where(gid >= 0, model.geom_bodyid[np.clip(gid, 0, model.ngeom - 1)], 0)
            img = (typ == int(mujoco.mjtObj.mjOBJ_GEOM)) & (body > 0)
        out.append(img)
    renderer.close()
    return np.asarray(out)


def through_screen(demo_seq: Path, frames: np.ndarray, fps: float) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = Path(tmp) / "self.mp4"
        imageio.mimsave(mp4, list(frames), fps=fps)
        observe_video_on_screen(demo_seq, mp4, Path(tmp) / "obs", also_world=False)
        return np.load(Path(tmp) / "obs" / "robot_camera_obs.npz")["rgb"]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--demo", type=Path, default=REPO / "datasets/motions/POL-WALK-EGO")
    p.add_argument("--variant", default="hand_floor")
    p.add_argument("--max-clips", type=int, default=18)
    p.add_argument("--video-clip", default=None, help="write a side-by-side mp4 for this clip")
    args = p.parse_args()

    roll = np.load(args.run / "eval_rollouts.npz")
    clips = sorted(k for k in roll.files if not k.endswith("__valid"))[: args.max_clips]
    rows = []
    for name in clips:
        seq = args.demo / name
        meta = json.loads((seq / "metadata.json").read_text())
        fps = float(meta["fps"])
        model = build_render_model(meta["floor_style"], meta["landmarks"])
        cam = np.load(seq / f"camera_{args.variant}.npz")
        demo_q = np.load(seq / "qpos_full.npz")["qpos"]
        exec_q = roll[name][: len(demo_q)]
        still_q = np.repeat(demo_q[:1], len(demo_q), 0)

        m_demo = render_clip(model, demo_q, cam, seg=True)
        scores = {}
        for tag, q in (("exec", exec_q), ("still", still_q)):
            m = render_clip(model, q, cam, seg=True)
            inter = np.logical_and(m, m_demo).sum((1, 2))
            union = np.logical_or(m, m_demo).sum((1, 2))
            scores[f"iou_{tag}"] = float(np.mean(inter / np.maximum(union, 1)))
            obs_self = through_screen(seq, render_clip(model, q, cam, seg=False), fps)
            obs_demo = np.load(seq / f"observations_{args.variant}" / "robot_camera_obs.npz")["rgb"]
            n = min(len(obs_self), len(obs_demo))
            diff = np.abs(obs_self[:n].astype(np.float32) - obs_demo[:n].astype(np.float32)) / 255.0
            scores[f"obs_l1_{tag}"] = float(diff.mean())
            if tag == "exec" and name == (args.video_clip or clips[0]):
                strip = [np.concatenate([obs_demo[t], obs_self[t], (diff[t] * 255 * 3).clip(0, 255).astype(np.uint8)], 1)
                         for t in range(n)]
                imageio.mimsave(args.run / f"render_compare_{name}.mp4", strip, fps=fps)
                for t in (0, n // 2, n - 1):
                    imageio.imwrite(args.run / f"render_compare_{name}_f{t:03d}.png", strip[t])
        rows.append({"clip": name, **scores})
        print(name, {k: round(v, 3) for k, v in scores.items()}, flush=True)

    summary = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "clip"}
    (args.run / "render_compare.json").write_text(json.dumps({"summary": summary, "clips": rows}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
