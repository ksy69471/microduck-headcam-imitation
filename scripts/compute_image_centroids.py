#!/usr/bin/env python3
"""Per-frame robot silhouette centroid / area in the demo video (segmentation of the demo render).

Writes ``image_centroid_<variant>.npz`` (uv (T,2), area (T,), proj_trunk (T,2)) into every sequence.
``proj_trunk`` (pinhole projection of the trunk subtree CoM) is only a sanity reference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from simulation.mujoco.capture_ego import build_render_model  # noqa: E402
from simulation.mujoco.image_centroid import mask_stats, project_point, robot_mask, set_free_camera  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=REPO / "datasets/motions/POL-WALK-EGO")
    p.add_argument("--variant", default="hand_floor")
    p.add_argument("--res", type=int, default=128)
    p.add_argument("--skip-existing", action="store_true")
    args = p.parse_args()
    errs = []
    for seq in sorted(args.data.glob("sequence_*")):
        if seq.is_symlink() or (args.skip_existing and (seq / f"image_centroid_{args.variant}.npz").exists()):
            continue
        meta = json.loads((seq / "metadata.json").read_text())
        model = build_render_model(meta["floor_style"], meta["landmarks"])
        data = mujoco.MjData(model)
        trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        r = mujoco.Renderer(model, height=args.res, width=args.res)
        r.enable_segmentation_rendering()
        cam_npz = np.load(seq / f"camera_{args.variant}.npz")
        fovy = float(np.asarray(cam_npz["fovy_deg"]).reshape(-1)[0])
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        qpos = np.load(seq / "qpos_full.npz")["qpos"]
        uv, area, proj = [], [], []
        for t, q in enumerate(qpos):
            data.qpos[:] = q
            mujoco.mj_forward(model, data)
            look, aed = cam_npz["lookat"][t], cam_npz["azimuth_elevation_distance"][t]
            set_free_camera(cam, look, aed)
            r.update_scene(data, camera=cam)
            u, v, a = mask_stats(robot_mask(r.render(), model))
            uv.append((u, v))
            area.append(a)
            proj.append(project_point(data.subtree_com[trunk], look, aed, fovy))
        r.close()
        uv, area, proj = np.asarray(uv), np.asarray(area), np.asarray(proj)
        np.savez_compressed(seq / f"image_centroid_{args.variant}.npz", uv=uv, area=area, proj_trunk=proj, fovy_deg=fovy)
        errs.append(float(np.nanmean(np.linalg.norm(uv - proj, axis=1))))
        print(seq.name, f"area={area.mean():.3f} empty={int(np.sum(area == 0))} |uv-proj|={errs[-1]:.3f}", flush=True)
    if errs:
        print(f"mean |silhouette centroid - projected CoM| = {np.mean(errs):.3f} (image widths)")


if __name__ == "__main__":
    main()
