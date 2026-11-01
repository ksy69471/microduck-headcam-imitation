"""Play a joint trajectory in MuJoCo and report physics feasibility metrics."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.mujoco.joints import load_microduck_model  # noqa: E402


def load_sequence(seq_dir: Path) -> dict:
    joints = np.load(seq_dir / "joint_positions.npz", allow_pickle=True)
    roots = np.load(seq_dir / "root_poses.npz", allow_pickle=True)
    meta = json.loads((seq_dir / "metadata.json").read_text())
    return {
        "joint_names": [str(x) for x in joints["joint_names"].tolist()],
        "q": joints["joint_positions"].astype(np.float64),
        "times": joints["times"].astype(np.float64),
        "root_pos": roots["root_pos"].astype(np.float64),
        "root_quat": roots["root_quat_wxyz"].astype(np.float64),
        "fps": float(meta["fps"]),
        "meta": meta,
    }


def set_state(model, data, md, q_servos, root_pos, root_quat, joint_names):
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    # freejoint: pos(3) + quat(4)
    data.qpos[0:3] = root_pos
    data.qpos[3:7] = root_quat
    name_to_q = {n: float(q_servos[i]) for i, n in enumerate(joint_names)}
    for j in md.servo_joints:
        data.qpos[j.qposadr] = name_to_q[j.name]
        if j.actuator_id is not None:
            data.ctrl[j.actuator_id] = name_to_q[j.name]


def evaluate_trajectory(
    seq_dir: Path,
    hold_seconds: float = 0.0,
    settle_seconds: float = 1.0,
    xml_path: Path | None = None,
) -> dict:
    md = load_microduck_model(xml_path)
    model = md.model
    data = mujoco.MjData(model)
    seq = load_sequence(seq_dir)
    joint_names = seq["joint_names"]
    assert list(md.joint_names) == joint_names, (
        f"joint order mismatch:\nMD={md.joint_names}\nSEQ={joint_names}"
    )

    trunk_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    n = seq["q"].shape[0]
    tilt = []
    height = []
    joint_limit_hits = 0
    for t in range(n):
        set_state(model, data, md, seq["q"][t], seq["root_pos"][t], seq["root_quat"][t], joint_names)
        mujoco.mj_forward(model, data)
        # orientation tilt from world up via trunk xmat
        # xmat is 3x3 row-major; local z axis is columns... use body xquat
        w, x, y, z = data.xquat[trunk_id]
        # rotate world up into body? simpler: trunk world z component of local up
        up_z = data.xmat[trunk_id].reshape(3, 3)[2, 2]
        tilt.append(float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0)))))
        height.append(float(data.xpos[trunk_id][2]))
        for j in md.servo_joints:
            lo, hi = j.range
            q = data.qpos[j.qposadr]
            if q < lo - 1e-4 or q > hi + 1e-4:
                joint_limit_hits += 1

    # Physics rollout from first frame with position ctrl tracking
    set_state(model, data, md, seq["q"][0], seq["root_pos"][0], seq["root_quat"][0], joint_names)
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep
    steps = int(round(settle_seconds / dt))
    fell = False
    min_z = float(data.xpos[trunk_id][2])
    max_tilt = 0.0
    tracking_err = []
    frame_idx = 0
    frame_dt = 1.0 / seq["fps"]
    next_frame_t = 0.0
    for s in range(steps):
        t = s * dt
        while frame_idx + 1 < n and t >= (frame_idx + 1) * frame_dt:
            frame_idx += 1
        q_ref = seq["q"][frame_idx]
        for j in md.servo_joints:
            data.ctrl[j.actuator_id] = float(q_ref[list(md.joint_names).index(j.name)])
        mujoco.mj_step(model, data)
        z = float(data.xpos[trunk_id][2])
        min_z = min(min_z, z)
        up_z = data.xmat[trunk_id].reshape(3, 3)[2, 2]
        tilt_deg = float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))
        max_tilt = max(max_tilt, tilt_deg)
        if z < 0.05 or tilt_deg > 60.0:
            fell = True
        # servo tracking error
        err = []
        for j in md.servo_joints:
            qi = list(md.joint_names).index(j.name)
            err.append(abs(float(data.qpos[j.qposadr]) - float(q_ref[qi])))
        tracking_err.append(float(np.mean(err)))

    report = {
        "sequence": str(seq_dir),
        "xml": str(md.xml_path),
        "n_frames": n,
        "kinematic_tilt_deg_mean": float(np.mean(tilt)),
        "kinematic_tilt_deg_max": float(np.max(tilt)),
        "kinematic_height_mean": float(np.mean(height)),
        "joint_limit_hits": int(joint_limit_hits),
        "physics_settle_s": settle_seconds,
        "physics_fell": bool(fell),
        "physics_min_trunk_z": min_z,
        "physics_max_tilt_deg": max_tilt,
        "physics_mean_tracking_abs_rad": float(np.mean(tracking_err)) if tracking_err else None,
        "note": (
            "Kinematic metrics force qpos each frame (no dynamics). "
            "Physics metrics track ctrl toward the trajectory with XML position actuators."
        ),
    }
    out = seq_dir / "mujoco_feasibility.json"
    out.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq", type=Path, required=True)
    p.add_argument("--settle", type=float, default=2.0)
    args = p.parse_args()
    report = evaluate_trajectory(args.seq, settle_seconds=args.settle)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
