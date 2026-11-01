"""Capture official alpha_walking.onnx under MuJoCo gravity (+BAM).

Produces dataset-compatible teacher videos + joint/root NPZ so the virtual
screen / V2M path can use *real policy walking*, not Blender kinematic march.

Obs layout (61-D) matches duck-control/obs.rs and mjlab symmetry.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.mujoco.stand_policy_demo import (  # noqa: E402
    DEFAULT_POSE,
    RL_ROOT,
    load_scene_with_bam,
    projected_gravity,
)

# Relative camera framing (world offset from trunk) — natural follow shots.
CAM_PRESETS = {
    "front": {"offset": np.array([0.72, 0.06, 0.04]), "look_z": 0.02, "azimuth_hint": None},
    "side": {"offset": np.array([0.08, 0.78, 0.06]), "look_z": 0.02, "azimuth_hint": None},
    "oblique": {"offset": np.array([0.55, -0.45, 0.10]), "look_z": 0.02, "azimuth_hint": None},
}


def build_command(vx: float, vy: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    """13-D command block: twist(3) + head(4) + body(6)."""
    cmd = np.zeros(13, dtype=np.float32)
    cmd[0:3] = [vx, vy, yaw]
    # head / body stay nominal zero (obs.rs)
    return cmd


def build_obs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    trunk: int,
    qpos_idx: np.ndarray,
    dof_idx: np.ndarray,
    last_action: np.ndarray,
    command: np.ndarray,
) -> np.ndarray:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
    if sid >= 0:
        adr = model.sensor_adr[sid]
        dim = model.sensor_dim[sid]
        ang_vel = data.sensordata[adr : adr + dim].astype(np.float32)
    else:
        ang_vel = data.qvel[3:6].astype(np.float32)
    grav = projected_gravity(data, trunk)
    jpos = (data.qpos[qpos_idx] - DEFAULT_POSE).astype(np.float32)
    jvel = data.qvel[dof_idx].astype(np.float32)
    return np.concatenate([ang_vel, grav, jpos, jvel, last_action, command]).astype(np.float32)


def look_camera(cam: mujoco.MjvCamera, trunk_pos: np.ndarray, offset: np.ndarray, look_z: float) -> None:
    """Free camera that follows trunk with a fixed relative offset (natural FOV)."""
    look = trunk_pos + np.array([0.0, 0.0, look_z])
    eye = trunk_pos + offset
    cam.lookat[:] = look
    cam.distance = float(np.linalg.norm(eye - look))
    # Spherical from look→eye.
    d = eye - look
    xy = float(np.hypot(d[0], d[1]))
    cam.azimuth = float(np.degrees(np.arctan2(d[1], d[0])))
    cam.elevation = float(np.degrees(np.arctan2(d[2], max(xy, 1e-6))))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--onnx",
        type=Path,
        default=REPO_ROOT / "work/local/scratch/policies/alpha_walking.onnx",
    )
    p.add_argument("--out", type=Path, required=True, help="Sequence dir (front.mp4 + npz)")
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument(
        "--vx",
        type=float,
        default=0.32,
        help="Forward twist command [m/s] (alpha_walking needs ~>=0.25 to leave stand deadzone)",
    )
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--yaw", type=float, default=0.0, help="Yaw rate command [rad/s]")
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--policy-hz", type=float, default=50.0)
    p.add_argument("--cameras", default="front,side,oblique")
    p.add_argument("--warmup-s", type=float, default=0.5, help="Stand still before walking")
    args = p.parse_args()

    xml = RL_ROOT / "src/mjlab_microduck/robot/microduck/scene_walk.xml"
    model, data, bam_ctrl, act_names = load_scene_with_bam(xml)
    assert abs(model.opt.gravity[2] + 9.81) < 1e-6

    joint_ids = np.array([model.actuator(n).trnid[0] for n in act_names])
    qpos_idx = model.jnt_qposadr[joint_ids]
    dof_idx = model.jnt_dofadr[joint_ids]
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")

    data.qpos[:] = 0
    data.qpos[0:3] = [0, 0, 0.12]
    data.qpos[3:7] = [1, 0, 0, 0]
    for i, adr in enumerate(qpos_idx):
        data.qpos[adr] = DEFAULT_POSE[i]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    bam_ctrl.q_target[:] = DEFAULT_POSE

    sess = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    in_dim = int(sess.get_inputs()[0].shape[1])
    assert in_dim == 61, f"unexpected obs dim {in_dim}"

    last_action = np.zeros(14, dtype=np.float32)
    cmd_walk = build_command(args.vx, args.vy, args.yaw)
    cmd_stand = build_command(0.0, 0.0, 0.0)

    dt = float(model.opt.timestep)
    policy_steps = max(1, int(round(1.0 / args.policy_hz / dt)))
    video_steps = max(1, int(round(1.0 / args.fps / dt)))
    n_frames = int(round(args.seconds * args.fps))
    warmup_frames = int(round(args.warmup_s * args.fps))

    wanted = [c.strip() for c in args.cameras.split(",") if c.strip()]
    for c in wanted:
        if c not in CAM_PRESETS:
            raise SystemExit(f"unknown camera {c}; choose from {list(CAM_PRESETS)}")

    renderer = mujoco.Renderer(model, height=256, width=256)
    cams = {name: mujoco.MjvCamera() for name in wanted}
    for name, cam in cams.items():
        mujoco.mjv_defaultCamera(cam)

    frame_bufs = {name: [] for name in wanted}
    q_log, root_pos_log, root_quat_log, times = [], [], [], []
    action_log, q_target_log = [], []
    zs, tilts, xy = [], [], []

    # Sim loop at physics rate; policy every policy_steps; record every video_steps.
    step_i = 0
    recorded = 0
    warmup_physics = int(round(args.warmup_s / dt))

    while recorded < n_frames:
        walking = step_i >= warmup_physics
        command = cmd_walk if walking else cmd_stand

        if step_i % policy_steps == 0:
            obs = build_obs(model, data, trunk, qpos_idx, dof_idx, last_action, command)
            action = sess.run([out_name], {in_name: obs[None]})[0][0].astype(np.float32)
            last_action = action
            bam_ctrl.q_target[:] = DEFAULT_POSE + action

        bam_ctrl.update()
        mujoco.mj_step(model, data)
        step_i += 1

        if step_i % video_steps != 0:
            continue
        if step_i <= warmup_physics:
            continue  # skip warmup frames from teacher video

        trunk_pos = data.xpos[trunk].copy()
        for name in wanted:
            look_camera(cams[name], trunk_pos, CAM_PRESETS[name]["offset"], CAM_PRESETS[name]["look_z"])
            renderer.update_scene(data, camera=cams[name])
            frame_bufs[name].append(renderer.render().copy())

        q_log.append(data.qpos[qpos_idx].copy())
        root_pos_log.append(data.qpos[0:3].copy())
        root_quat_log.append(data.qpos[3:7].copy())
        action_log.append(last_action.copy())
        q_target_log.append(bam_ctrl.q_target.copy())
        times.append(recorded / args.fps)
        zs.append(float(trunk_pos[2]))
        up_z = float(data.xmat[trunk].reshape(3, 3)[2, 2])
        tilts.append(float(np.degrees(np.arccos(np.clip(up_z, -1, 1)))))
        xy.append(trunk_pos[:2].copy())
        recorded += 1

    renderer.close()
    args.out.mkdir(parents=True, exist_ok=True)

    q = np.stack(q_log, axis=0)
    root_pos = np.stack(root_pos_log, axis=0)
    root_quat = np.stack(root_quat_log, axis=0)
    t = np.asarray(times, dtype=np.float64)
    xy_arr = np.stack(xy, axis=0)
    travel = float(np.linalg.norm(xy_arr[-1] - xy_arr[0]))

    np.savez_compressed(
        args.out / "joint_positions.npz",
        times=t,
        joint_positions=q,
        joint_names=np.array(act_names),
        fps=np.array([args.fps]),
    )
    np.savez_compressed(
        args.out / "root_poses.npz",
        times=t,
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
    )
    # Teacher policy outputs (DEFAULT_POSE + action = BAM target). Needed for tracking RL / BC.
    np.savez_compressed(
        args.out / "teacher_actions.npz",
        times=t,
        actions=np.stack(action_log, axis=0).astype(np.float32),
        q_targets=np.stack(q_target_log, axis=0).astype(np.float64),
        joint_names=np.array(act_names),
        action_space="default_pose_residual",
    )

    cam_meta = {"cameras": {}, "camera_mode": "natural_follow_policy"}
    for name in wanted:
        mp4 = args.out / f"{name}.mp4"
        imageio.mimsave(mp4, frame_bufs[name], fps=args.fps)
        cam_meta["cameras"][name] = {
            "file": str(mp4),
            "offset_from_trunk": CAM_PRESETS[name]["offset"].tolist(),
            "n_frames": len(frame_bufs[name]),
            "fps": args.fps,
        }

    (args.out / "camera_parameters.json").write_text(json.dumps(cam_meta, indent=2))
    meta = {
        "motion_id": "POL-WALK",
        "source": "mujoco_alpha_walking_onnx_bam",
        "onnx": str(args.onnx),
        "fps": args.fps,
        "policy_hz": args.policy_hz,
        "duration_s": args.seconds,
        "warmup_s": args.warmup_s,
        "command_twist": {"vx": args.vx, "vy": args.vy, "yaw": args.yaw},
        "n_frames": int(q.shape[0]),
        "joint_names": act_names,
        "gravity": model.opt.gravity.tolist(),
        "travel_xy_m": travel,
        "min_z": float(np.min(zs)),
        "max_tilt_deg": float(np.max(tilts)),
        "fell": bool(np.min(zs) < 0.05 or np.max(tilts) > 60),
        "contact_note": "Physics contacts from MuJoCo+BAM; teacher videos are policy rollouts.",
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
