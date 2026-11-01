"""Headless gravity stand demo with official alpha_stand.onnx + installed BAM API."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import onnxruntime as ort
from bam.model import load_model
from bam.mujoco import MujocoController

REPO_ROOT = Path(__file__).resolve().parents[2]
RL_ROOT = REPO_ROOT / "readonly" / "microduck_rl"

DEFAULT_POSE = np.array(
    [
        0.0,
        -0.0873,
        -0.4579,
        -0.0049,
        0.4530,
        0.3491,
        0.3491,
        0.0,
        0.0,
        0.0,
        0.0873,
        0.4579,
        0.0049,
        -0.4530,
    ],
    dtype=np.float32,
)


def load_scene_with_bam(xml_path: Path, vin: float = 7.4, kp_fw: float = 200.0, spec_hook=None):
    """``spec_hook(spec)`` may add visual-only geoms/materials before compile (must not touch physics)."""
    bam_model = load_model(motor_name="xl330", model="m6")
    bam_model.actuator.kp = kp_fw
    bam_model.actuator.vin = vin
    bam_model.actuator.max_current = None
    kt = bam_model.kt.value
    R = bam_model.R.value
    force_limit = vin * kt / R

    spec = mujoco.MjSpec.from_file(str(xml_path))
    names = []
    for act in spec.actuators:
        tgt = act.target
        tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
        if tgt_name.startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        for joint in spec.joints:
            if joint.name == tgt_name:
                joint.damping = np.zeros((3, 1))
                joint.frictionloss = 0.0
                break
    if spec_hook is not None:
        spec_hook(spec)
    model = spec.compile()
    model.opt.timestep = 0.005
    data = mujoco.MjData(model)
    ctrl = MujocoController(
        bam_model,
        names,
        model,
        data,
        vin_drop_resistance=0.1,
        vin_min=6.0,
    )
    return model, data, ctrl, names


def projected_gravity(data, trunk_id: int) -> np.ndarray:
    R = data.xmat[trunk_id].reshape(3, 3)
    g_world = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    return (R.T @ g_world).astype(np.float32)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=4.0)
    args = p.parse_args()

    xml = RL_ROOT / "src/mjlab_microduck/robot/microduck/scene_walk.xml"
    model, data, bam_ctrl, act_names = load_scene_with_bam(xml)
    assert abs(model.opt.gravity[2] + 9.81) < 1e-6

    joint_qpos_idx = np.array([model.actuator(n).trnid[0] for n in act_names])
    # trnid is joint id; convert to qpos/dof adr
    joint_ids = joint_qpos_idx
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
    last_action = np.zeros(14, dtype=np.float32)
    command = np.zeros(13 if in_dim >= 61 else 3, dtype=np.float32)

    renderer = mujoco.Renderer(model, height=256, width=256)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.distance = 0.55
    camera.azimuth = 90
    camera.elevation = -15

    fps = 30.0
    n = int(args.seconds * fps)
    steps_per = max(1, int(round((1.0 / fps) / model.opt.timestep)))
    frames, zs, tilts = [], [], []

    for _ in range(n):
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
        parts = [ang_vel, grav, jpos, jvel, last_action, command[: 13 if in_dim >= 61 else 3]]
        obs = np.concatenate(parts).astype(np.float32)
        if obs.shape[0] != in_dim:
            fixed = np.zeros(in_dim, dtype=np.float32)
            fixed[: min(in_dim, obs.shape[0])] = obs[:in_dim]
            obs = fixed
        action = sess.run([out_name], {in_name: obs[None]})[0][0].astype(np.float32)
        last_action = action
        bam_ctrl.q_target[:] = DEFAULT_POSE + action
        for _s in range(steps_per):
            bam_ctrl.update()
            mujoco.mj_step(model, data)

        camera.lookat[:] = data.xpos[trunk]
        renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())
        z = float(data.xpos[trunk][2])
        zs.append(z)
        up_z = float(data.xmat[trunk].reshape(3, 3)[2, 2])
        tilts.append(float(np.degrees(np.arccos(np.clip(up_z, -1, 1)))))

    renderer.close()
    args.out.mkdir(parents=True, exist_ok=True)
    mp4 = args.out / "official_stand_gravity.mp4"
    imageio.mimsave(mp4, frames, fps=fps)
    report = {
        "onnx": str(args.onnx),
        "gravity": model.opt.gravity.tolist(),
        "seconds": args.seconds,
        "min_z": float(np.min(zs)),
        "final_z": float(zs[-1]),
        "max_tilt_deg": float(np.max(tilts)),
        "fell": bool(np.min(zs) < 0.05 or np.max(tilts) > 60),
        "video": str(mp4),
        "obs_dim": in_dim,
        "actuators": act_names,
    }
    (args.out / "official_stand_gravity.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
