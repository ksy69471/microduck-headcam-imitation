"""Capture alpha_walking rollouts for root / camera-motion estimation (POL-WALK-EGO).

Physics is simulated once (MuJoCo + BAM, official ONNX walking policy with
piecewise twist commands). The recorded qpos is then *replayed kinematically*
into a render-only model that adds visual landmarks, so every visual variant
shows exactly the same motion:

  follow_floor : camera rigidly follows the trunk (legacy POL-WALK framing), checker floor
  hand_floor   : hand-held camera (lagged follow + pan + shake), checker floor + landmarks
  hand_plain   : same hand-held camera, untextured floor, no landmarks (no ground cues)

Per frame we store joints, root pose, foot contacts, twist command and the true
camera pose taken from the renderer (pos / forward / up), so camera motion and
robot motion can be supervised separately.
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

from simulation.mujoco.stand_policy_demo import DEFAULT_POSE, RL_ROOT, load_scene_with_bam  # noqa: E402
from simulation.mujoco.walk_policy_capture import CAM_PRESETS, build_command, build_obs, look_camera  # noqa: E402

SCENE_XML = RL_ROOT / "src/mjlab_microduck/robot/microduck/scene_walk.xml"
VARIANTS = ("follow_floor", "hand_floor", "hand_plain")
LANDMARK_GROUP = 4
IMG = 256


def sample_schedule(rng: np.random.Generator, seconds: float) -> list[dict]:
    """2–3 constant-twist segments so root motion is not a per-clip constant."""
    n_seg = int(rng.integers(2, 4))
    cuts = np.sort(rng.uniform(0.25, 0.75, size=n_seg - 1)) * seconds
    bounds = [0.0, *cuts.tolist(), seconds]
    segs = []
    for i in range(n_seg):
        segs.append(
            {
                "t0": float(bounds[i]),
                "t1": float(bounds[i + 1]),
                "vx": float(rng.uniform(0.26, 0.40)),
                "vy": float(rng.choice([0.0, 0.0, 0.04, -0.04])),
                "yaw": float(rng.choice([0.0, 1.0]) * rng.uniform(-0.35, 0.35)),
            }
        )
    return segs


def command_at(segs: list[dict], t: float) -> dict:
    for s in segs:
        if t < s["t1"]:
            return s
    return segs[-1]


def simulate(onnx: Path, segs: list[dict], seconds: float, fps: float, warmup_s: float, policy_hz: float = 50.0,
             command_fn=None, sess=None) -> dict:
    """``command_fn(t_walk) -> (vx, vy, yaw)`` overrides the segment schedule when given."""
    model, data, bam_ctrl, act_names = load_scene_with_bam(SCENE_XML)
    joint_ids = np.array([model.actuator(n).trnid[0] for n in act_names])
    qpos_idx = model.jnt_qposadr[joint_ids]
    dof_idx = model.jnt_dofadr[joint_ids]
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    feet = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in ("left_foot_collision", "right_foot_collision")]

    data.qpos[:] = 0
    data.qpos[0:3] = [0, 0, 0.12]
    data.qpos[3:7] = [1, 0, 0, 0]
    for i, adr in enumerate(qpos_idx):
        data.qpos[adr] = DEFAULT_POSE[i]
    mujoco.mj_forward(model, data)
    bam_ctrl.q_target[:] = DEFAULT_POSE

    sess = sess or ort.InferenceSession(str(onnx), providers=["CPUExecutionProvider"])
    in_name, out_name = sess.get_inputs()[0].name, sess.get_outputs()[0].name
    last_action = np.zeros(14, dtype=np.float32)
    cmd_stand = build_command(0.0, 0.0, 0.0)

    dt = float(model.opt.timestep)
    policy_steps = max(1, int(round(1.0 / policy_hz / dt)))
    video_steps = max(1, int(round(1.0 / fps / dt)))
    n_frames = int(round(seconds * fps))
    warmup_physics = int(round(warmup_s / dt))

    log = {k: [] for k in ("qpos", "q", "root_pos", "root_quat", "action", "q_target", "contact", "command", "tilt")}
    step_i = 0
    while len(log["q"]) < n_frames:
        walking = step_i >= warmup_physics
        t_walk = (step_i - warmup_physics) * dt
        if not walking:
            command = cmd_stand
        elif command_fn is not None:
            command = build_command(*command_fn(t_walk))
        else:
            seg = command_at(segs, t_walk)
            command = build_command(seg["vx"], seg["vy"], seg["yaw"])
        if step_i % policy_steps == 0:
            obs = build_obs(model, data, trunk, qpos_idx, dof_idx, last_action, command)
            last_action = sess.run([out_name], {in_name: obs[None]})[0][0].astype(np.float32)
            bam_ctrl.q_target[:] = DEFAULT_POSE + last_action
        bam_ctrl.update()
        mujoco.mj_step(model, data)
        step_i += 1
        if step_i % video_steps != 0 or step_i <= warmup_physics:
            continue
        contact = np.zeros(2, dtype=np.float32)
        for c in data.contact[: data.ncon]:
            for k, g in enumerate(feet):
                if {int(c.geom1), int(c.geom2)} == {g, floor}:
                    contact[k] = 1.0
        up_z = float(data.xmat[trunk].reshape(3, 3)[2, 2])
        log["qpos"].append(data.qpos.copy())
        log["q"].append(data.qpos[qpos_idx].copy())
        log["root_pos"].append(data.qpos[0:3].copy())
        log["root_quat"].append(data.qpos[3:7].copy())
        log["action"].append(last_action.copy())
        log["q_target"].append(bam_ctrl.q_target.copy())
        log["contact"].append(contact)
        log["command"].append(command[:3].copy())
        log["tilt"].append(float(np.degrees(np.arccos(np.clip(up_z, -1, 1)))))
    out = {k: np.asarray(v) for k, v in log.items()}
    out["act_names"] = act_names
    return out


def place_landmarks(rng: np.random.Generator, path_xy: np.ndarray, n: int = 8) -> list[dict]:
    """Static coloured props near (but not on) the walked path."""
    lo, hi = path_xy.min(0) - 0.9, path_xy.max(0) + 0.9
    marks: list[dict] = []
    tries = 0
    while len(marks) < n and tries < 500:
        tries += 1
        p = rng.uniform(lo, hi)
        if np.min(np.linalg.norm(path_xy - p, axis=1)) < 0.28:
            continue
        if any(np.linalg.norm(p - np.array(m["xy"])) < 0.18 for m in marks):
            continue
        kind = str(rng.choice(["box", "cylinder", "sphere"]))
        s = float(rng.uniform(0.025, 0.07))
        h = float(rng.uniform(0.03, 0.12))
        marks.append(
            {
                "xy": p.tolist(),
                "kind": kind,
                "size": s,
                "height": h,
                "rgba": [*rng.uniform(0.15, 0.95, size=3).tolist(), 1.0],
            }
        )
    return marks


def floor_style(rng: np.random.Generator) -> dict:
    base = rng.uniform(0.1, 0.45, size=3)
    alt = np.clip(base + rng.uniform(0.18, 0.4) * rng.choice([-1, 1]), 0.02, 0.95)
    return {"rgb1": base.tolist(), "rgb2": alt.tolist(), "texrepeat": float(rng.uniform(4.0, 7.0))}


def build_render_model(style: dict, marks: list[dict]) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENE_XML))
    for tex in spec.textures:
        if tex.name == "groundplane":
            tex.rgb1[:] = style["rgb1"]
            tex.rgb2[:] = style["rgb2"]
    for mat in spec.materials:
        if mat.name == "groundplane":
            mat.texrepeat[:] = [style["texrepeat"], style["texrepeat"]]
    plain = spec.add_material()
    plain.name = "plain_floor"
    plain.rgba[:] = [*np.mean([style["rgb1"], style["rgb2"]], axis=0).tolist(), 1.0]
    plain.reflectance = 0.0
    kinds = {"box": mujoco.mjtGeom.mjGEOM_BOX, "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER, "sphere": mujoco.mjtGeom.mjGEOM_SPHERE}
    for i, m in enumerate(marks):
        if m["kind"] == "box":
            size, z = [m["size"], m["size"], m["height"] / 2], m["height"] / 2
        elif m["kind"] == "cylinder":
            size, z = [m["size"], m["height"] / 2, 0], m["height"] / 2
        else:
            size, z = [m["size"], 0, 0], m["size"]
        spec.worldbody.add_geom(
            name=f"landmark_{i}",
            type=kinds[m["kind"]],
            size=size,
            pos=[m["xy"][0], m["xy"][1], z],
            rgba=m["rgba"],
            contype=0,
            conaffinity=0,
            group=LANDMARK_GROUP,
        )
    return spec.compile()


class HandheldCamera:
    """Lagged follow with slow pan / elevation / distance drift and small shake."""

    def __init__(self, rng: np.random.Generator, fps: float, base_az: float):
        self.fps = fps
        self.base_az = base_az
        self.tau = float(rng.uniform(0.4, 0.9))
        self.pan_amp = float(rng.uniform(8.0, 22.0))
        self.pan_f = float(rng.uniform(0.06, 0.2))
        self.pan_ph = float(rng.uniform(0, 2 * np.pi))
        self.el0 = float(rng.uniform(-9.0, -3.0))
        self.el_amp = float(rng.uniform(1.0, 3.0))
        self.el_f = float(rng.uniform(0.1, 0.3))
        self.d0 = float(rng.uniform(0.62, 0.82))
        self.d_amp = float(rng.uniform(0.03, 0.10))
        self.d_f = float(rng.uniform(0.05, 0.2))
        self.shake = float(rng.uniform(0.002, 0.006))
        self.rng = rng
        self.target: np.ndarray | None = None
        self.noise = np.zeros(3)

    def params(self) -> dict:
        return {k: v for k, v in vars(self).items() if isinstance(v, float)}

    def update(self, cam: mujoco.MjvCamera, trunk_pos: np.ndarray, t: float) -> None:
        goal = trunk_pos + np.array([0.0, 0.0, 0.02])
        if self.target is None:
            self.target = goal.copy()
        a = 1.0 - np.exp(-1.0 / (self.fps * self.tau))
        self.target = self.target + a * (goal - self.target)
        self.noise = 0.7 * self.noise + 0.3 * self.rng.normal(0.0, self.shake, size=3)
        cam.lookat[:] = self.target + self.noise
        cam.azimuth = self.base_az + self.pan_amp * np.sin(2 * np.pi * self.pan_f * t + self.pan_ph)
        cam.elevation = self.el0 + self.el_amp * np.sin(2 * np.pi * self.el_f * t)
        cam.distance = self.d0 + self.d_amp * np.sin(2 * np.pi * self.d_f * t + 1.3)


def render_variants(sim: dict, style: dict, marks: list[dict], rng: np.random.Generator, fps: float, out: Path) -> dict:
    model = build_render_model(style, marks)
    data = mujoco.MjData(model)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    trunk = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
    mat_checker = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "groundplane")
    mat_plain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "plain_floor")
    renderer = mujoco.Renderer(model, height=IMG, width=IMG)
    fovy = float(model.vis.global_.fovy)

    probe = mujoco.MjvCamera()
    look_camera(probe, np.zeros(3), CAM_PRESETS["front"]["offset"], CAM_PRESETS["front"]["look_z"])
    # Same hand-held path for hand_floor and hand_plain (only the scene differs).
    hand_seed = int(rng.integers(1 << 31))
    meta = {}
    for variant in VARIANTS:
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        hand = HandheldCamera(np.random.default_rng(hand_seed), fps, probe.azimuth) if variant.startswith("hand") else None
        model.geom_matid[floor] = mat_plain if variant.endswith("plain") else mat_checker
        opt = mujoco.MjvOption()
        mujoco.mjv_defaultOption(opt)
        opt.geomgroup[LANDMARK_GROUP] = 0 if variant.endswith("plain") else 1
        frames, pos, fwd, up, lookat, aed = [], [], [], [], [], []
        for t_i, qpos in enumerate(sim["qpos"]):
            data.qpos[:] = qpos
            mujoco.mj_forward(model, data)
            trunk_pos = data.xpos[trunk].copy()
            if hand is None:
                look_camera(cam, trunk_pos, CAM_PRESETS["front"]["offset"], CAM_PRESETS["front"]["look_z"])
            else:
                hand.update(cam, trunk_pos, t_i / fps)
            renderer.update_scene(data, camera=cam, scene_option=opt)
            gl = renderer.scene.camera
            pos.append(0.5 * (np.array(gl[0].pos) + np.array(gl[1].pos)))
            fwd.append(np.array(gl[0].forward))
            up.append(np.array(gl[0].up))
            lookat.append(np.array(cam.lookat))
            aed.append([cam.azimuth, cam.elevation, cam.distance])
            frames.append(renderer.render().copy())
        mp4 = out / f"front_{variant}.mp4"
        imageio.mimsave(mp4, frames, fps=fps)
        np.savez_compressed(
            out / f"camera_{variant}.npz",
            pos=np.asarray(pos),
            forward=np.asarray(fwd),
            up=np.asarray(up),
            lookat=np.asarray(lookat),
            azimuth_elevation_distance=np.asarray(aed),
            fovy_deg=np.array([fovy]),
            image_size=np.array([IMG, IMG]),
        )
        meta[variant] = {
            "video": mp4.name,
            "camera": f"camera_{variant}.npz",
            "camera_mode": "handheld" if hand else "rigid_follow",
            "floor": "plain" if variant.endswith("plain") else "checker",
            "landmarks": 0 if variant.endswith("plain") else len(marks),
            "handheld_params": hand.params() if hand else None,
        }
    renderer.close()
    return meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--seconds", type=float, default=5.0)
    p.add_argument("--fps", type=float, default=30.0)
    p.add_argument("--warmup-s", type=float, default=0.6)
    p.add_argument("--onnx", type=Path, default=REPO_ROOT / "work/local/scratch/policies/alpha_walking.onnx")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    for attempt in range(6):
        segs = sample_schedule(rng, args.seconds)
        sim = simulate(args.onnx, segs, args.seconds, args.fps, args.warmup_s)
        if float(np.max(sim["tilt"])) < 40.0 and float(np.min(sim["root_pos"][:, 2])) > 0.07:
            break
        print(f"attempt {attempt}: fell (max tilt {np.max(sim['tilt']):.1f}), resampling", flush=True)
    else:
        raise SystemExit("could not produce a stable rollout")

    args.out.mkdir(parents=True, exist_ok=True)
    style = floor_style(rng)
    marks = place_landmarks(rng, sim["root_pos"][:, :2])
    variants = render_variants(sim, style, marks, rng, args.fps, args.out)

    n = len(sim["q"])
    t = np.arange(n) / args.fps
    names = np.array(sim["act_names"])
    np.savez_compressed(args.out / "joint_positions.npz", times=t, joint_positions=sim["q"], joint_names=names, fps=np.array([args.fps]))
    np.savez_compressed(args.out / "root_poses.npz", times=t, root_pos=sim["root_pos"], root_quat_wxyz=sim["root_quat"])
    np.savez_compressed(
        args.out / "teacher_actions.npz",
        times=t,
        actions=sim["action"].astype(np.float32),
        q_targets=sim["q_target"],
        joint_names=names,
        action_space="default_pose_residual",
    )
    np.savez_compressed(args.out / "contacts.npz", times=t, contact=sim["contact"], feet=np.array(["left", "right"]))
    np.savez_compressed(args.out / "commands.npz", times=t, twist=sim["command"])
    np.savez_compressed(args.out / "qpos_full.npz", qpos=sim["qpos"])
    xy = sim["root_pos"][:, :2]
    meta = {
        "motion_id": "POL-WALK-EGO",
        "source": "mujoco_alpha_walking_onnx_bam",
        "renderer": "mujoco",
        "seed": args.seed,
        "fps": args.fps,
        "n_frames": n,
        "duration_s": args.seconds,
        "warmup_s": args.warmup_s,
        "schedule": segs,
        "joint_names": names.tolist(),
        "travel_m": float(np.linalg.norm(xy[-1] - xy[0])),
        "path_len_m": float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))),
        "max_tilt_deg": float(np.max(sim["tilt"])),
        "floor_style": style,
        "landmarks": marks,
        "variants": variants,
        "ground_truth_note": "contacts measured from MuJoCo foot/floor contacts",
    }
    (args.out / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({"out": str(args.out), "travel_m": meta["travel_m"], "segments": len(segs)}))


if __name__ == "__main__":
    main()
