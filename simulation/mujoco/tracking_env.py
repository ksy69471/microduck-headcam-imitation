"""MuJoCo + BAM motion-tracking environment (PLAN §9).

Action space matches official policies: q_target = DEFAULT_POSE + action_scale * action.
Reward tracks a reference clip (joint / root / upright). Teacher ONNX actions (if present
in teacher_actions.npz) are exposed for BC warm-start.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from simulation.mujoco.image_centroid import mask_stats, project_point, robot_mask, set_free_camera
from simulation.mujoco.stand_policy_demo import (
    DEFAULT_POSE,
    RL_ROOT,
    load_scene_with_bam,
    projected_gravity,
)

DEFAULT_XML = RL_ROOT / "src/mjlab_microduck/robot/microduck/scene_walk.xml"


@dataclass
class MotionClip:
    joint_names: list[str]
    q: np.ndarray  # (T, 14) measured joints
    root_pos: np.ndarray
    root_quat: np.ndarray
    fps: float
    path: str
    teacher_actions: np.ndarray | None = None  # (T, 14) DEFAULT_POSE residuals

    @property
    def n_frames(self) -> int:
        return int(self.q.shape[0])

    @property
    def duration(self) -> float:
        return (self.n_frames - 1) / self.fps


def load_motion_clip(seq_dir: Path) -> MotionClip:
    seq_dir = Path(seq_dir)
    joints = np.load(seq_dir / "joint_positions.npz", allow_pickle=True)
    roots = np.load(seq_dir / "root_poses.npz", allow_pickle=True)
    meta = json.loads((seq_dir / "metadata.json").read_text())
    teacher = None
    ta = seq_dir / "teacher_actions.npz"
    if ta.exists():
        blob = np.load(ta, allow_pickle=True)
        teacher = blob["actions"].astype(np.float32)
    return MotionClip(
        joint_names=[str(x) for x in joints["joint_names"].tolist()],
        q=joints["joint_positions"].astype(np.float64),
        root_pos=roots["root_pos"].astype(np.float64),
        root_quat=roots["root_quat_wxyz"].astype(np.float64),
        fps=float(meta.get("fps", 30.0)),
        path=str(seq_dir),
        teacher_actions=teacher,
    )


def load_motion_library(root: Path, pattern: str = "sequence_*") -> list[MotionClip]:
    root = Path(root)
    dirs = sorted(root.glob(pattern))
    if not dirs:
        raise FileNotFoundError(f"no motions under {root}/{pattern}")
    return [load_motion_clip(d) for d in dirs]


def _yaw_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


class MotionTrackingEnv:
    OBS_DIM = 78
    ROOT_COND_DIM = 6
    ACT_DIM = 14

    def __init__(
        self,
        motions: list[MotionClip],
        *,
        xml_path: Path | None = None,
        policy_hz: float = 50.0,
        action_scale: float = 1.0,
        fall_z: float = 0.04,
        fall_tilt_deg: float = 70.0,
        max_episode_s: float | None = None,
        reward_weights: dict[str, float] | None = None,
        seed: int = 0,
        phase_rate: float = 1.0,
        root_cond: bool = False,
        xy_sigma: float = 0.08,
        img_variant: str = "hand_floor",
        img_sigma: float = 0.15,
        img_res: int = 64,
        img_every: int = 2,
        joint_sigma: float = 0.5,
    ) -> None:
        """``root_cond`` adds the reference root motion to the observation (planar velocity in
        the reference heading frame, position error in the robot heading frame, heading error).

        Reward weight ``img`` > 0 renders the robot from the demo's camera and rewards matching
        the demo's silhouette centroid / size in the image (needs ``image_centroid_<variant>.npz``
        and ``camera_<variant>.npz`` in the demo sequence)."""
        if not motions:
            raise ValueError("motions empty")
        self.motions = motions
        self.root_cond = bool(root_cond)
        self.obs_dim = self.OBS_DIM + (self.ROOT_COND_DIM if self.root_cond else 0)
        self.xy_sigma = float(xy_sigma)
        self._root_aux: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.policy_hz = float(policy_hz)
        self.action_scale = float(action_scale)
        self.fall_z = float(fall_z)
        self.fall_tilt_deg = float(fall_tilt_deg)
        self.max_episode_s = max_episode_s
        self.phase_rate = float(phase_rate)
        self.w = {
            "joint": 1.2,
            "root_xy": 0.25,
            "root_z": 0.4,
            "upright": 0.8,
            "smooth": 0.02,
            "energy": 0.005,
            "alive": 0.1,
            "heading": 0.0,
            "teacher": 0.0,  # set >0 to add action imitation reward online
            # Match forward step along reference heading (encourages walking, not standing).
            "travel": 0.0,
            "img": 0.0,
            **(reward_weights or {}),
        }
        self.img_variant = img_variant
        self.img_sigma = float(img_sigma)
        self.img_res = int(img_res)
        self.img_every = max(1, int(img_every))
        self.joint_sigma = float(joint_sigma)
        self._img_targets: dict[int, dict[str, np.ndarray]] = {}
        self._renderer: mujoco.Renderer | None = None
        self._img_last = (0.0, 1.0)
        self.rng = np.random.default_rng(seed)

        self.model, self.data, self.bam, self.act_names = load_scene_with_bam(
            Path(xml_path) if xml_path else DEFAULT_XML
        )
        joint_ids = np.array([self.model.actuator(n).trnid[0] for n in self.act_names])
        self.qpos_idx = self.model.jnt_qposadr[joint_ids]
        self.dof_idx = self.model.jnt_dofadr[joint_ids]
        self.trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.dt = float(self.model.opt.timestep)
        self.substeps = max(1, int(round(1.0 / self.policy_hz / self.dt)))
        self.default_pose = DEFAULT_POSE.astype(np.float64).copy()
        self._name_to_act = {n: i for i, n in enumerate(self.act_names)}

        self.clip: MotionClip | None = None
        self.phase_t = 0.0
        self.last_action = np.zeros(self.ACT_DIM, dtype=np.float32)
        self.prev_action = np.zeros(self.ACT_DIM, dtype=np.float32)
        self.prev_root_xy = np.zeros(2, dtype=np.float64)
        self.ep_travel = 0.0
        self.ep_forward = 0.0
        self.ep_expected_forward = 0.0
        self.steps = 0
        self.ep_return = 0.0

    def _align_q(self, q_clip: np.ndarray) -> np.ndarray:
        out = np.zeros(self.ACT_DIM, dtype=np.float64)
        assert self.clip is not None
        for i, name in enumerate(self.clip.joint_names):
            out[self._name_to_act[name]] = q_clip[i]
        return out

    def _align_act(self, a_clip: np.ndarray) -> np.ndarray:
        out = np.zeros(self.ACT_DIM, dtype=np.float32)
        assert self.clip is not None
        for i, name in enumerate(self.clip.joint_names):
            out[self._name_to_act[name]] = a_clip[i]
        return out

    def _sample_frame(self, t: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        assert self.clip is not None
        fps = self.clip.fps
        idx_f = np.clip(t * fps, 0, self.clip.n_frames - 1.001)
        i0 = int(np.floor(idx_f))
        i1 = min(i0 + 1, self.clip.n_frames - 1)
        a = float(idx_f - i0)
        q = (1 - a) * self.clip.q[i0] + a * self.clip.q[i1]
        rp = (1 - a) * self.clip.root_pos[i0] + a * self.clip.root_pos[i1]
        rq = (1 - a) * self.clip.root_quat[i0] + a * self.clip.root_quat[i1]
        rq = rq / max(np.linalg.norm(rq), 1e-8)
        dt = 1.0 / fps
        q1 = self.clip.q[min(i0 + 1, self.clip.n_frames - 1)]
        qd = (q1 - self.clip.q[i0]) / dt
        teacher = None
        if self.clip.teacher_actions is not None:
            ta = (1 - a) * self.clip.teacher_actions[i0] + a * self.clip.teacher_actions[i1]
            teacher = self._align_act(ta.astype(np.float32))
        return self._align_q(q), self._align_q(qd), rp, rq, teacher

    def _set_state_from_ref(self, t: float) -> None:
        q, _qd, rp, rq, teacher = self._sample_frame(t)
        d = self.data
        d.qpos[:] = 0.0
        d.qvel[:] = 0.0
        d.qpos[0:3] = rp
        d.qpos[3:7] = rq
        for i, adr in enumerate(self.qpos_idx):
            d.qpos[adr] = q[i]
        # Small state noise for robustness.
        d.qpos[self.qpos_idx] += self.rng.normal(0, 0.01, size=self.ACT_DIM)
        mujoco.mj_forward(self.model, d)
        if teacher is not None:
            self.bam.q_target[:] = self.default_pose + teacher
        else:
            self.bam.q_target[:] = q

    def reset(self, *, clip_idx: int | None = None, start_frac: float | None = None) -> np.ndarray:
        self.clip = (
            self.motions[int(self.rng.integers(0, len(self.motions)))]
            if clip_idx is None
            else self.motions[clip_idx]
        )
        if start_frac is None:
            start_frac = float(self.rng.uniform(0.0, 0.15))
        self.phase_t = float(start_frac) * self.clip.duration
        self._set_state_from_ref(self.phase_t)
        self.last_action[:] = 0
        self.prev_action[:] = 0
        self.prev_root_xy = self.data.xpos[self.trunk][:2].astype(np.float64).copy()
        self.ep_travel = 0.0
        self.ep_forward = 0.0
        self.ep_expected_forward = 0.0
        self.steps = 0
        self.ep_return = 0.0
        return self._obs()

    def teacher_action(self) -> np.ndarray | None:
        _q, _qd, _rp, _rq, teacher = self._sample_frame(self.phase_t)
        return teacher

    def _clip_root_aux(self) -> tuple[np.ndarray, np.ndarray]:
        """(yaw (T,), planar velocity in reference heading frame (T, 3)) for the current clip."""
        assert self.clip is not None
        key = id(self.clip)
        if key not in self._root_aux:
            yaw = _yaw_from_quat(self.clip.root_quat)
            xy = self.clip.root_pos[:, :2]
            dxy = np.diff(xy, axis=0)
            c, s = np.cos(yaw[:-1]), np.sin(yaw[:-1])
            vel = np.stack([c * dxy[:, 0] + s * dxy[:, 1], -s * dxy[:, 0] + c * dxy[:, 1], _wrap(np.diff(yaw))], -1)
            vel = np.concatenate([vel, vel[-1:]], 0) * self.clip.fps
            self._root_aux[key] = (yaw, vel)
        return self._root_aux[key]

    def _img_target(self) -> dict[str, np.ndarray]:
        """Demo silhouette stats + the demo camera expressed in this clip's reference frame.

        Uses the clip's own ``image_target.npz`` (estimated from the head-camera observation) when
        present, otherwise the privileged demo camera / segmentation aligned at frame 0."""
        assert self.clip is not None
        key = id(self.clip)
        own = Path(self.clip.path) / "image_target.npz"
        if key not in self._img_targets and own.exists():
            tg = np.load(own)
            self._img_targets[key] = {
                "uv": tg["uv"], "area": tg["area"], "lookat": tg["lookat"], "aed": tg["aed"],
                "fovy": float(np.asarray(tg["fovy_deg"]).reshape(-1)[0]),
            }
        if key not in self._img_targets:
            meta = json.loads((Path(self.clip.path) / "metadata.json").read_text())
            demo = Path(meta.get("demo_sequence", self.clip.path))
            cen = np.load(demo / f"image_centroid_{self.img_variant}.npz")
            cam = np.load(demo / f"camera_{self.img_variant}.npz")
            droot = np.load(demo / "root_poses.npz")
            d_xy, d_yaw = droot["root_pos"][0, :2], float(_yaw_from_quat(droot["root_quat_wxyz"][0]))
            r_xy, r_yaw = self.clip.root_pos[0, :2], float(_yaw_from_quat(self.clip.root_quat[0]))
            dyaw = r_yaw - d_yaw
            c, s = np.cos(dyaw), np.sin(dyaw)
            rel = cam["lookat"][:, :2] - d_xy
            look = cam["lookat"].copy()
            look[:, 0] = c * rel[:, 0] - s * rel[:, 1] + r_xy[0]
            look[:, 1] = s * rel[:, 0] + c * rel[:, 1] + r_xy[1]
            aed = cam["azimuth_elevation_distance"].copy()
            aed[:, 0] += np.degrees(dyaw)
            self._img_targets[key] = {
                "uv": cen["uv"], "area": cen["area"], "lookat": look, "aed": aed,
                "fovy": float(np.asarray(cen["fovy_deg"]).reshape(-1)[0]),
            }
        return self._img_targets[key]

    def _img_error(self) -> tuple[float, float]:
        """(centroid distance, |log size ratio|) between own render and the demo at this frame."""
        tg = self._img_target()
        idx = int(np.clip(round(self.phase_t * self.clip.fps), 0, len(tg["uv"]) - 1))  # type: ignore[union-attr]
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.img_res, width=self.img_res)
            self._renderer.enable_segmentation_rendering()
            self._cam = mujoco.MjvCamera()
            mujoco.mjv_defaultCamera(self._cam)
        set_free_camera(self._cam, tg["lookat"][idx], tg["aed"][idx])
        self._renderer.update_scene(self.data, camera=self._cam)
        u, v, area = mask_stats(robot_mask(self._renderer.render(), self.model))
        if area <= 0:
            uv = project_point(self.data.subtree_com[self.trunk], tg["lookat"][idx], tg["aed"][idx], tg["fovy"])
            return float(np.linalg.norm(uv - tg["uv"][idx])), 1.0
        e_c = float(np.hypot(u - tg["uv"][idx, 0], v - tg["uv"][idx, 1]))
        e_s = float(abs(0.5 * np.log((area + 1e-4) / (tg["area"][idx] + 1e-4))))
        return e_c, e_s

    def robot_yaw(self) -> float:
        R = self.data.xmat[self.trunk].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def _root_errors(self) -> tuple[np.ndarray, float, np.ndarray]:
        """Reference velocity, heading error and position error (robot heading frame)."""
        assert self.clip is not None
        yaw_ref, vel_ref = self._clip_root_aux()
        idx = int(np.clip(round(self.phase_t * self.clip.fps), 0, self.clip.n_frames - 1))
        _q, _qd, rp, _rq, _t = self._sample_frame(self.phase_t)
        yaw = self.robot_yaw()
        d = rp[:2] - self.data.xpos[self.trunk][:2]
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
        return vel_ref[idx], float(_wrap(yaw_ref[idx] - yaw)), local

    def _obs(self) -> np.ndarray:
        d = self.data
        q_ref, qd_ref, _rp, _rq, _teacher = self._sample_frame(self.phase_t)
        q = d.qpos[self.qpos_idx].astype(np.float32)
        qd = d.qvel[self.dof_idx].astype(np.float32)
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity")
        if sid >= 0:
            adr = self.model.sensor_adr[sid]
            dim = self.model.sensor_dim[sid]
            ang = d.sensordata[adr : adr + dim].astype(np.float32)
        else:
            ang = d.qvel[3:6].astype(np.float32)
        grav = projected_gravity(d, self.trunk)
        phase = self.phase_t / max(self.clip.duration, 1e-6)  # type: ignore[union-attr]
        phase_feat = np.array(
            [np.sin(2 * np.pi * phase), np.cos(2 * np.pi * phase)], dtype=np.float32
        )
        obs = np.concatenate(
            [
                ang,
                grav,
                (q - self.default_pose).astype(np.float32),
                qd,
                self.last_action,
                (q_ref - self.default_pose).astype(np.float32),
                qd_ref.astype(np.float32),
                phase_feat,
            ]
        ).astype(np.float32)
        if self.root_cond:
            vel_ref, dyaw, local = self._root_errors()
            obs = np.concatenate(
                [obs, vel_ref.astype(np.float32), np.clip(local, -0.5, 0.5).astype(np.float32), np.float32([dyaw])]
            )
        assert obs.shape[0] == self.obs_dim, obs.shape
        return obs

    def _tilt_deg(self) -> float:
        up_z = float(self.data.xmat[self.trunk].reshape(3, 3)[2, 2])
        return float(np.degrees(np.arccos(np.clip(up_z, -1.0, 1.0))))

    def _reward(self, action: np.ndarray, teacher: np.ndarray | None) -> tuple[float, dict[str, float]]:
        q_ref, _qd_ref, rp_ref, _rq, _t = self._sample_frame(self.phase_t)
        q = self.data.qpos[self.qpos_idx]
        root = self.data.xpos[self.trunk]
        joint_err = float(np.mean((q - q_ref) ** 2))
        r_joint = float(np.exp(-joint_err / max(self.joint_sigma, 1e-8)))
        xy_err = float(np.sum((root[:2] - rp_ref[:2]) ** 2))
        r_xy = float(np.exp(-xy_err / self.xy_sigma))
        r_heading = 0.0
        if self.w["heading"] > 0:
            yaw_ref, _vel = self._clip_root_aux()
            idx = int(np.clip(round(self.phase_t * self.clip.fps), 0, self.clip.n_frames - 1))  # type: ignore[union-attr]
            r_heading = float(np.exp(-(_wrap(yaw_ref[idx] - self.robot_yaw()) ** 2) / 0.1))
        z_err = float((root[2] - rp_ref[2]) ** 2)
        r_z = float(np.exp(-z_err / 0.02))
        tilt = self._tilt_deg()
        r_up = float(np.exp(-((tilt / 30.0) ** 2)))
        smooth = float(np.mean((action - self.prev_action) ** 2))
        energy = float(np.mean(action**2))
        r_teacher = 0.0
        if teacher is not None and self.w["teacher"] > 0:
            # action is already in teacher residual space (scale~1)
            r_teacher = float(np.exp(-float(np.mean((action - teacher) ** 2)) / 0.3))
        # Travel: linear per-step progress (no saturation when standing) + cumulative lag.
        dxy = root[:2] - self.prev_root_xy
        yaw_ref, vel_ref = self._clip_root_aux()
        idx = int(np.clip(round(self.phase_t * self.clip.fps), 0, self.clip.n_frames - 1))  # type: ignore[union-attr]
        c, s = float(np.cos(yaw_ref[idx])), float(np.sin(yaw_ref[idx]))
        forward = float(c * dxy[0] + s * dxy[1])
        expected = float(vel_ref[idx, 0]) / self.policy_hz
        self.ep_forward += forward
        self.ep_expected_forward += max(expected, 0.0)
        if expected > 1e-4:
            r_step = float(np.clip(forward / expected, 0.0, 1.5))
        else:
            r_step = 1.0
        gap = max(0.0, self.ep_expected_forward - self.ep_forward)
        r_cum = float(np.exp(-(gap**2) / 0.04))  # ~0.2 m lag scale
        r_travel = 0.5 * r_step + 0.5 * r_cum
        r_img, img_c, img_s = 0.0, float("nan"), float("nan")
        if self.w["img"] > 0:
            if self.steps % self.img_every == 0:
                self._img_last = self._img_error()
            img_c, img_s = self._img_last
            r_img = float(np.exp(-(img_c + 0.5 * img_s) / self.img_sigma))
        total = (
            self.w["joint"] * r_joint
            + self.w["root_xy"] * r_xy
            + self.w["root_z"] * r_z
            + self.w["upright"] * r_up
            + self.w["heading"] * r_heading
            + self.w["alive"] * 1.0
            + self.w["teacher"] * r_teacher
            + self.w["travel"] * r_travel
            + self.w["img"] * r_img
            - self.w["smooth"] * smooth
            - self.w["energy"] * energy
        )
        info = {
            "r_joint": r_joint,
            "r_xy": r_xy,
            "r_z": r_z,
            "r_up": r_up,
            "r_teacher": r_teacher,
            "r_heading": r_heading,
            "r_travel": r_travel,
            "r_travel_step": r_step,
            "r_travel_cum": r_cum,
            "r_img": r_img,
            "img_centroid_err": img_c,
            "img_size_err": img_s,
            "forward_step": forward,
            "expected_step": expected,
            "travel_gap_m": gap,
            "root_xy": root[:2].copy(),
            "xy_err_ref": float(np.sqrt(xy_err)),
            "yaw": self.robot_yaw(),
            "joint_mae": float(np.mean(np.abs(q - q_ref))),
            "tilt_deg": tilt,
            "z": float(root[2]),
        }
        return float(total), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float32).reshape(self.ACT_DIM)
        action = np.clip(action, -1.5, 1.5)
        _q_ref, _qd, _rp, _rq, teacher = self._sample_frame(self.phase_t)
        target = self.default_pose + self.action_scale * action.astype(np.float64)
        for i, jid in enumerate([self.model.actuator(n).trnid[0] for n in self.act_names]):
            lo, hi = self.model.jnt_range[jid]
            target[i] = float(np.clip(target[i], lo, hi))
        self.bam.q_target[:] = target
        for _ in range(self.substeps):
            self.bam.update()
            mujoco.mj_step(self.model, self.data)

        self.phase_t += self.phase_rate / self.policy_hz
        reward, info = self._reward(action, teacher)
        self.prev_action[:] = self.last_action
        self.last_action[:] = action
        root_xy = self.data.xpos[self.trunk][:2].astype(np.float64)
        self.ep_travel += float(np.linalg.norm(root_xy - self.prev_root_xy))
        self.prev_root_xy = root_xy.copy()
        self.steps += 1
        self.ep_return += reward

        fell = bool(info["z"] < self.fall_z or info["tilt_deg"] > self.fall_tilt_deg)
        timed_out = self.phase_t >= self.clip.duration - 1e-6  # type: ignore[union-attr]
        if self.max_episode_s is not None:
            timed_out = timed_out or (self.steps / self.policy_hz >= self.max_episode_s)
        done = fell or timed_out
        if fell:
            reward -= 1.0
        info.update(
            {
                "fell": fell,
                "success": bool(timed_out and not fell),
                "ep_return": self.ep_return,
                "travel_xy": self.ep_travel,
                "phase_t": self.phase_t,
                "clip": self.clip.path if self.clip else "",
                "teacher": teacher,
            }
        )
        return self._obs(), reward, done, info
