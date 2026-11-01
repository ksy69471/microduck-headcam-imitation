"""Closed-loop official ONNX teachers for BC / DAgger (walk, sitstand, stand, …)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort

from simulation.mujoco.walk_policy_capture import build_command, build_obs


class OnnxWalkingTeacher:
    """Generic 61-D obs / 14-D action teacher. Name kept for compatibility."""

    def __init__(
        self,
        onnx_path: Path,
        *,
        vx: float = 0.32,
        vy: float = 0.0,
        yaw: float = 0.0,
        mode: str = "walk",
    ) -> None:
        self.sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        self.mode = mode
        self.command = build_command(vx, vy, yaw)
        self.last_action = np.zeros(14, dtype=np.float32)
        self.clip_meta: dict = {}
        self.phase_period = 4.0

    def set_command(self, vx: float, vy: float = 0.0, yaw: float = 0.0) -> None:
        self.command = build_command(vx, vy, yaw)

    def set_command_from_clip_meta(self, meta: dict) -> None:
        self.clip_meta = meta or {}
        self.mode = str(meta.get("mode", self.mode))
        tw = meta.get("command_twist") or {}
        if tw:
            self.set_command(float(tw.get("vx", 0.0)), float(tw.get("vy", 0.0)), float(tw.get("yaw", 0.0)))
        if "phase_period" in meta:
            self.phase_period = float(meta["phase_period"])

    def reset(self) -> None:
        self.last_action[:] = 0

    def _command_for_env(self, env) -> np.ndarray:
        mode = self.mode
        # Prefer live phase from tracking env when available.
        t = float(getattr(env, "phase_t", 0.0))
        dur = float(env.clip.duration) if env.clip is not None else 5.0
        if mode in ("stand",):
            return build_command(0.0, 0.0, 0.0)
        if mode in ("walk",):
            return self.command
        if mode == "velstand":
            if t < dur * 0.4:
                return build_command(0.0, 0.0, 0.0)
            return self.command
        if mode == "sitstand":
            p = t / max(dur, 1e-6)
            if p < 0.12:
                sit = 0.0
            elif p < 0.28:
                sit = (p - 0.12) / 0.16
            elif p < 0.48:
                sit = 1.0
            elif p < 0.72:
                sit = 1.0 - (p - 0.48) / 0.24
            else:
                sit = 0.0
            return build_command(float(np.clip(sit, 0.0, 1.0)), 0.0, 0.0)
        if mode == "ground_pick":
            phase = min(0.699, (t / self.phase_period) % 1.0)
            cmd = build_command(0.0, 0.0, 0.0)
            cmd[0] = math.cos(2 * math.pi * phase)
            cmd[1] = math.sin(2 * math.pi * phase)
            return cmd
        if mode in ("roulade", "kick_left", "kick_right"):
            return build_command(0.0, 0.0, 0.0)
        return self.command

    def act_env(self, env) -> np.ndarray:
        if env.clip is not None:
            meta_path = Path(env.clip.path) / "metadata.json"
            if meta_path.exists():
                self.set_command_from_clip_meta(json.loads(meta_path.read_text()))
        command = self._command_for_env(env)
        obs61 = build_obs(
            env.model,
            env.data,
            env.trunk,
            env.qpos_idx,
            env.dof_idx,
            self.last_action,
            command,
        )
        action = self.sess.run([self.out_name], {self.in_name: obs61[None]})[0][0].astype(np.float32)
        self.last_action = action
        return action
