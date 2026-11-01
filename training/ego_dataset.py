"""POL-WALK-EGO dataset: observation frames + joint / contact / root / camera-motion targets."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from torch.utils.data import Dataset


def quat_yaw(q_wxyz: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(a: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(a), np.cos(a))


def se2_inv(p: np.ndarray) -> np.ndarray:
    c, s = np.cos(p[..., 2]), np.sin(p[..., 2])
    return np.stack([-(c * p[..., 0] + s * p[..., 1]), -(-s * p[..., 0] + c * p[..., 1]), -p[..., 2]], -1)


def se2_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    c, s = np.cos(a[..., 2]), np.sin(a[..., 2])
    return np.stack(
        [a[..., 0] + c * b[..., 0] - s * b[..., 1], a[..., 1] + s * b[..., 0] + c * b[..., 1], wrap(a[..., 2] + b[..., 2])],
        -1,
    )


def step_deltas(poses: np.ndarray, fps: float) -> np.ndarray:
    d = se2_mul(se2_inv(poses[:-1]), poses[1:]) * fps
    return np.concatenate([d[:1], d], 0)


def integrate(vel: np.ndarray, fps: float) -> np.ndarray:
    pose = np.zeros(3)
    out = [pose]
    for t in range(1, len(vel)):
        pose = se2_mul(pose, vel[t] / fps)
        out.append(pose)
    return np.stack(out, 0)


def load_frames(seq: Path, variant: str, source: str, size: int) -> np.ndarray:
    if source == "C":
        rgb = np.load(seq / f"observations_{variant}" / "robot_camera_obs.npz")["rgb"]
    else:
        rgb = np.stack([fr[..., :3] for fr in imageio.get_reader(seq / f"front_{variant}.mp4")], 0)
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in rgb], 0)


def load_targets(seq: Path, variant: str) -> dict[str, np.ndarray]:
    meta = json.loads((seq / "metadata.json").read_text())
    fps = float(meta["fps"])
    q = np.load(seq / "joint_positions.npz")["joint_positions"].astype(np.float32)
    roots = np.load(seq / "root_poses.npz")
    rp, rq = roots["root_pos"], roots["root_quat_wxyz"]
    cam = np.load(seq / f"camera_{variant}.npz")
    P = np.stack([rp[:, 0], rp[:, 1], quat_yaw(rq)], -1)
    C = np.stack([cam["pos"][:, 0], cam["pos"][:, 1], np.arctan2(cam["forward"][:, 1], cam["forward"][:, 0])], -1)
    cam_vel = step_deltas(C, fps)
    vz = np.diff(cam["pos"][:, 2], prepend=cam["pos"][:1, 2]) * fps
    vz[0] = vz[1]
    out = {}
    cen_path = seq / f"image_centroid_{variant}.npz"
    if cen_path.exists():
        cen = np.load(cen_path)
        el = np.arcsin(np.clip(cam["forward"][:, 2], -1.0, 1.0))
        out["view"] = np.stack(
            [cen["uv"][:, 0], cen["uv"][:, 1], np.log(cen["area"] + 1e-4), cam["pos"][:, 2], el], -1
        ).astype(np.float32)
    return out | {
        "fps": fps,
        "cam_pose": C.astype(np.float32),
        "joint": q,
        "contact": np.load(seq / "contacts.npz")["contact"].astype(np.float32),
        "root_z": rp[:, 2].astype(np.float32),
        "pose": P.astype(np.float32),
        "rel": se2_mul(se2_inv(C), P).astype(np.float32),
        "cam_vel": np.concatenate([cam_vel[:, :2], vz[:, None], cam_vel[:, 2:]], -1).astype(np.float32),
        "root_vel": step_deltas(P, fps).astype(np.float32),
    }


class EgoDataset(Dataset):
    def __init__(self, seqs: list[Path], variant: str, source: str = "C", size: int = 128, clip_len: int | None = None, augment: bool = False):
        self.seqs, self.clip_len, self.augment = seqs, clip_len, augment
        self.frames = [load_frames(s, variant, source, size) for s in seqs]
        self.targets = [load_targets(s, variant) for s in seqs]
        self.fps = self.targets[0]["fps"]

    def __len__(self) -> int:
        return len(self.seqs)

    def __getitem__(self, i: int) -> dict:
        fr, tg = self.frames[i], self.targets[i]
        n = min(len(fr), len(tg["joint"]))
        s = 0
        if self.clip_len and n > self.clip_len:
            s = np.random.randint(0, n - self.clip_len + 1)
            n = self.clip_len
        sl = slice(s, s + n)
        x = fr[sl].astype(np.float32) / 255.0
        if self.augment:
            x = np.clip(x * np.random.uniform(0.8, 1.2) + np.random.uniform(-0.08, 0.08), 0, 1)
        pose = tg["pose"][sl]
        traj = se2_mul(se2_inv(pose[:1]), pose)
        keys = ("joint", "contact", "root_z", "rel", "cam_vel", "root_vel") + (("view",) if "view" in tg else ())
        out = {k: torch.from_numpy(np.ascontiguousarray(tg[k][sl])) for k in keys}
        out["traj"] = torch.from_numpy(traj.astype(np.float32))
        cp = tg["cam_pose"][sl]
        out["cam_traj"] = torch.from_numpy(se2_mul(se2_inv(cp[:1]), cp).astype(np.float32))
        out["frames"] = torch.from_numpy(np.ascontiguousarray(x.transpose(0, 3, 1, 2)))
        out["sequence"] = self.seqs[i].name
        return out


def split_sequences(root: Path) -> dict[str, list[Path]]:
    split = json.loads((root / "split.json").read_text())
    return {k: [root / n for n in split[k]] for k in ("train", "val", "test")}
