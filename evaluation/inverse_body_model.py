#!/usr/bin/env python3
"""Learned inverse body model: body velocity sequence -> walking command sequence.

Trained only on the robot's own "motor babbling": random piecewise-constant commands are sent to the
walking skill (alpha_walking.onnx + BAM) and the resulting planar body velocity is recorded. No demo
clip is used. Input noise mimics video-estimated velocity, so the model can be applied to V2M output.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from training.ego_dataset import quat_yaw, step_deltas  # noqa: E402

ONNX = REPO / "work/local/scratch/policies/alpha_walking.onnx"
OUT = REPO / "work/local/runs/ego/cmd/inverse_nn"
FPS, SECONDS, WARMUP = 30.0, 5.0, 0.6
LO, HI = np.array([-0.1, -0.15, -0.8]), np.array([0.5, 0.15, 0.8])


def babble_schedule(rng: np.random.Generator) -> list[dict]:
    n_seg = int(rng.integers(1, 5))
    cuts = np.sort(rng.uniform(0.15, 0.85, size=n_seg - 1)) * SECONDS
    bounds = [0.0, *cuts.tolist(), SECONDS]
    return [{"t0": float(bounds[i]), "t1": float(bounds[i + 1]), "vx": float(rng.uniform(0.15, 0.45)),
             "vy": float(rng.uniform(-0.08, 0.08)) * float(rng.random() < 0.7),
             "yaw": float(rng.uniform(-0.5, 0.5)) * float(rng.random() < 0.6)} for i in range(n_seg)]


def babble_job(seed: int) -> dict:
    import onnxruntime as ort

    from simulation.mujoco.capture_ego import simulate

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    segs = babble_schedule(np.random.default_rng(seed))
    sim = simulate(ONNX, segs, SECONDS, FPS, WARMUP, sess=sess)
    pose = np.stack([sim["root_pos"][:, 0], sim["root_pos"][:, 1], quat_yaw(sim["root_quat"])], -1)
    return {"vel": step_deltas(pose, FPS).astype(np.float32), "cmd": np.asarray(sim["command"], np.float32),
            "fell": bool(np.max(sim["tilt"]) > 45.0)}


class InverseNet(nn.Module):
    def __init__(self, ch: int = 64):
        super().__init__()
        layers, c_in = [], 3
        for d in (1, 2, 4, 8, 16):
            layers += [nn.Conv1d(c_in, ch, 5, padding=2 * d, dilation=d), nn.GELU()]
            c_in = ch
        self.body = nn.Sequential(*layers)
        self.head = nn.Conv1d(ch, 3, 1)
        self.register_buffer("v_scale", torch.tensor([0.3, 0.1, 0.5]))
        self.register_buffer("c_mid", torch.tensor((LO + HI) / 2, dtype=torch.float32))
        self.register_buffer("c_half", torch.tensor((HI - LO) / 2, dtype=torch.float32))

    def forward(self, vel: torch.Tensor) -> torch.Tensor:
        h = self.body((vel / self.v_scale).transpose(1, 2))
        return self.c_mid + self.c_half * torch.tanh(self.head(h).transpose(1, 2))


def smooth_noise(shape, std: np.ndarray, rng: np.random.Generator, k: int = 9) -> np.ndarray:
    n = rng.standard_normal(shape)
    ker = np.ones(k) / np.sqrt(k)
    n = np.apply_along_axis(lambda x: np.convolve(x, ker, mode="same"), 1, n)
    return n * std


def train(data: list[dict], steps: int, seed: int = 0) -> tuple[InverseNet, dict]:
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    V = np.stack([d["vel"] for d in data])
    C = np.stack([d["cmd"] for d in data])
    n_val = max(1, len(V) // 10)
    Vt, Ct, Vv, Cv = V[n_val:], C[n_val:], V[:n_val], C[:n_val]
    net = InverseNet()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    noise_std = np.array([0.03, 0.03, 0.12])
    for it in range(steps):
        idx = rng.integers(0, len(Vt), 64)
        v = Vt[idx] + smooth_noise(Vt[idx].shape, noise_std * rng.uniform(0, 1, (64, 1, 1)), rng)
        loss = torch.mean((net(torch.tensor(v, dtype=torch.float32)) - torch.tensor(Ct[idx])) ** 2
                          / torch.tensor([0.05, 0.03, 0.2]) ** 2)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        pv = net(torch.tensor(Vv)).numpy()
    r2 = 1 - ((pv - Cv) ** 2).sum((0, 1)) / ((Cv - Cv.mean((0, 1))) ** 2).sum((0, 1))
    return net, {"val_r2_clean": r2.tolist(), "n_train": len(Vt), "n_val": n_val, "final_loss": float(loss.detach())}


def load(path: Path = OUT / "model.pt") -> InverseNet:
    net = InverseNet()
    net.load_state_dict(torch.load(path, map_location="cpu"))
    net.eval()
    return net


def nn_velocity_to_command(vel: np.ndarray, net: InverseNet) -> np.ndarray:
    with torch.no_grad():
        c = net(torch.tensor(vel[None], dtype=torch.float32))[0].numpy().astype(np.float64)
    return np.clip(c, LO, HI)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=600)
    p.add_argument("--workers", type=int, default=10)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "babble.npz"
    t0 = time.time()
    if cache.exists():
        z = np.load(cache)
        data = [{"vel": v, "cmd": c} for v, c in zip(z["vel"], z["cmd"])]
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            runs = list(ex.map(babble_job, range(10_000, 10_000 + args.n)))
        data = [r for r in runs if not r["fell"]]
        np.savez_compressed(cache, vel=np.stack([d["vel"] for d in data]), cmd=np.stack([d["cmd"] for d in data]))
        print(f"babbled {len(data)}/{len(runs)} walks in {time.time() - t0:.0f}s", flush=True)
    net, info = train(data, args.steps)
    torch.save(net.state_dict(), out / "model.pt")
    info["seconds"] = time.time() - t0
    (out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
