"""Control: estimate root motion from GT joint trajectories only (no images).

Measures how much of the walked path is already implied by the gait itself, i.e. what
any video model could get without seeing the floor. Uses the same split as V2M-Ego.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.temporal_encoder import TemporalTransformer  # noqa: E402
from models.video_to_motion_ego import integrate_deltas  # noqa: E402
from training.ego_dataset import load_targets, se2_inv, se2_mul, split_sequences  # noqa: E402
from training.train_v2m_ego import aggregate, sequence_metrics  # noqa: E402


class JointsToRoot(nn.Module):
    def __init__(self, n_in: int, dim: int = 128):
        super().__init__()
        self.inp = nn.Linear(2 * n_in, dim)
        self.temporal = TemporalTransformer(dim=dim, nhead=4, nlayers=3)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))

    def forward(self, q: torch.Tensor) -> torch.Tensor:
        dq = torch.cat([q[:, :1] * 0, q[:, 1:] - q[:, :-1]], 1) * 30.0
        return self.out(self.temporal(self.inp(torch.cat([q, dq], -1))))


def tensors(seqs):
    tg = [load_targets(s, "hand_floor") for s in seqs]
    q = torch.tensor(np.stack([t["joint"] for t in tg]))
    v = torch.tensor(np.stack([t["root_vel"] for t in tg]))
    traj = torch.tensor(np.stack([se2_mul(se2_inv(t["pose"][:1]), t["pose"]) for t in tg]), dtype=torch.float32)
    return tg, q, v, traj


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=REPO_ROOT / "datasets/motions/POL-WALK-EGO")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3000)
    args = p.parse_args()
    torch.manual_seed(0)
    torch.set_num_threads(4)
    split = split_sequences(args.data)
    _, q_tr, v_tr, traj_tr = tensors(split["train"])
    tg_va, q_va, _, _ = tensors(split["val"])
    tg_te, q_te, _, _ = tensors(split["test"])
    fps = 30.0
    mean_vel = v_tr.reshape(-1, 3).mean(0).numpy()
    model = JointsToRoot(q_tr.shape[-1])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    best, best_state = float("inf"), None

    def evaluate(q, tg):
        model.eval()
        with torch.no_grad():
            vel = model(q).numpy()
        rows = []
        for i, t in enumerate(tg):
            pred = {"root_vel": vel[i], "joint": t["joint"], "contact_logits": t["contact"] * 2 - 1,
                    "root_z": t["root_z"], "rel": t["rel"], "cam_vel": t["cam_vel"]}
            rows.append(sequence_metrics(pred, t, fps, mean_vel))
        return aggregate(rows)

    for ep in range(1, args.epochs + 1):
        model.train()
        idx = torch.randperm(len(q_tr))
        for b in range(0, len(idx), 8):
            j = idx[b : b + 8]
            pred = model(q_tr[j])
            loss = torch.mean((pred[:, 1:] - v_tr[j, 1:]) ** 2) * 2.0
            tr = integrate_deltas(pred / fps)
            loss = loss + torch.mean((tr[..., :2] - traj_tr[j, :, :2]) ** 2) * 10.0 + torch.mean(
                (torch.sin(tr[..., 2]) - torch.sin(traj_tr[j, :, 2])) ** 2 + (torch.cos(tr[..., 2]) - torch.cos(traj_tr[j, :, 2])) ** 2
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 50 == 0:
            val = evaluate(q_va, tg_va)
            if val["ate_m"] < best:
                best, best_state = val["ate_m"], {k: v.clone() for k, v in model.state_dict().items()}
            print(f"ep={ep} val ate={val['ate_m']:.4f} fpe={val['final_pos_err_m']:.3f} yaw={val['final_yaw_err_deg']:.1f}", flush=True)
    model.load_state_dict(best_state)
    test = evaluate(q_te, tg_te)
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": best_state, "n_in": q_tr.shape[-1], "mean_vel": mean_vel}, args.out / "model.pt")
    keep = ("ate_m", "final_pos_err_m", "final_yaw_err_deg", "ate_baseline_m", "final_pos_err_baseline_m", "final_yaw_err_baseline_deg")
    (args.out / "test_metrics.json").write_text(json.dumps({k: test[k] for k in keep}, indent=2))
    print("TEST", json.dumps({k: test[k] for k in keep}))


if __name__ == "__main__":
    main()
