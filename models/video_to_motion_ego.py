"""Video-to-Motion with camera / root motion decomposition (POL-WALK-EGO).

Per frame t the model predicts, besides joints and foot contacts:

  rel_t      robot planar pose in the camera heading frame  (x, y, yaw)
  cam_vel_t  camera planar motion C_{t-1}^{-1} C_t * fps     (vx, vy, wz) + vertical vz
  root_vel_t robot planar motion P_{t-1}^{-1} P_t * fps      (vx, vy, wz)
  root_z_t   trunk height

Planar poses compose as SE(2):  P_t = C_t ∘ rel_t, hence

  P_{t-1}^{-1} P_t = rel_{t-1}^{-1} ∘ (C_{t-1}^{-1} C_t) ∘ rel_t

which is used as a consistency loss between the three heads.
"""

from __future__ import annotations

import torch
from torch import nn

from models.motion_decoder import MotionDecoder
from models.temporal_encoder import TemporalTransformer
from models.vision_encoder import SmallCNNEncoder


def wrap(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def se2_compose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a ∘ b for (..., 3) poses (x, y, yaw)."""
    c, s = torch.cos(a[..., 2]), torch.sin(a[..., 2])
    x = a[..., 0] + c * b[..., 0] - s * b[..., 1]
    y = a[..., 1] + s * b[..., 0] + c * b[..., 1]
    return torch.stack([x, y, wrap(a[..., 2] + b[..., 2])], dim=-1)


def se2_inverse(a: torch.Tensor) -> torch.Tensor:
    c, s = torch.cos(a[..., 2]), torch.sin(a[..., 2])
    x = -(c * a[..., 0] + s * a[..., 1])
    y = -(-s * a[..., 0] + c * a[..., 1])
    return torch.stack([x, y, -a[..., 2]], dim=-1)


def integrate_deltas(deltas: torch.Tensor) -> torch.Tensor:
    """(B, T, 3) per-step deltas (delta[0] ignored) -> (B, T, 3) poses relative to frame 0."""
    pose = torch.zeros_like(deltas[:, 0])
    out = [pose]
    for t in range(1, deltas.shape[1]):
        pose = se2_compose(pose, deltas[:, t])
        out.append(pose)
    return torch.stack(out, dim=1)


class VideoToMotionEgo(nn.Module):
    def __init__(self, n_joints: int, dim: int = 256, nhead: int = 4, nlayers: int = 4, with_view: bool = False):
        super().__init__()
        self.appearance = SmallCNNEncoder(out_dim=dim, in_ch=3)
        # (I_{t-1}, I_t, I_t - I_{t-1}): lets the CNN see image motion (floor flow) directly.
        self.motion = SmallCNNEncoder(out_dim=dim, in_ch=9)
        self.fuse = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.temporal = TemporalTransformer(dim=dim, nhead=nhead, nlayers=nlayers)
        self.decoder = MotionDecoder(dim=dim, n_joints=n_joints)

        def head(k: int) -> nn.Module:
            return nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, k))

        self.rel = head(3)
        self.cam_vel = head(4)
        self.root_vel = head(3)
        self.root_z = head(1)
        # Demo image view: silhouette (u, v, log area) + camera height and elevation.
        self.view = head(5) if with_view else None

    def forward(self, frames: torch.Tensor) -> dict[str, torch.Tensor]:
        b, t, c, h, w = frames.shape
        prev = torch.cat([frames[:, :1], frames[:, :-1]], dim=1)
        pair = torch.cat([prev, frames, frames - prev], dim=2)
        fa = self.appearance(frames.reshape(b * t, c, h, w)).reshape(b, t, -1)
        fm = self.motion(pair.reshape(b * t, 3 * c, h, w)).reshape(b, t, -1)
        feat = self.temporal(self.fuse(torch.cat([fa, fm], dim=-1)))
        out = self.decoder(feat)
        out["rel"] = self.rel(feat)
        out["cam_vel"] = self.cam_vel(feat)
        out["root_vel"] = self.root_vel(feat)
        out["root_z"] = self.root_z(feat).squeeze(-1) + 0.12
        if self.view is not None:
            out["view"] = self.view(feat) + feat.new_tensor([0.5, 0.5, -3.0, 0.15, -0.1])
        return out


def angle_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    return torch.mean((torch.sin(pred) - torch.sin(gt)) ** 2 + (torch.cos(pred) - torch.cos(gt)) ** 2)


def ego_loss(pred: dict, batch: dict, fps: float, lambdas: dict[str, float]) -> tuple[torch.Tensor, dict[str, float]]:
    L: dict[str, torch.Tensor] = {}
    L["joint"] = torch.mean((pred["joint"] - batch["joint"]) ** 2)
    vj = pred["joint"][:, 1:] - pred["joint"][:, :-1]
    vg = batch["joint"][:, 1:] - batch["joint"][:, :-1]
    L["joint_vel"] = torch.mean((vj - vg) ** 2) * fps
    L["contact"] = nn.functional.binary_cross_entropy_with_logits(pred["contact_logits"], batch["contact"])
    L["root_z"] = torch.mean((pred["root_z"] - batch["root_z"]) ** 2) * 100.0

    rel_p, rel_g = pred["rel"], batch["rel"]
    L["rel"] = torch.mean((rel_p[..., :2] - rel_g[..., :2]) ** 2) + angle_loss(rel_p[..., 2], rel_g[..., 2])
    L["cam_vel"] = torch.mean((pred["cam_vel"][:, 1:] - batch["cam_vel"][:, 1:]) ** 2)
    L["root_vel"] = torch.mean((pred["root_vel"][:, 1:] - batch["root_vel"][:, 1:]) ** 2)

    # Consistency: robot motion composed from camera motion + relative poses.
    rel_prev, rel_cur = rel_p[:, :-1], rel_p[:, 1:]
    cam_d = pred["cam_vel"][:, 1:, [0, 1, 3]] / fps
    composed = se2_compose(se2_compose(se2_inverse(rel_prev), cam_d), rel_cur) * fps
    L["consistency"] = torch.mean((composed - pred["root_vel"][:, 1:]) ** 2)

    # Drift: integrate predicted root deltas over the clip vs GT relative trajectory.
    traj_p = integrate_deltas(pred["root_vel"] / fps)
    L["traj"] = torch.mean((traj_p[..., :2] - batch["traj"][..., :2]) ** 2) * 10.0 + angle_loss(
        traj_p[..., 2], batch["traj"][..., 2]
    )

    if "view" in pred and "view" in batch:
        vp, vg = pred["view"], batch["view"]
        L["view_uv"] = torch.mean((vp[..., :2] - vg[..., :2]) ** 2) * 100.0
        L["view_area"] = torch.mean((vp[..., 2] - vg[..., 2]) ** 2)
        L["cam_height"] = torch.mean((vp[..., 3] - vg[..., 3]) ** 2) * 100.0
        L["cam_el"] = torch.mean((vp[..., 4] - vg[..., 4]) ** 2) * 100.0
        cam_p = integrate_deltas(pred["cam_vel"][..., [0, 1, 3]] / fps)
        L["cam_traj"] = torch.mean((cam_p[..., :2] - batch["cam_traj"][..., :2]) ** 2) * 10.0 + angle_loss(
            cam_p[..., 2], batch["cam_traj"][..., 2]
        )

    total = sum(lambdas.get(k, 0.0) * v for k, v in L.items())
    stats = {k: float(v.detach().item()) for k, v in L.items()}
    stats["total"] = float(total.detach().item())
    return total, stats
