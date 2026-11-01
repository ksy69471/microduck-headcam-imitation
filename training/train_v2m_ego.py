"""Train VideoToMotionEgo on POL-WALK-EGO (one visual variant, one input source)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.video_to_motion_ego import VideoToMotionEgo, ego_loss  # noqa: E402
from training.ego_dataset import EgoDataset, integrate, se2_inv, se2_mul, split_sequences, wrap  # noqa: E402

LAMBDAS = {
    "joint": 1.0,
    "joint_vel": 0.5,
    "contact": 0.05,
    "root_z": 0.2,
    "rel": 1.0,
    "cam_vel": 1.0,
    "root_vel": 2.0,
    "consistency": 0.5,
    "traj": 1.0,
    "view_uv": 1.0,
    "view_area": 0.5,
    "cam_height": 0.5,
    "cam_el": 0.5,
    "cam_traj": 1.0,
}


@torch.no_grad()
def predict(model: VideoToMotionEgo, ds: EgoDataset, device) -> list[dict]:
    model.eval()
    outs = []
    for i in range(len(ds)):
        item = ds[i]
        pred = model(item["frames"][None].to(device))
        outs.append({k: v[0].float().cpu().numpy() for k, v in pred.items()})
    return outs


def sequence_metrics(pred: dict, tg: dict, fps: float, mean_vel: np.ndarray) -> dict:
    n = min(len(pred["joint"]), len(tg["joint"]))
    gt_traj = se2_mul(se2_inv(tg["pose"][:1]), tg["pose"][:n])
    est = integrate(pred["root_vel"][:n], fps)
    rel = pred["rel"][:n]
    cam_d = pred["cam_vel"][:n][:, [0, 1, 3]]
    comp = np.concatenate([np.zeros((1, 3)), se2_mul(se2_mul(se2_inv(rel[:-1]), cam_d[1:] / fps), rel[1:]) * fps], 0)
    est_comp = integrate(comp, fps)
    base = integrate(np.repeat(mean_vel[None], n, 0), fps)
    path_len = float(np.sum(np.linalg.norm(np.diff(gt_traj[:, :2], axis=0), axis=1)))

    def ate(tr):
        return float(np.mean(np.linalg.norm(tr[:, :2] - gt_traj[:, :2], axis=1)))

    def fpe(tr):
        return float(np.linalg.norm(tr[-1, :2] - gt_traj[-1, :2]))

    def fye(tr):
        return float(np.degrees(abs(wrap(tr[-1, 2] - gt_traj[-1, 2]))))

    contact_pred = pred["contact_logits"][:n] > 0
    view = {}
    if "view" in pred and "view" in tg:
        vp, vg = pred["view"][:n], tg["view"][:n]
        gt_cam = se2_mul(se2_inv(tg["cam_pose"][:1]), tg["cam_pose"][:n])
        est_cam = integrate(pred["cam_vel"][:n][:, [0, 1, 3]], fps)
        view = {
            "view_uv_err": float(np.mean(np.linalg.norm(vp[:, :2] - vg[:, :2], axis=1))),
            "view_logarea_mae": float(np.mean(np.abs(vp[:, 2] - vg[:, 2]))),
            "cam_height_mae": float(np.mean(np.abs(vp[:, 3] - vg[:, 3]))),
            "cam_el_mae_deg": float(np.degrees(np.mean(np.abs(vp[:, 4] - vg[:, 4])))),
            "cam_ate_m": float(np.mean(np.linalg.norm(est_cam[:, :2] - gt_cam[:, :2], axis=1))),
        }
    return view | {
        "joint_mae": float(np.mean(np.abs(pred["joint"][:n] - tg["joint"][:n]))),
        "contact_acc": float(np.mean(contact_pred == (tg["contact"][:n] > 0.5))),
        "root_z_mae": float(np.mean(np.abs(pred["root_z"][:n] - tg["root_z"][:n]))),
        "root_vel_rmse": np.sqrt(np.mean((pred["root_vel"][1:n] - tg["root_vel"][1:n]) ** 2, 0)).tolist(),
        "cam_vel_rmse": np.sqrt(np.mean((pred["cam_vel"][1:n] - tg["cam_vel"][1:n]) ** 2, 0)).tolist(),
        "rel_xy_mae": float(np.mean(np.linalg.norm(rel[:, :2] - tg["rel"][:n, :2], axis=1))),
        "ate_m": ate(est),
        "final_pos_err_m": fpe(est),
        "final_yaw_err_deg": fye(est),
        "ate_composed_m": ate(est_comp),
        "ate_baseline_m": ate(base),
        "final_pos_err_baseline_m": fpe(base),
        "final_yaw_err_baseline_deg": fye(base),
        "path_len_m": path_len,
        "gt_final_yaw_deg": float(np.degrees(gt_traj[-1, 2])),
    }


def aggregate(rows: list[dict]) -> dict:
    out = {}
    for k in rows[0]:
        vals = np.asarray([r[k] for r in rows], dtype=np.float64)
        out[k] = vals.mean(0).tolist() if vals.ndim > 1 else float(vals.mean())
    return out


def evaluate(model, ds: EgoDataset, device, mean_vel: np.ndarray) -> tuple[dict, list[dict], list[dict]]:
    preds = predict(model, ds, device)
    rows = [sequence_metrics(p, t, ds.fps, mean_vel) for p, t in zip(preds, ds.targets)]
    return aggregate(rows), rows, preds


def build_model(ckpt: dict) -> VideoToMotionEgo:
    return VideoToMotionEgo(n_joints=ckpt["n_joints"], dim=ckpt["dim"], nhead=ckpt["nhead"], nlayers=ckpt["nlayers"],
                            with_view=ckpt.get("with_view", False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=REPO_ROOT / "datasets/motions/POL-WALK-EGO")
    p.add_argument("--variant", default="hand_floor")
    p.add_argument("--source", default="C", choices=["A", "C"], help="A=teacher video, C=screen via head camera")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--time-budget-s", type=float, default=0.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--clip-len", type=int, default=150)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--nlayers", type=int, default=4)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--with-view", action="store_true", help="Also predict demo-image silhouette + camera height/elevation")
    p.add_argument("--init", type=Path, default=None, help="Initialize from a checkpoint (missing heads stay random)")
    p.add_argument("--lambda", dest="lambda_over", action="append", default=[], metavar="NAME=VALUE",
                   help="override a loss weight, e.g. --lambda cam_traj=5")
    args = p.parse_args()
    for kv in args.lambda_over:
        k, v = kv.split("=")
        LAMBDAS[k] = float(v)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    split = split_sequences(args.data)
    t_load = time.time()
    train_ds = EgoDataset(split["train"], args.variant, args.source, args.image_size, args.clip_len, augment=True)
    val_ds = EgoDataset(split["val"], args.variant, args.source, args.image_size)
    print(f"loaded train={len(train_ds)} val={len(val_ds)} in {time.time() - t_load:.0f}s", flush=True)
    fps = train_ds.fps
    mean_vel = np.mean(np.concatenate([t["root_vel"] for t in train_ds.targets]), 0)

    n_joints = int(train_ds.targets[0]["joint"].shape[1])
    model = VideoToMotionEgo(n_joints, dim=args.dim, nhead=args.nhead, nlayers=args.nlayers, with_view=args.with_view).to(device)
    if args.init:
        missing, _ = model.load_state_dict(torch.load(args.init, map_location=device, weights_only=False)["model"], strict=False)
        print(f"init from {args.init}; new params: {sorted({k.split('.')[0] for k in missing})}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=0)

    args.out.mkdir(parents=True, exist_ok=True)
    cfg = {**{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "lambdas": LAMBDAS,
           "n_train": len(train_ds), "n_val": len(val_ds), "mean_train_root_vel": mean_vel.tolist()}
    (args.out / "config.json").write_text(json.dumps(cfg, indent=2))
    hist_path = args.out / "history.jsonl"
    hist_path.write_text("")

    def ckpt(epoch: int, val: dict) -> dict:
        return {"model": model.state_dict(), "n_joints": n_joints, "dim": args.dim, "nhead": args.nhead,
                "nlayers": args.nlayers, "epoch": epoch, "val": val, "variant": args.variant,
                "source": args.source, "image_size": args.image_size, "mean_train_root_vel": mean_vel.tolist(),
                "with_view": args.with_view}

    best = float("inf")
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        if args.time_budget_s and time.time() - t0 > args.time_budget_s:
            print(f"time budget reached at epoch {epoch}", flush=True)
            break
        model.train()
        run = []
        for batch in loader:
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pred = model(batch["frames"])
            pred = {k: v.float() for k, v in pred.items()}
            loss, stats = ego_loss(pred, batch, fps, LAMBDAS)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run.append(stats)
        sched.step()
        row = {"epoch": epoch, "elapsed_s": time.time() - t0,
               "train": {k: float(np.mean([r[k] for r in run])) for k in run[0]}}
        if epoch % args.eval_every == 0 or epoch == 1:
            val, _, _ = evaluate(model, val_ds, device, mean_vel)
            row["val"] = val
            score = val["joint_mae"] / 0.02 + val["ate_m"] / 0.05
            if "view_uv_err" in val:
                score += val["view_uv_err"] / 0.02 + val["cam_ate_m"] / 0.05
            if score < best:
                best = score
                torch.save(ckpt(epoch, val), args.out / "best.pt")
            print(f"ep={epoch} loss={row['train']['total']:.4f} val joint_mae={val['joint_mae']:.4f} "
                  f"ate={val['ate_m']:.3f} (base {val['ate_baseline_m']:.3f}) fpe={val['final_pos_err_m']:.3f} "
                  f"yaw={val['final_yaw_err_deg']:.1f}"
                  + (f" uv={val['view_uv_err']:.3f} cam_ate={val['cam_ate_m']:.3f} h={val['cam_height_mae']:.3f}"
                     f" el={val['cam_el_mae_deg']:.1f}" if "view_uv_err" in val else "")
                  + f" t={row['elapsed_s'] / 60:.1f}m", flush=True)
        with hist_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
    torch.save(ckpt(epoch, row.get("val", {})), args.out / "last.pt")
    print("TRAIN_OK", flush=True)


if __name__ == "__main__":
    main()
