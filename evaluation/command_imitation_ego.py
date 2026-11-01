#!/usr/bin/env python3
"""Command-level imitation: the robot's own walking skill + "where to walk" inferred from images.

The body skill is the official walking controller (alpha_walking.onnx + BAM); no per-demo teacher
actions are used. What to do is inferred from the head-camera observation (Condition C):

  open loop  V2M-estimated root velocity -> walking command via an inverse model of the robot's own
             body (actual velocity -> command), fitted on train-split walks
  image CEM  test-time analysis-by-synthesis: per-segment command offsets are optimized so that the
             robot rendered from the (estimated) demo camera matches the (estimated) demo silhouette
             centroid / size; the targets come from image_target.npz (V2M view head)

Every condition writes eval_vs_demo.json / eval_paths.json / eval_rollouts.npz like
evaluate_ego_tracking.py, so render_compare_ego.py and the report scripts work unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from simulation.mujoco.capture_ego import SCENE_XML, simulate  # noqa: E402
from simulation.mujoco.image_centroid import mask_stats, project_point, robot_mask, set_free_camera  # noqa: E402
from evaluation.inverse_body_model import load as load_inverse_nn  # noqa: E402
from evaluation.inverse_body_model import nn_velocity_to_command  # noqa: E402
from training.ego_dataset import integrate, quat_yaw, se2_inv, se2_mul, step_deltas, wrap  # noqa: E402

ONNX = REPO / "work/local/scratch/policies/alpha_walking.onnx"
DEMO = REPO / "datasets/motions/POL-WALK-EGO"
FPS, SECONDS, WARMUP = 30.0, 5.0, 0.6
SMOOTH = 15
N_SEG = 3
CEM_SIGMA = np.array([0.06, 0.02, 0.15])  # vx, vy, yaw offsets per segment


def smooth(x: np.ndarray, k: int = SMOOTH) -> np.ndarray:
    pad = np.pad(x, ((k // 2, k - 1 - k // 2), (0, 0)), mode="edge")
    ker = np.ones(k) / k
    return np.stack([np.convolve(pad[:, j], ker, mode="valid") for j in range(x.shape[1])], -1)


def planar(root_pos: np.ndarray, root_quat: np.ndarray) -> np.ndarray:
    return np.stack([root_pos[:, 0], root_pos[:, 1], quat_yaw(root_quat)], -1)


def fit_inverse_model(train: list[str]) -> dict:
    """Linear map smoothed body velocity (t + lag) -> walking command (t), from the robot's own walks."""
    best = None
    for lag in range(0, 16, 3):
        X, Y = [], []
        for name in train:
            r = np.load(DEMO / name / "root_poses.npz")
            v = smooth(step_deltas(planar(r["root_pos"], r["root_quat_wxyz"]), FPS))
            cmd = np.load(DEMO / name / "commands.npz")["twist"]
            n = len(v) - lag
            X.append(v[lag:]), Y.append(cmd[:n])
        X, Y = np.concatenate(X), np.concatenate(Y)
        Xb = np.concatenate([X, np.ones((len(X), 1))], 1)
        W, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
        r2 = 1 - np.sum((Xb @ W - Y) ** 2, 0) / np.sum((Y - Y.mean(0)) ** 2, 0)
        if best is None or r2.mean() > best["r2"].mean():
            best = {"W": W, "lag": lag, "r2": r2, "mean_cmd": Y.mean(0)}
    return best


def velocity_to_command(vel: np.ndarray, inv: dict) -> np.ndarray:
    v = smooth(vel)
    v = np.concatenate([v[inv["lag"]:], np.repeat(v[-1:], inv["lag"], 0)], 0)
    cmd = np.concatenate([v, np.ones((len(v), 1))], 1) @ inv["W"]
    return np.clip(cmd, [-0.1, -0.15, -0.8], [0.5, 0.15, 0.8])


def run(cmd: np.ndarray, sess) -> dict:
    n = len(cmd)
    return simulate(ONNX, [], SECONDS, FPS, WARMUP, command_fn=lambda t: cmd[min(int(t * FPS), n - 1)], sess=sess)


def to_frame(qpos: np.ndarray, target0: np.ndarray) -> np.ndarray:
    """Re-express a rollout so that its first planar pose becomes ``target0``."""
    own = planar(qpos[:, 0:3], qpos[:, 3:7])
    T = se2_mul(target0, se2_inv(own[0]))
    new = se2_mul(np.repeat(T[None], len(own), 0), own)
    out = qpos.copy()
    out[:, 0], out[:, 1] = new[:, 0], new[:, 1]
    dq = np.array([np.cos(T[2] / 2), 0, 0, np.sin(T[2] / 2)])
    w1, x1, y1, z1 = dq
    w2, x2, y2, z2 = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    out[:, 3:7] = np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)
    return out


class ImageScorer:
    def __init__(self, target_npz: Path, res: int = 64):
        self.tg = np.load(target_npz)
        self.fovy = float(np.asarray(self.tg["fovy_deg"]).reshape(-1)[0])
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
        self.data = mujoco.MjData(self.model)
        self.trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")
        self.r = mujoco.Renderer(self.model, height=res, width=res)
        self.r.enable_segmentation_rendering()
        self.cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.cam)

    def __call__(self, qpos: np.ndarray, every: int = 3) -> float:
        """Mean (centroid distance + 0.5 |log size ratio|) over frames; robot frame 0 = reference start (identity)."""
        q = to_frame(qpos, np.zeros(3))
        errs = []
        for t in range(0, min(len(q), len(self.tg["uv"])), every):
            self.data.qpos[:] = q[t]
            mujoco.mj_forward(self.model, self.data)
            look, aed = self.tg["lookat"][t], self.tg["aed"][t]
            set_free_camera(self.cam, look, aed)
            self.r.update_scene(self.data, camera=self.cam)
            u, v, area = mask_stats(robot_mask(self.r.render(), self.model))
            if area <= 0:
                uv = project_point(self.data.subtree_com[self.trunk], look, aed, self.fovy)
                errs.append(float(np.linalg.norm(uv - self.tg["uv"][t])) + 0.5)
            else:
                e_s = abs(0.5 * np.log((area + 1e-4) / (self.tg["area"][t] + 1e-4)))
                errs.append(float(np.hypot(u - self.tg["uv"][t, 0], v - self.tg["uv"][t, 1])) + 0.5 * e_s)
        return float(np.mean(errs))


class Objective:
    """w_img * image error / 0.03 + w_root * mean distance to an estimated root path / 0.03 (both from video)."""

    def __init__(self, img_target: str | None, root_path: list | None, w_img: float, w_root: float):
        self.img = ImageScorer(Path(img_target)) if img_target and w_img > 0 else None
        self.root = np.asarray(root_path) if root_path is not None and w_root > 0 else None
        self.w_img, self.w_root = w_img, w_root

    def __call__(self, qpos: np.ndarray) -> float:
        cost = 0.0
        if self.img is not None:
            cost += self.w_img * self.img(qpos) / 0.03
        if self.root is not None:
            own = planar(*(lambda q: (q[:, 0:3], q[:, 3:7]))(to_frame(qpos, np.zeros(3))))
            n = min(len(own), len(self.root))
            cost += self.w_root * float(np.mean(np.linalg.norm(own[:n, :2] - self.root[:n, :2], axis=1))) / 0.03
        return cost


def with_offsets(base: np.ndarray, theta: np.ndarray) -> np.ndarray:
    seg = np.minimum((np.arange(len(base)) * N_SEG) // len(base), N_SEG - 1)
    return np.clip(base + theta.reshape(N_SEG, 3)[seg], [-0.1, -0.15, -0.8], [0.5, 0.15, 0.8])


def cem(base: np.ndarray, scorer, sess, iters: int, pop: int, n_elite: int, seed: int) -> tuple[np.ndarray, list]:
    rng = np.random.default_rng(seed)
    mu, sd = np.zeros(N_SEG * 3), np.tile(CEM_SIGMA, N_SEG)
    hist, best = [], (np.inf, mu.copy())
    for _ in range(iters):
        cands = [mu.copy()] + [mu + sd * rng.standard_normal(mu.shape) for _ in range(pop - 1)]
        scores = []
        for th in cands:
            sim = run(with_offsets(base, th), sess)
            fell = float(sim["tilt"].max()) > 45.0
            scores.append(scorer(sim["qpos"]) + (10.0 if fell else 0.0))
        order = np.argsort(scores)
        if scores[order[0]] < best[0]:
            best = (scores[order[0]], cands[order[0]].copy())
        elite = np.stack([cands[i] for i in order[:n_elite]])
        mu, sd = elite.mean(0), np.maximum(elite.std(0), 0.15 * np.tile(CEM_SIGMA, N_SEG))
        hist.append(float(scores[order[0]]))
    return with_offsets(base, best[1]), hist


def job(spec: dict) -> dict:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(ONNX), providers=["CPUExecutionProvider"])
    base = np.asarray(spec["base"])
    hist = []
    if spec.get("cem_target") or spec.get("root_path") is not None:
        scorer = Objective(spec.get("cem_target"), spec.get("root_path"), spec.get("w_img", 1.0), spec.get("w_root", 0.0))
        cmd, hist = cem(base, scorer, sess, spec["iters"], spec["pop"], spec["elite"], spec["seed"])
    else:
        cmd = base
    sim = run(cmd, sess)
    out = {"clip": spec["clip"], "cond": spec["cond"], "qpos": sim["qpos"], "q": sim["q"], "tilt": sim["tilt"],
           "cmd": cmd, "cem_hist": hist}
    est_target = spec.get("score_target")
    if est_target:
        out["img_err_est"] = ImageScorer(Path(est_target))(sim["qpos"], every=1)
    return out


def summarize(results: list[dict], out_dir: Path, label: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, paths, rollouts = [], [], {}
    for r in sorted(results, key=lambda x: x["clip"]):
        name = r["clip"]
        demo = np.load(DEMO / name / "root_poses.npz")
        dq = np.load(DEMO / name / "joint_positions.npz")["joint_positions"]
        dpose = planar(demo["root_pos"], demo["root_quat_wxyz"])
        q = to_frame(r["qpos"], dpose[0])
        n = min(len(q), len(dpose))
        own = planar(q[:n, 0:3], q[:n, 3:7])
        err = np.linalg.norm(own[:, :2] - dpose[:n, :2], axis=1)
        fell = bool(r["tilt"].max() > 45.0)
        rows.append({
            "clip": name, "success": float(not fell), "fell": float(fell), "duration_s": n / FPS,
            "ate_vs_demo_m": float(err.mean()), "final_err_vs_demo_m": float(err[-1]),
            "final_yaw_err_deg": float(np.degrees(abs(wrap(own[-1, 2] - dpose[n - 1, 2])))),
            "joint_mae_vs_demo": float(np.mean(np.abs(r["q"][:n] - dq[:n]))),
            "demo_travel_m": float(np.linalg.norm(dpose[n - 1, :2] - dpose[0, :2])),
            **({"img_err_est": r["img_err_est"]} if "img_err_est" in r else {}),
        })
        paths.append({"clip": name, "exec": own[:, :2].tolist(), "demo": dpose[:n, :2].tolist(), "ref": own[:, :2].tolist(),
                      "cmd": np.asarray(r["cmd"]).tolist(), "cem_hist": r["cem_hist"]})
        rollouts[name] = q[:n]
        rollouts[name + "__valid"] = np.ones(n, bool)
    keys = [k for k in rows[0] if k != "clip"]
    summary = {k: float(np.mean([r[k] for r in rows])) for k in keys} | {"n": len(rows)}
    (out_dir / "eval_vs_demo.json").write_text(json.dumps({"label": label, "summary": summary, "clips": rows}, indent=2))
    (out_dir / "eval_paths.json").write_text(json.dumps(paths))
    np.savez_compressed(out_dir / "eval_rollouts.npz", **rollouts)
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=REPO / "work/local/runs/ego/cmd")
    p.add_argument("--preds", type=Path, default=REPO / "work/local/runs/ego/eval/v2m_preds_hand_floor_C_view.npz")
    p.add_argument("--preds-plain", type=Path, default=REPO / "work/local/runs/ego/eval/v2m_preds_hand_plain_C.npz")
    p.add_argument("--est-targets", type=Path, default=REPO / "datasets/motions/POL-WALK-EGO-V2M-EST-IMG")
    p.add_argument("--gt-targets", type=Path, default=REPO / "datasets/motions/POL-WALK-EGO-V2M-EST-IMGGT")
    p.add_argument("--conds", nargs="+", default=["gt_cmd", "mean_cmd", "v2m_open", "v2m_open_plain", "v2m_cem_est",
                                                   "mean_cem_est", "v2m_cem_gt", "v2m_cem_root", "v2m_cem_root_img",
                                                   "v2m_cem_root_imggt"])
    p.add_argument("--inverse-nn", type=Path, default=REPO / "work/local/runs/ego/cmd/inverse_nn/model.pt")
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--pop", type=int, default=12)
    p.add_argument("--elite", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--max-clips", type=int, default=18)
    args = p.parse_args()

    split = json.loads((DEMO / "split.json").read_text())
    inv = fit_inverse_model(split["train"])
    print(f"inverse model: lag={inv['lag']} r2={np.round(inv['r2'], 3)}", flush=True)
    preds, preds_plain = np.load(args.preds), np.load(args.preds_plain)
    inv_nn = load_inverse_nn(args.inverse_nn) if args.inverse_nn.exists() else None
    n = int(SECONDS * FPS)

    specs = []
    for name in split["test"][: args.max_clips]:
        est_t = str(args.est_targets / name / "image_target.npz")
        gt_t = str(args.gt_targets / name / "image_target.npz")
        base = {
            "gt_cmd": np.load(DEMO / name / "commands.npz")["twist"],
            "mean_cmd": np.repeat(inv["mean_cmd"][None], n, 0),
            "v2m": velocity_to_command(preds[f"{name}__root_vel"].astype(np.float64), inv),
            "v2m_plain": velocity_to_command(preds_plain[f"{name}__root_vel"].astype(np.float64), inv),
        }
        demo_r = np.load(DEMO / name / "root_poses.npz")
        gt_vel = step_deltas(planar(demo_r["root_pos"], demo_r["root_quat_wxyz"]), FPS)
        base["gtvel"] = velocity_to_command(gt_vel, inv)
        if inv_nn is not None:
            base["gtvel_nn"] = nn_velocity_to_command(gt_vel, inv_nn)
            base["v2m_nn"] = nn_velocity_to_command(preds[f"{name}__root_vel"].astype(np.float64), inv_nn)
        root_est = integrate(preds[f"{name}__root_vel"].astype(np.float64), FPS).tolist()
        table = {
            "gt_cmd": dict(base=base["gt_cmd"]), "mean_cmd": dict(base=base["mean_cmd"]),
            "v2m_open": dict(base=base["v2m"]), "v2m_open_plain": dict(base=base["v2m_plain"]),
            "v2m_cem_est": dict(base=base["v2m"], cem_target=est_t),
            "mean_cem_est": dict(base=base["mean_cmd"], cem_target=est_t),
            "v2m_cem_gt": dict(base=base["v2m"], cem_target=gt_t),
            "v2m_cem_root": dict(base=base["v2m"], root_path=root_est, w_img=0.0, w_root=1.0),
            "v2m_cem_root_img": dict(base=base["v2m"], cem_target=est_t, root_path=root_est, w_img=1.0, w_root=1.0),
            "v2m_cem_root_imggt": dict(base=base["v2m"], cem_target=gt_t, root_path=root_est, w_img=1.0, w_root=1.0),
            "gtvel_open": dict(base=base["gtvel"]),
        }
        if inv_nn is not None:
            table |= {
                "gtvel_open_nn": dict(base=base["gtvel_nn"]), "v2m_open_nn": dict(base=base["v2m_nn"]),
                "v2m_cem_root_nn": dict(base=base["v2m_nn"], root_path=root_est, w_img=0.0, w_root=1.0),
                "v2m_cem_root_img_nn": dict(base=base["v2m_nn"], cem_target=est_t, root_path=root_est, w_img=1.0, w_root=1.0),
                "v2m_cem_root_imggt_nn": dict(base=base["v2m_nn"], cem_target=gt_t, root_path=root_est, w_img=1.0, w_root=1.0),
            }
        for c in args.conds:
            spec = dict(table[c])
            spec["base"] = spec["base"].tolist()
            specs.append({"clip": name, "cond": c, "score_target": est_t,
                          "iters": args.iters, "pop": args.pop, "elite": args.elite, "seed": 0, **spec})

    results: dict[str, list] = {c: [] for c in args.conds}
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(job, specs):
            results[r["cond"]].append(r)
            print(f"{r['cond']:16s} {r['clip']} done" + (f" cem {np.round(r['cem_hist'], 3)}" if r["cem_hist"] else ""), flush=True)
    table = {}
    for c, rs in results.items():
        table[c] = summarize(rs, args.out / c, c)
        print(c, json.dumps({k: round(v, 3) for k, v in table[c].items()}), flush=True)
    prev = json.loads((args.out / "summary.json").read_text())["conditions"] if (args.out / "summary.json").exists() else {}
    table = prev | table
    (args.out / "summary.json").write_text(json.dumps({"inverse_model": {"lag": inv["lag"], "r2": inv["r2"].tolist()},
                                                      "conditions": table}, indent=2))


if __name__ == "__main__":
    main()
