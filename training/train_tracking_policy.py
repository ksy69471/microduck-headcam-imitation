"""Train PPO motion-tracking policy (PLAN §9).

Saves checkpoints + JSONL metrics. Pair with evaluation/visualize_tracking_stages.py
for stage videos.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.mujoco.onnx_teacher import OnnxWalkingTeacher  # noqa: E402
from simulation.mujoco.tracking_env import (  # noqa: E402
    MotionTrackingEnv,
    load_motion_library,
)
from training.ppo import ActorCritic, PPO, PPOConfig, compute_gae  # noqa: E402


def collect_rollout(
    envs: list[MotionTrackingEnv],
    agent: PPO,
    steps_per_env: int,
    *,
    bc_mix: float = 0.0,
    teachers: list[OnnxWalkingTeacher] | None = None,
) -> dict[str, np.ndarray]:
    n_env = len(envs)
    obs_buf, act_buf, logp_buf, rew_buf, done_buf, val_buf = [], [], [], [], [], []
    teacher_buf = []
    ep_returns, ep_lens, successes, falls = [], [], [], []

    obs = [e.reset() for e in envs]
    if teachers:
        for t in teachers:
            t.reset()
    for _ in range(steps_per_env):
        actions = []
        logps = []
        vals = []
        teacher_acts = []
        for i, o in enumerate(obs):
            a, lp, v = agent.act_np(o)
            if teachers is not None:
                t_act = teachers[i].act_env(envs[i])
            else:
                t_act = envs[i].teacher_action()
                if t_act is None:
                    t_act = a
            teacher_acts.append(np.asarray(t_act, dtype=np.float32))
            if bc_mix > 0 and np.random.random() < bc_mix:
                a = teacher_acts[-1].copy()
                with torch.no_grad():
                    ot = torch.as_tensor(o, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    at = torch.as_tensor(a, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    lp = float(agent.model.dist(ot).log_prob(at).sum(-1).item())
            actions.append(a)
            logps.append(lp)
            vals.append(v)
        next_obs = []
        for i, env in enumerate(envs):
            o2, r, done, info = env.step(actions[i])
            obs_buf.append(obs[i])
            act_buf.append(actions[i])
            logp_buf.append(logps[i])
            rew_buf.append(r)
            done_buf.append(float(done))
            val_buf.append(vals[i])
            teacher_buf.append(teacher_acts[i])
            if done:
                ep_returns.append(float(info["ep_return"]))
                ep_lens.append(int(env.steps))
                successes.append(float(info.get("success", False)))
                falls.append(float(info.get("fell", False)))
                o2 = env.reset()
                if teachers:
                    teachers[i].reset()
            next_obs.append(o2)
        obs = next_obs

    # Bootstrap values
    last_values = []
    for o in obs:
        _, _, v = agent.act_np(o)
        last_values.append(v)

    obs_arr = np.asarray(obs_buf, dtype=np.float32)
    act_arr = np.asarray(act_buf, dtype=np.float32)
    logp_arr = np.asarray(logp_buf, dtype=np.float32)
    rew_arr = np.asarray(rew_buf, dtype=np.float32)
    done_arr = np.asarray(done_buf, dtype=np.float32)
    val_arr = np.asarray(val_buf, dtype=np.float32)

    # GAE per-env interleaved: reshape (T, N) 
    t_steps = steps_per_env
    rew = rew_arr.reshape(t_steps, n_env)
    done = done_arr.reshape(t_steps, n_env)
    val = val_arr.reshape(t_steps, n_env)
    adv = np.zeros_like(rew)
    ret = np.zeros_like(rew)
    for i in range(n_env):
        a, r = compute_gae(
            rew[:, i],
            val[:, i],
            done[:, i],
            float(last_values[i]),
            agent.cfg.gamma,
            agent.cfg.gae_lambda,
        )
        adv[:, i] = a
        ret[:, i] = r

    return {
        "obs": obs_arr,
        "act": act_arr,
        "logp": logp_arr,
        "adv": adv.reshape(-1).astype(np.float32),
        "ret": ret.reshape(-1).astype(np.float32),
        "teacher": np.asarray(teacher_buf, dtype=np.float32),
        "ep_returns": np.asarray(ep_returns, dtype=np.float32),
        "ep_lens": np.asarray(ep_lens, dtype=np.float32),
        "success_rate": float(np.mean(successes)) if successes else 0.0,
        "fall_rate": float(np.mean(falls)) if falls else 0.0,
    }


def bc_update(agent: PPO, batch: dict[str, np.ndarray], weight: float = 1.0) -> float:
    """Behavioral cloning toward teacher actions (auxiliary loss)."""
    if weight <= 0:
        return 0.0
    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=agent.device)
    teacher = torch.as_tensor(batch["teacher"], dtype=torch.float32, device=agent.device)
    n = obs.shape[0]
    mb = min(agent.cfg.minibatch, n)
    losses = []
    for start in range(0, n, mb):
        o = obs[start : start + mb]
        t = teacher[start : start + mb]
        mu = agent.model.mu(agent.model.trunk(o))
        loss = weight * torch.mean((mu - t) ** 2)
        agent.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.model.parameters(), agent.cfg.max_grad_norm)
        agent.opt.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def eval_policy(
    env: MotionTrackingEnv,
    agent: PPO,
    n_episodes: int = 8,
    deterministic: bool = True,
) -> dict:
    rets, lens, succ, falls, maes, tilts, xy_errs, travels, img_errs = [], [], [], [], [], [], [], [], []
    for ep in range(n_episodes):
        obs = env.reset(clip_idx=ep % len(env.motions), start_frac=0.0)
        done = False
        joint_maes, ep_xy, ep_img = [], [], []
        while not done:
            if deterministic:
                with torch.no_grad():
                    o = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
                    a = agent.model.dist(o).mean.squeeze(0).cpu().numpy()
            else:
                a, _, _ = agent.act_np(obs)
            obs, _r, done, info = env.step(a)
            joint_maes.append(info["joint_mae"])
            ep_xy.append(info["xy_err_ref"])
            if np.isfinite(info.get("img_centroid_err", np.nan)):
                ep_img.append(info["img_centroid_err"])
        xy_errs.append(float(np.mean(ep_xy)))
        img_errs.append(float(np.mean(ep_img)) if ep_img else float("nan"))
        rets.append(info["ep_return"])
        lens.append(env.steps)
        succ.append(float(info["success"]))
        falls.append(float(info["fell"]))
        maes.append(float(np.mean(joint_maes)))
        tilts.append(info["tilt_deg"])
        travels.append(float(info.get("travel_xy", 0.0)))
    return {
        "mean_return": float(np.mean(rets)),
        "mean_len": float(np.mean(lens)),
        "success_rate": float(np.mean(succ)),
        "fall_rate": float(np.mean(falls)),
        "joint_mae": float(np.mean(maes)),
        "final_tilt_deg": float(np.mean(tilts)),
        "xy_err_ref_m": float(np.mean(xy_errs)),
        "travel_xy_m": float(np.mean(travels)),
        "img_centroid_err": float(np.nanmean(img_errs)) if np.isfinite(img_errs).any() else float("nan"),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--motions", type=Path, default=REPO_ROOT / "datasets/motions/POL-WALK")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=8)
    p.add_argument("--rollout-steps", type=int, default=512)
    p.add_argument("--updates", type=int, default=2000)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=25)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--time-budget-s", type=float, default=0.0, help="Stop after N seconds (0=ignore)")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--bc-epochs", type=int, default=60, help="Closed-loop ONNX BC warm-start epochs")
    p.add_argument("--bc-mix-start", type=float, default=0.85, help="ONNX action mix at update 0")
    p.add_argument("--bc-mix-end", type=float, default=0.15, help="ONNX action mix after decay")
    p.add_argument("--bc-decay-updates", type=int, default=800)
    p.add_argument("--bc-aux-weight", type=float, default=0.25, help="Aux BC loss weight during PPO")
    p.add_argument("--teacher-reward", type=float, default=0.2)
    p.add_argument(
        "--onnx-teacher",
        type=Path,
        default=REPO_ROOT / "work/local/scratch/policies/alpha_walking.onnx",
    )
    p.add_argument(
        "--no-onnx-teacher",
        action="store_true",
        help="Observation-derived only: no ONNX BC/DAgger, no teacher-action reward/init",
    )
    p.add_argument("--root-cond", action="store_true", help="Add reference root motion to the policy observation")
    p.add_argument("--root-xy-weight", type=float, default=None, help="Override root_xy reward weight")
    p.add_argument(
        "--travel-weight",
        type=float,
        default=0.0,
        help="Reward weight for matching forward progress along reference heading",
    )
    p.add_argument("--xy-sigma", type=float, default=0.08)
    p.add_argument("--heading-weight", type=float, default=0.0)
    p.add_argument("--joint-weight", type=float, default=None, help="Override joint reward weight")
    p.add_argument("--joint-sigma", type=float, default=0.5,
                   help="Scale of joint reward exp(-mean sq err / sigma); smaller = stricter imitation")
    p.add_argument("--upright-weight", type=float, default=None, help="Override upright reward weight")
    p.add_argument("--alive-weight", type=float, default=None, help="Override alive reward weight")
    p.add_argument(
        "--best-metric",
        choices=("success", "travel", "travel_x_succ"),
        default="success",
        help="Criterion for saving best.pt",
    )
    p.add_argument("--img-weight", type=float, default=0.0, help="Reward for matching the demo silhouette centroid/size in the demo camera image")
    p.add_argument("--img-sigma", type=float, default=0.15)
    p.add_argument("--img-every", type=int, default=2, help="Render the own silhouette every N policy steps")
    p.add_argument("--eval-motions", type=Path, default=None, help="Separate (held-out) clip library for eval")
    p.add_argument("--clip-names", type=Path, default=None, help="JSON list of sequence names to train on")
    p.add_argument("--eval-clip-names", type=Path, default=None, help="JSON list of sequence names to evaluate on")
    args = p.parse_args()

    if args.no_onnx_teacher:
        args.bc_epochs = 0
        args.bc_mix_start = 0.0
        args.bc_mix_end = 0.0
        args.bc_aux_weight = 0.0
        args.teacher_reward = 0.0
        args.onnx_teacher = Path("/dev/null/no_onnx_teacher.onnx")

    args.out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = args.out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    metrics_path = args.out / "metrics.jsonl"

    motions = load_motion_library(args.motions)
    eval_motions = load_motion_library(args.eval_motions) if args.eval_motions else motions
    if args.clip_names:
        keep = set(json.loads(args.clip_names.read_text()))
        motions = [m for m in motions if Path(m.path).name in keep]
    if args.eval_clip_names:
        keep = set(json.loads(args.eval_clip_names.read_text()))
        eval_motions = [m for m in eval_motions if Path(m.path).name in keep]
    if args.no_onnx_teacher:
        # Drop privileged ONNX residuals so reset/reward cannot use them.
        for m in [*motions, *eval_motions]:
            m.teacher_actions = None
    n_teacher = sum(1 for m in motions if m.teacher_actions is not None)
    print(
        f"loaded {len(motions)} clips, {n_teacher} with teacher_actions"
        + (" [OBS-ONLY: no ONNX/DAgger]" if args.no_onnx_teacher else ""),
        flush=True,
    )
    device = torch.device(args.device)
    rw = {"teacher": args.teacher_reward} if n_teacher and args.teacher_reward > 0 else {"teacher": 0.0}
    rw["heading"] = args.heading_weight
    rw["travel"] = float(args.travel_weight)
    rw["img"] = float(args.img_weight)
    if args.root_xy_weight is not None:
        rw["root_xy"] = args.root_xy_weight
    if args.joint_weight is not None:
        rw["joint"] = float(args.joint_weight)
    if args.upright_weight is not None:
        rw["upright"] = float(args.upright_weight)
    if args.alive_weight is not None:
        rw["alive"] = float(args.alive_weight)
    # Sit/stand and acrobatics need looser fall tilt than walking.
    motion_tag = str(args.motions).lower()
    fall_tilt = 85.0 if any(k in motion_tag for k in ("sitstand", "roulade", "pick")) else 70.0
    fall_z = 0.03 if "sitstand" in motion_tag else 0.04
    xml_path = None
    if "sitstand" in motion_tag or "pick" in motion_tag:
        from simulation.mujoco.stand_policy_demo import RL_ROOT

        xml_path = RL_ROOT / "src/mjlab_microduck/robot/microduck/scene.xml"
    root_kw = dict(root_cond=args.root_cond, xy_sigma=args.xy_sigma, img_sigma=args.img_sigma,
                   img_every=args.img_every, joint_sigma=args.joint_sigma)
    env_kw = dict(reward_weights=rw, fall_tilt_deg=fall_tilt, fall_z=fall_z, xml_path=xml_path, **root_kw)
    envs = [
        MotionTrackingEnv(motions, seed=args.seed + i, **env_kw)
        for i in range(args.num_envs)
    ]
    eval_env = MotionTrackingEnv(
        eval_motions, seed=args.seed + 999, reward_weights={**rw, "teacher": 0.0}, fall_tilt_deg=fall_tilt,
        fall_z=fall_z, **root_kw,
    )
    obs_dim = envs[0].obs_dim

    model = ActorCritic(obs_dim, MotionTrackingEnv.ACT_DIM, hidden=args.hidden)
    agent = PPO(model, PPOConfig(), device)
    start_update = 0
    if args.resume and args.resume.exists():
        blob = torch.load(args.resume, map_location=device, weights_only=False)
        agent.model.load_state_dict(blob["model"])
        agent.opt.load_state_dict(blob["opt"])
        start_update = int(blob.get("update", 0)) + 1
        print(f"resumed from {args.resume} at update {start_update}")

    meta = {
        "motions": str(args.motions),
        "n_clips": len(motions),
        "n_teacher": n_teacher,
        "num_envs": args.num_envs,
        "rollout_steps": args.rollout_steps,
        "updates": args.updates,
        "obs_dim": obs_dim,
        "act_dim": MotionTrackingEnv.ACT_DIM,
        "root_cond": args.root_cond,
        "reward_weights": envs[0].w,
        "xy_sigma": args.xy_sigma,
        "joint_sigma": args.joint_sigma,
        "img_sigma": args.img_sigma,
        "img_every": args.img_every,
        "n_eval_clips": len(eval_motions),
        "eval_motions": str(args.eval_motions or args.motions),
        "device": str(device),
        "seed": args.seed,
        "bc_epochs": args.bc_epochs,
        "teacher_reward": args.teacher_reward,
        "best_metric": args.best_metric,
    }
    (args.out / "config.json").write_text(json.dumps(meta, indent=2))

    t0 = time.time()
    best_score = -1.0

    def _best_score(ev: dict) -> float:
        travel = float(ev.get("travel_xy_m", 0.0))
        succ = float(ev.get("success_rate", 0.0))
        if args.best_metric == "travel":
            return travel
        if args.best_metric == "travel_x_succ":
            return travel * succ
        return succ

    onnx_teachers = None
    if args.onnx_teacher.exists():
        onnx_teachers = [OnnxWalkingTeacher(args.onnx_teacher) for _ in range(args.num_envs)]
        print(f"ONNX teacher: {args.onnx_teacher}", flush=True)

    # Closed-loop ONNX BC warm-start.
    if onnx_teachers is not None and args.bc_epochs > 0 and start_update == 0:
        print("BC warm-start (closed-loop ONNX)…", flush=True)
        bc_env = MotionTrackingEnv(motions, seed=args.seed + 7, reward_weights={"teacher": 0.0})
        bc_teacher = OnnxWalkingTeacher(args.onnx_teacher)
        for ep in range(args.bc_epochs):
            obs_list, act_list = [], []
            for ci in range(len(motions)):
                o = bc_env.reset(clip_idx=ci, start_frac=0.0)
                bc_teacher.reset()
                done = False
                while not done:
                    t_act = bc_teacher.act_env(bc_env)
                    obs_list.append(o)
                    act_list.append(t_act)
                    o, _r, done, _info = bc_env.step(t_act)
            batch = {
                "obs": np.asarray(obs_list, dtype=np.float32),
                "teacher": np.asarray(act_list, dtype=np.float32),
            }
            loss = bc_update(agent, batch, weight=1.0)
            if ep % 5 == 0 or ep == args.bc_epochs - 1:
                ev = eval_policy(eval_env, agent, n_episodes=min(4, len(motions)))
                print(
                    f"  bc_ep={ep} loss={loss:.4f} eval_succ={ev['success_rate']:.2f} "
                    f"len={ev['mean_len']:.0f} fall={ev['fall_rate']:.2f}",
                    flush=True,
                )
        torch.save(
            {"model": agent.model.state_dict(), "opt": agent.opt.state_dict(), "update": -1},
            ckpt_dir / "update_bc.pt",
        )

    for upd in range(start_update, args.updates):
        if args.time_budget_s > 0 and (time.time() - t0) >= args.time_budget_s:
            print(f"time budget reached at update {upd}")
            break
        # Linear decay of teacher mix.
        if args.bc_decay_updates > 0:
            frac = min(1.0, upd / args.bc_decay_updates)
            bc_mix = args.bc_mix_start + frac * (args.bc_mix_end - args.bc_mix_start)
        else:
            bc_mix = args.bc_mix_end
        # Alternate: mostly DAgger mix, every 4th update pure student (learn to survive alone).
        mix_now = 0.0 if (upd % 4 == 3) else bc_mix
        batch = collect_rollout(
            envs, agent, args.rollout_steps, bc_mix=mix_now, teachers=onnx_teachers
        )
        stats = agent.update(batch)
        # BC aux only on DAgger batches (teacher labels meaningful on mixed states).
        bc_loss = (
            bc_update(agent, batch, weight=args.bc_aux_weight)
            if onnx_teachers and mix_now > 0
            else 0.0
        )
        stats["bc_loss"] = bc_loss
        stats["bc_mix"] = mix_now
        row = {
            "update": upd,
            "elapsed_s": time.time() - t0,
            "steps": (upd + 1) * args.num_envs * args.rollout_steps,
            "rollout_return_mean": float(batch["ep_returns"].mean()) if batch["ep_returns"].size else 0.0,
            "rollout_success": batch["success_rate"],
            "rollout_fall": batch["fall_rate"],
            **stats,
        }
        if upd % args.eval_every == 0 or upd == start_update:
            ev = eval_policy(eval_env, agent, n_episodes=min(8, len(motions)))
            row["eval"] = ev
            score = _best_score(ev)
            if score >= best_score:
                best_score = score
                torch.save(
                    {
                        "model": agent.model.state_dict(),
                        "opt": agent.opt.state_dict(),
                        "update": upd,
                        "eval": ev,
                        "best_metric": args.best_metric,
                        "best_score": best_score,
                    },
                    args.out / "best.pt",
                )
        if upd % args.ckpt_every == 0 or upd == start_update:
            path = ckpt_dir / f"update_{upd:05d}.pt"
            torch.save(
                {"model": agent.model.state_dict(), "opt": agent.opt.state_dict(), "update": upd, "row": row},
                path,
            )
            row["checkpoint"] = str(path)
        with metrics_path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        msg = (
            f"upd={upd:4d} steps={row['steps']:8d} "
            f"ret={row['rollout_return_mean']:.2f} "
            f"succ={row['rollout_success']:.2f} fall={row['rollout_fall']:.2f} "
            f"kl={stats['approx_kl']:.4f}"
        )
        if "eval" in row:
            e = row["eval"]
            msg += (
                f" | eval succ={e['success_rate']:.2f} mae={e['joint_mae']:.3f} "
                f"len={e['mean_len']:.0f} travel={e.get('travel_xy_m', 0.0):.3f}"
            )
            if np.isfinite(e.get("img_centroid_err", float("nan"))):
                msg += f" img={e['img_centroid_err']:.3f}"
        print(msg, flush=True)

    # Final save
    torch.save(
        {"model": agent.model.state_dict(), "opt": agent.opt.state_dict(), "update": upd},
        args.out / "last.pt",
    )
    summary = {
        "elapsed_s": time.time() - t0,
        "final_update": upd,
        "best_metric": args.best_metric,
        "best_score": best_score,
        "out": str(args.out),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
