"""Minimal PPO (actor-critic MLP) for motion tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 512):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.5))
        self.v = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.mu.weight, gain=0.01)

    def dist(self, obs: torch.Tensor) -> Normal:
        h = self.trunk(obs)
        mu = torch.tanh(self.mu(h)) * 1.5
        log_std = self.log_std.clamp(-3.0, -0.3)
        std = log_std.exp().expand_as(mu)
        return Normal(mu, std)

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.dist(obs)
        a = d.sample()
        logp = d.log_prob(a).sum(-1)
        v = self.v(obs).squeeze(-1)
        return a, logp, v

    def evaluate(
        self, obs: torch.Tensor, act: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        d = self.dist(obs)
        logp = d.log_prob(act).sum(-1)
        ent = d.entropy().sum(-1)
        v = self.v(obs).squeeze(-1)
        return logp, ent, v


@dataclass
class PPOConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    epochs: int = 4
    minibatch: int = 1024
    target_kl: float = 0.03


class PPO:
    def __init__(self, model: ActorCritic, cfg: PPOConfig, device: torch.device):
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device
        self.opt = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)

    @torch.no_grad()
    def act_np(self, obs: np.ndarray) -> tuple[np.ndarray, float, float]:
        o = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        a, logp, v = self.model.act(o)
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        return a.squeeze(0).cpu().numpy(), float(logp.item()), float(v.item())

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        cfg = self.cfg
        obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=self.device)
        act = torch.as_tensor(batch["act"], dtype=torch.float32, device=self.device)
        old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32, device=self.device)
        adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=self.device)
        ret = torch.as_tensor(batch["ret"], dtype=torch.float32, device=self.device)
        # Sanitize
        obs = torch.nan_to_num(obs)
        act = torch.nan_to_num(act)
        old_logp = torch.nan_to_num(old_logp)
        adv = torch.nan_to_num(adv)
        ret = torch.nan_to_num(ret)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        n = obs.shape[0]
        idx = np.arange(n)
        stats = {"loss_pi": 0.0, "loss_v": 0.0, "entropy": 0.0, "approx_kl": 0.0}
        n_updates = 0
        early_stop = False
        for _ in range(cfg.epochs):
            if early_stop:
                break
            self.rng_shuffle = np.random.permutation(idx)
            for start in range(0, n, cfg.minibatch):
                mb = self.rng_shuffle[start : start + cfg.minibatch]
                logp, ent, v = self.model.evaluate(obs[mb], act[mb])
                ratio = torch.exp((logp - old_logp[mb]).clamp(-20, 20))
                surr1 = ratio * adv[mb]
                surr2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv[mb]
                loss_pi = -torch.min(surr1, surr2).mean()
                loss_v = 0.5 * (ret[mb] - v).pow(2).mean()
                loss = loss_pi + cfg.vf_coef * loss_v - cfg.ent_coef * ent.mean()
                if not torch.isfinite(loss):
                    early_stop = True
                    break
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.opt.step()
                approx_kl = float((old_logp[mb] - logp).mean().clamp(-10, 10).item())
                stats["loss_pi"] += float(loss_pi.item())
                stats["loss_v"] += float(loss_v.item())
                stats["entropy"] += float(ent.mean().item())
                stats["approx_kl"] += approx_kl
                n_updates += 1
                if approx_kl > cfg.target_kl:
                    early_stop = True
                    break
        for k in stats:
            stats[k] /= max(n_updates, 1)
        stats["early_stop"] = float(early_stop)
        return stats


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_value: float,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    t = len(rewards)
    adv = np.zeros(t, dtype=np.float32)
    last_gae = 0.0
    for i in reversed(range(t)):
        next_nonterminal = 1.0 - float(dones[i])
        next_v = last_value if i == t - 1 else values[i + 1]
        delta = rewards[i] + gamma * next_v * next_nonterminal - values[i]
        last_gae = delta + gamma * lam * next_nonterminal * last_gae
        adv[i] = last_gae
    ret = adv + values
    return adv, ret
