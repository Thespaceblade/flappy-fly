#!/usr/bin/env python3
"""Improve the playable champion: PyTorch PPO + imitation anchor + CEM polish.

Why the old PPO failed: pure RL forgot the good policy.
This run keeps a behavioral-cloning term toward an improved rule teacher
(and optionally the init champion), so updates refine instead of erase.

Exports C-compatible play/brain/champion_params.bin
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_play import (  # noqa: E402
    HIDDEN,
    PHYS_HZ,
    BIRD_H,
    PIPE_GAP,
    GROUND_Y,
    Brain,
    Game,
    cem_finetune,
    eval_pipes,
    load_ffsg,
    write_params,
)


def improved_rule(g: Game) -> int:
    """Slightly stronger than the v1 heuristic (mean ~0.8 pipes vs ~0.5)."""
    p = g.next_pipe()
    mid = g.bird_y + BIRD_H * 0.5
    gap = p["height"] - PIPE_GAP * 0.5 if p else GROUND_Y * 0.5
    err = mid - gap  # + => too low
    if err > 8:
        return 1
    if err > -6 and g.bird_vy > 1.5:
        return 1
    return 0


class ReadoutAC(nn.Module):
    """Actor-critic; actor weights pack to C FFSG readout layout."""

    def __init__(self, n_dn: int, hidden: int = HIDDEN):
        super().__init__()
        self.n_dn = n_dn
        self.hidden = hidden
        self.actor = nn.Sequential(
            nn.Linear(n_dn, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 2),
        )
        self.critic = nn.Sequential(
            nn.Linear(n_dn, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, acts: torch.Tensor):
        logits = self.actor(acts)
        value = self.critic(acts).squeeze(-1)
        return logits, value

    def load_c_params(self, flat: np.ndarray) -> None:
        nd, H = self.n_dn, self.hidden
        W1 = flat[: H * nd].reshape(H, nd)
        b1 = flat[H * nd : H * nd + H]
        W2 = flat[H * nd + H : H * nd + H + 2 * H].reshape(2, H)
        b2 = flat[-2:]
        with torch.no_grad():
            self.actor[0].weight.copy_(torch.from_numpy(W1.astype(np.float32)))
            self.actor[0].bias.copy_(torch.from_numpy(b1.astype(np.float32)))
            self.actor[2].weight.copy_(torch.from_numpy(W2.astype(np.float32)))
            self.actor[2].bias.copy_(torch.from_numpy(b2.astype(np.float32)))

    def export_c_params(self) -> np.ndarray:
        W1 = self.actor[0].weight.detach().cpu().numpy()
        b1 = self.actor[0].bias.detach().cpu().numpy()
        W2 = self.actor[2].weight.detach().cpu().numpy()
        b2 = self.actor[2].bias.detach().cpu().numpy()
        return np.concatenate([W1.ravel(), b1, W2.ravel(), b2]).astype(np.float64)


def dense_reward(g: Game, prev_score: int, died: bool) -> float:
    r = 0.15
    if g.score > prev_score:
        r += 12.0
    feat = g.features()
    r -= 0.12 * abs(float(feat[5]) - 0.5)
    # Prefer not slamming into ceiling/floor bands
    mid = g.bird_y + BIRD_H * 0.5
    if mid < 40 or mid > GROUND_Y - 40:
        r -= 0.2
    if died:
        r -= 6.0
    return r


def collect(
    brain: Brain,
    net: ReadoutAC,
    seeds: list[int],
    device: torch.device,
):
    net.eval()
    episodes = []
    pipe_scores = []
    for seed in seeds:
        g = Game(int(seed))
        brain.reset()
        traj = {
            "acts": [],
            "actions": [],
            "logps": [],
            "rewards": [],
            "values": [],
            "teachers": [],
        }
        while g.alive and g.frame < 60 * PHYS_HZ:
            if g.frame % 2 == 0:
                acts_np = brain.dn_acts(g.features())
                acts = torch.from_numpy(acts_np.astype(np.float32)).to(device)
                with torch.no_grad():
                    logits, value = net(acts)
                    dist = Categorical(logits=logits)
                    action = dist.sample()  # 0=flap, 1=idle (matches C class order)
                    logp = dist.log_prob(action)
                teacher = improved_rule(g)  # 1=flap
                teacher_cls = 0 if teacher == 1 else 1
                prev = g.score
                flap = 1 if int(action.item()) == 0 else 0
                g.step(flap)
                died = not g.alive
                traj["acts"].append(acts_np)
                traj["actions"].append(int(action.item()))
                traj["logps"].append(float(logp.item()))
                traj["values"].append(float(value.item()))
                traj["rewards"].append(dense_reward(g, prev, died))
                traj["teachers"].append(teacher_cls)
            else:
                g.step(0)
        pipe_scores.append(g.score)
        episodes.append(traj)
    return episodes, float(np.mean(pipe_scores))


def gae_advantages(rewards, values, gamma=0.99, lam=0.95):
    adv = []
    last = 0.0
    values = values + [0.0]
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * values[t + 1] - values[t]
        last = delta + gamma * lam * last
        adv.append(last)
    adv.reverse()
    adv = np.asarray(adv, dtype=np.float32)
    rets = adv + np.asarray(values[:-1], dtype=np.float32)
    return adv, rets


def ppo_bc_update(
    net: ReadoutAC,
    opt: torch.optim.Optimizer,
    episodes,
    device: torch.device,
    *,
    clip: float,
    ent_coef: float,
    bc_coef: float,
    epochs: int,
    batch_size: int,
):
    acts, actions, old_logps, advs, rets, teachers = [], [], [], [], [], []
    for ep in episodes:
        if not ep["rewards"]:
            continue
        a, r = gae_advantages(ep["rewards"], ep["values"])
        acts.extend(ep["acts"])
        actions.extend(ep["actions"])
        old_logps.extend(ep["logps"])
        advs.extend(a.tolist())
        rets.extend(r.tolist())
        teachers.extend(ep["teachers"])
    if not acts:
        return {}

    acts_t = torch.tensor(np.asarray(acts, dtype=np.float32), device=device)
    actions_t = torch.tensor(actions, dtype=torch.long, device=device)
    old_logp_t = torch.tensor(old_logps, dtype=torch.float32, device=device)
    adv_t = torch.tensor(advs, dtype=torch.float32, device=device)
    ret_t = torch.tensor(rets, dtype=torch.float32, device=device)
    teach_t = torch.tensor(teachers, dtype=torch.long, device=device)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)

    n = len(actions)
    stats = {"policy": 0.0, "value": 0.0, "ent": 0.0, "bc": 0.0, "n": 0}
    net.train()
    idx = np.arange(n)
    for _ in range(epochs):
        np.random.shuffle(idx)
        for start in range(0, n, batch_size):
            mb = idx[start : start + batch_size]
            logits, values = net(acts_t[mb])
            dist = Categorical(logits=logits)
            logp = dist.log_prob(actions_t[mb])
            ratio = torch.exp(logp - old_logp_t[mb])
            surr1 = ratio * adv_t[mb]
            surr2 = torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t[mb]
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, ret_t[mb])
            ent = dist.entropy().mean()
            bc_loss = F.cross_entropy(logits, teach_t[mb])
            loss = policy_loss + 0.5 * value_loss - ent_coef * ent + bc_coef * bc_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            stats["policy"] += float(policy_loss.item())
            stats["value"] += float(value_loss.item())
            stats["ent"] += float(ent.item())
            stats["bc"] += float(bc_loss.item())
            stats["n"] += 1
    if stats["n"]:
        for k in ("policy", "value", "ent", "bc"):
            stats[k] /= stats["n"]
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=ROOT / "data" / "processed" / "subgraph_v1.bin")
    ap.add_argument(
        "--init",
        type=Path,
        default=ROOT / "results" / "play_train" / "champion_params.bin",
    )
    ap.add_argument("--bin", type=Path, default=ROOT / "c" / "flappy_fly")
    ap.add_argument("--updates", type=int, default=120)
    ap.add_argument("--episodes", type=int, default=24)
    ap.add_argument("--train-pool", type=int, default=96)
    ap.add_argument("--bc-coef", type=float, default=0.35)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cem-gens", type=int, default=25)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "improve")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    play = ROOT / "play" / "brain"
    play.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    g = load_ffsg(args.graph)
    brain = Brain(g)
    net = ReadoutAC(brain.n_dn).to(device)
    init = np.frombuffer(args.init.read_bytes(), dtype=np.float32)
    net.load_c_params(init)
    brain.set_params(init)

    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    rng = np.random.default_rng(args.seed)
    train_pool = list(range(args.seed, args.seed + args.train_pool))
    eval_seeds = list(range(args.seed + 200_000, args.seed + 200_016))

    best_params = net.export_c_params()
    best_pipes = eval_pipes(brain, best_params, eval_seeds)
    write_params(args.out / "champion_params.bin", best_params)
    write_params(play / "champion_params.bin", best_params)
    print(
        f"init eval={best_pipes:.2f} train_pool={train_pool[0]}..{train_pool[-1]} "
        f"eval={eval_seeds[0]}..{eval_seeds[-1]} bc={args.bc_coef}"
    )

    history = []
    for upd in range(args.updates):
        ep_seeds = rng.choice(train_pool, size=args.episodes, replace=True).tolist()
        episodes, train_pipes = collect(brain, net, ep_seeds, device)
        # Keep brain DN dynamics in sync with exported actor during rollouts:
        brain.set_params(net.export_c_params())
        stats = ppo_bc_update(
            net,
            opt,
            episodes,
            device,
            clip=0.2,
            ent_coef=args.ent_coef,
            bc_coef=args.bc_coef,
            epochs=4,
            batch_size=256,
        )
        params = net.export_c_params()
        brain.set_params(params)
        ev = eval_pipes(brain, params, eval_seeds)
        if ev > best_pipes + 1e-6:
            best_pipes = ev
            best_params = params.copy()
            write_params(args.out / "champion_params.bin", best_params)
            write_params(play / "champion_params.bin", best_params)
            # Reload best into net if we want strict keep-best for actor — instead
            # keep exploring from current; champion file stores best.
        row = {
            "update": upd,
            "train_pipes": train_pipes,
            "eval_pipes": ev,
            "best_pipes": best_pipes,
            **{k: stats.get(k) for k in ("policy", "value", "ent", "bc")},
        }
        history.append(row)
        print(
            f"upd {upd:03d} train={train_pipes:.2f} eval={ev:.2f} best={best_pipes:.2f} "
            f"bc={stats.get('bc', 0):.3f} ent={stats.get('ent', 0):.3f}"
        )

    # Restore best actor before CEM
    net.load_c_params(best_params.astype(np.float32))
    print(f"CEM polish from best={best_pipes:.2f}…")
    if args.bin.exists() and args.cem_gens > 0:
        polished = cem_finetune(
            args.bin,
            args.graph,
            best_params,
            generations=args.cem_gens,
            population=28,
            elites=6,
            courses=4,
            seed=args.seed + 99,
            out=args.out,
        )
        brain.set_params(polished)
        ev = eval_pipes(brain, polished, eval_seeds)
        if ev > best_pipes:
            best_pipes = ev
            best_params = polished
            print(f"CEM improved eval to {best_pipes:.2f}")
        else:
            write_params(args.out / "champion_params.bin", best_params)
            write_params(play / "champion_params.bin", best_params)
            print(f"CEM did not beat best; kept {best_pipes:.2f}")

    write_params(args.out / "champion_params.bin", best_params)
    write_params(play / "champion_params.bin", best_params)
    (args.out / "improve.json").write_text(
        json.dumps(
            {
                "best_pipes": best_pipes,
                "eval_seeds": eval_seeds,
                "train_pool_size": len(train_pool),
                "bc_coef": args.bc_coef,
                "teacher": "improved_rule thr=8 fall=-6 vy=1.5",
                "history": history,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"done best_pipes={best_pipes:.2f} — refresh play viewer")


if __name__ == "__main__":
    main()
