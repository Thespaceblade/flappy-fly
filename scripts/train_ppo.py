#!/usr/bin/env python3
"""PPO fine-tune of the MaleCNS readout (graph frozen).

Starts from play/brain/champion_params.bin (or --init). Dense rewards:
  +0.1 per decision while alive
  +10.0 per pipe cleared
  -0.05 * |height error| (from feature channel 5)
  -5.0 on death

Value net is separate (not exported). Policy params stay C-compatible.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_play import (  # noqa: E402
    HIDDEN,
    PHYS_HZ,
    Brain,
    Game,
    eval_pipes,
    load_ffsg,
    write_params,
)


def softmax_logits(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max()
    e = np.exp(z)
    return e / e.sum()


def forward(params: np.ndarray, acts: np.ndarray, n_dn: int):
    W1 = params[: HIDDEN * n_dn].reshape(HIDDEN, n_dn)
    b1 = params[HIDDEN * n_dn : HIDDEN * n_dn + HIDDEN]
    W2 = params[HIDDEN * n_dn + HIDDEN : HIDDEN * n_dn + HIDDEN + 2 * HIDDEN].reshape(
        2, HIDDEN
    )
    b2 = params[-2:]
    pre = W1 @ acts + b1
    hid = np.tanh(pre)
    logits = W2 @ hid + b2
    return logits, hid, pre, W1, b1, W2, b2


def policy_logp_grad(params: np.ndarray, acts: np.ndarray, action: int, n_dn: int):
    """Grad of log π(action|acts). action: 1=flap→class0, 0=idle→class1."""
    logits, hid, pre, W1, b1, W2, b2 = forward(params, acts, n_dn)
    prob = softmax_logits(logits)
    cls = 0 if action == 1 else 1
    logp = np.log(prob[cls] + 1e-12)

    dlogits = -prob
    dlogits[cls] += 1.0
    dW2 = np.outer(dlogits, hid)
    db2 = dlogits
    dhid = W2.T @ dlogits
    dpre = dhid * (1.0 - hid**2)
    dW1 = np.outer(dpre, acts)
    db1 = dpre
    grad = np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])
    return logp, grad, prob


class ValueNet:
    def __init__(self, n_dn: int, hidden: int = 32):
        self.n_dn = n_dn
        self.h = hidden
        rng = np.random.default_rng(1)
        self.W1 = rng.normal(0, 0.1, size=(hidden, n_dn))
        self.b1 = np.zeros(hidden)
        self.W2 = rng.normal(0, 0.1, size=hidden)
        self.b2 = 0.0

    def predict(self, acts: np.ndarray) -> float:
        h = np.tanh(self.W1 @ acts + self.b1)
        return float(self.W2 @ h + self.b2)

    def train_batch(self, X: np.ndarray, y: np.ndarray, lr: float = 1e-2, steps: int = 20):
        rng = np.random.default_rng()
        n = len(y)
        for _ in range(steps):
            idx = rng.integers(0, n, size=min(128, n))
            for j in idx:
                acts, target = X[j], y[j]
                pre = self.W1 @ acts + self.b1
                h = np.tanh(pre)
                pred = float(self.W2 @ h + self.b2)
                err = pred - target
                dW2 = err * h
                db2 = err
                dh = err * self.W2 * (1 - h**2)
                dW1 = np.outer(dh, acts)
                db1 = dh
                self.W1 -= lr * dW1
                self.b1 -= lr * db1
                self.W2 -= lr * dW2
                self.b2 -= lr * db2


def dense_reward(game: Game, prev_score: int, died: bool) -> float:
    r = 0.1  # alive
    if game.score > prev_score:
        r += 10.0
    # feature[5] is mapped height error; prefer centered (~0.5)
    feat = game.features()
    r -= 0.08 * abs(float(feat[5]) - 0.5)
    if died:
        r -= 5.0
    return r


def rollout_episode(brain: Brain, seed: int, rng: np.random.Generator, stochastic: bool):
    g = Game(seed)
    brain.reset()
    traj = []
    max_frames = 60 * PHYS_HZ
    while g.alive and g.frame < max_frames:
        if g.frame % 2 == 0:
            feat = g.features()
            acts = brain.dn_acts(feat)
            logits, _, _, _, _, _, _ = forward(brain.params, acts, brain.n_dn)
            prob = softmax_logits(logits)
            if stochastic:
                action = 1 if rng.random() < prob[0] else 0  # class0=flap
            else:
                action = 1 if logits[0] > logits[1] else 0
            prev = g.score
            g.step(action)
            died = not g.alive
            r = dense_reward(g, prev, died)
            traj.append({"acts": acts, "action": action, "reward": r, "prob0": prob[0]})
        else:
            g.step(0)
    return traj, g.score


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    adv = np.zeros(len(rewards))
    last = 0.0
    for t in reversed(range(len(rewards))):
        nxt = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + gamma * nxt - values[t]
        last = delta + gamma * lam * last
        adv[t] = last
    rets = adv + values
    return adv, rets


def ppo_update(params, batch, n_dn, clip=0.2, lr=3e-4, epochs=4):
    """batch: list of dicts with acts, action, adv, old_logp"""
    params = params.copy()
    advs = np.array([b["adv"] for b in batch])
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    for _ in range(epochs):
        order = np.random.permutation(len(batch))
        for i in order:
            b = batch[i]
            logp, grad, _ = policy_logp_grad(params, b["acts"], b["action"], n_dn)
            ratio = np.exp(logp - b["old_logp"])
            a = advs[i]
            unclipped = ratio * a
            clipped = np.clip(ratio, 1 - clip, 1 + clip) * a
            # maximize min → gradient ascent on surrogate
            if unclipped < clipped:
                # use unclipped path
                params += lr * a * grad * ratio  # d/dθ (ratio*A) ≈ ratio * grad_logp * A
            else:
                if (a > 0 and ratio > 1 + clip) or (a < 0 and ratio < 1 - clip):
                    continue  # clipped flat
                params += lr * a * grad * ratio
    return params


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=ROOT / "data" / "processed" / "subgraph_v1.bin")
    ap.add_argument(
        "--init",
        type=Path,
        default=ROOT / "play" / "brain" / "champion_params.bin",
    )
    ap.add_argument("--updates", type=int, default=80)
    ap.add_argument("--episodes", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument(
        "--train-pool",
        type=int,
        default=64,
        help="Fixed train course pool size (sampled with replacement each update)",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "ppo")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    play = ROOT / "play" / "brain"
    play.mkdir(parents=True, exist_ok=True)

    g = load_ffsg(args.graph)
    brain = Brain(g)
    raw = np.frombuffer(args.init.read_bytes(), dtype=np.float32)
    brain.set_params(raw)
    params = brain.params.copy()
    vnet = ValueNet(brain.n_dn)
    rng = np.random.default_rng(args.seed)

    # Fixed disjoint pools (reproducible). Do NOT use a single seed — that overfits.
    train_pool = list(range(args.seed, args.seed + args.train_pool))
    eval_seeds = list(range(args.seed + 100_000, args.seed + 100_012))
    print(f"train_pool={train_pool[0]}..{train_pool[-1]} ({len(train_pool)})")
    print(f"eval_seeds={eval_seeds[0]}..{eval_seeds[-1]}")

    best_pipes = eval_pipes(brain, params, eval_seeds)
    best_params = params.copy()
    write_params(args.out / "champion_params.bin", best_params)
    write_params(play / "champion_params.bin", best_params)
    print(f"init eval pipes={best_pipes:.2f} n_params={brain.n_params}")
    history = []

    for upd in range(args.updates):
        batch = []
        pipe_scores = []
        all_acts, all_rets = [], []
        # Cycle a fixed curriculum; shuffle order each update for variance.
        ep_seeds = [int(x) for x in rng.choice(train_pool, size=args.episodes, replace=True)]
        for seed in ep_seeds:
            brain.set_params(params)
            traj, pipes = rollout_episode(brain, seed, rng, stochastic=True)
            pipe_scores.append(pipes)
            rewards = [t["reward"] for t in traj]
            values = np.array([vnet.predict(t["acts"]) for t in traj])
            adv, rets = compute_gae(rewards, values)
            for i, t in enumerate(traj):
                logp, _, _ = policy_logp_grad(params, t["acts"], t["action"], brain.n_dn)
                batch.append(
                    {
                        "acts": t["acts"],
                        "action": t["action"],
                        "adv": float(adv[i]),
                        "old_logp": float(logp),
                    }
                )
                all_acts.append(t["acts"])
                all_rets.append(rets[i])
        if all_acts:
            vnet.train_batch(np.asarray(all_acts), np.asarray(all_rets), lr=1e-2, steps=40)
        if batch:
            params = ppo_update(params, batch, brain.n_dn, clip=0.2, lr=2.5e-4, epochs=3)

        brain.set_params(params)
        ev = eval_pipes(brain, params, eval_seeds)
        mean_train = float(np.mean(pipe_scores))
        if ev >= best_pipes:
            best_pipes = ev
            best_params = params.copy()
            write_params(args.out / "champion_params.bin", best_params)
            write_params(play / "champion_params.bin", best_params)
        row = {
            "update": upd,
            "train_pipes": mean_train,
            "eval_pipes": ev,
            "best_pipes": best_pipes,
            "batch": len(batch),
            "ep_seeds": ep_seeds,
        }
        history.append(row)
        print(
            f"ppo {upd:03d} train_pipes={mean_train:.2f} eval={ev:.2f} "
            f"best={best_pipes:.2f} steps={len(batch)}"
        )

    write_params(args.out / "champion_params.bin", best_params)
    write_params(play / "champion_params.bin", best_params)
    (args.out / "ppo.json").write_text(
        json.dumps(
            {
                "best_pipes": best_pipes,
                "updates": args.updates,
                "train_pool": train_pool,
                "eval_seeds": eval_seeds,
                "history": history,
                "reward": "0.1 alive + 10 pipe - 0.08*|feat5-0.5| - 5 death",
            },
            indent=2,
        )
        + "\n"
    )
    print(f"done best_pipes={best_pipes:.2f} — refresh play viewer")


if __name__ == "__main__":
    main()
