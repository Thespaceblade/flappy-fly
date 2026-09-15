#!/usr/bin/env python3
"""Train a playable MaleCNS readout: imitate rule policy, then CEM fine-tune.

1) Imitation: run many courses with the hand-written rule teacher while the
   frozen graph dynamics run; fit the MLP readout to match flap/no-flap.
2) CEM: polish with dense fitness (100*pipes + seconds) via the C binary.

Exports play/brain/champion_params.bin for the viewer.
"""

from __future__ import annotations

import argparse
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HIDDEN = 16  # must match c/brain.h FF_BRAIN_HIDDEN


def load_ffsg(path: Path) -> dict:
    raw = path.read_bytes()
    if raw[:4] != b"FFSG":
        raise SystemExit(f"bad magic in {path}")
    version, n, n_in, n_dn, n_edges = struct.unpack_from("<IIIII", raw, 4)
    if version != 1 or n_in != 6:
        raise SystemExit(f"unsupported graph header in {path}")
    o = 24
    feature_to_cell = list(struct.unpack_from("<6i", raw, o))
    o += 24
    dn_pad = list(struct.unpack_from("<32i", raw, o))
    o += 128
    row_ptr = list(struct.unpack_from(f"<{n + 1}i", raw, o))
    o += 4 * (n + 1)
    col_idx = list(struct.unpack_from(f"<{n_edges}i", raw, o))
    o += 4 * n_edges
    weight = list(struct.unpack_from(f"<{n_edges}f", raw, o))
    return {
        "n": n,
        "n_dn": n_dn,
        "n_edges": n_edges,
        "feature_to_cell": feature_to_cell,
        "dn_index": dn_pad[:n_dn],
        "row_ptr": row_ptr,
        "col_idx": col_idx,
        "weight": weight,
    }


class Brain:
    def __init__(self, g: dict):
        self.g = g
        self.n = g["n"]
        self.n_dn = g["n_dn"]
        self.h = np.zeros(self.n, dtype=np.float64)
        self.n_params = HIDDEN * self.n_dn + HIDDEN + 2 * HIDDEN + 2
        self.params = np.zeros(self.n_params, dtype=np.float64)

    def reset(self) -> None:
        self.h[:] = 0

    def set_params(self, p: np.ndarray) -> None:
        self.params = np.asarray(p, dtype=np.float64).reshape(-1)[: self.n_params].copy()

    def _step(self, u: np.ndarray) -> None:
        g = self.g
        h_new = np.empty(self.n, dtype=np.float64)
        for i in range(self.n):
            acc = u[i]
            a, b = g["row_ptr"][i], g["row_ptr"][i + 1]
            for k in range(a, b):
                acc += 1.4 * g["weight"][k] * self.h[g["col_idx"][k]]
            h_new[i] = 0.3 * self.h[i] + 0.7 * np.tanh(acc)
        self.h = h_new

    def dn_acts(self, features: np.ndarray) -> np.ndarray:
        u = np.zeros(self.n, dtype=np.float64)
        for f, cell in enumerate(self.g["feature_to_cell"]):
            if 0 <= cell < self.n:
                u[cell] = 2.0 * (float(features[f]) - 0.5)
        for _ in range(3):
            self._step(u)
        acts = np.zeros(self.n_dn, dtype=np.float64)
        for i, cell in enumerate(self.g["dn_index"]):
            if 0 <= cell < self.n:
                acts[i] = 4.0 * self.h[cell]
        return acts

    def logits(self, acts: np.ndarray) -> np.ndarray:
        p = self.params
        nd = self.n_dn
        W1 = p[: HIDDEN * nd].reshape(HIDDEN, nd)
        b1 = p[HIDDEN * nd : HIDDEN * nd + HIDDEN]
        W2 = p[HIDDEN * nd + HIDDEN : HIDDEN * nd + HIDDEN + 2 * HIDDEN].reshape(2, HIDDEN)
        b2 = p[-2:]
        hid = np.tanh(W1 @ acts + b1)
        return W2 @ hid + b2

    def decide_from_acts(self, acts: np.ndarray) -> int:
        z = self.logits(acts)
        return 1 if z[0] > z[1] else 0


# --- minimal Flappy twin (matches c/flappy.c / play) ---
W, H = 288, 512
BIRD_W, BIRD_H = 20, 20
BIRD_X, BIRD_START_Y = 80, 246
PIPE_W, PIPE_H, PIPE_GAP = 52, 320, 96
PIPE_H_MIN, PIPE_H_MAX = 180, 360
GROUND_Y, SCROLL, PHYS_HZ = 400, 2, 60
GRAVITY, FLAP_V, MAX_FALL = 0.3, -5.0, 8.0


class Rng:
    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF or 1

    def u32(self) -> int:
        self.s = (self.s * 1664525 + 1013904223) & 0xFFFFFFFF
        return self.s

    def range(self, lo: int, hi_ex: int) -> int:
        return (self.u32() % (hi_ex - lo)) + lo


def pipe_spacing() -> float:
    return (W - ((PIPE_W * 3) / 2)) / 2


class Game:
    def __init__(self, seed: int):
        self.rng = Rng(seed)
        self.bird_y = float(BIRD_START_Y)
        self.bird_vy = 0.0
        self.pipes: list[dict] = []
        self.frame = 0
        self.score = 0
        self.alive = True
        sp = pipe_spacing()
        x0 = sp - (PIPE_W >> 1) + W
        self._spawn(x0)
        self._spawn(self.pipes[0]["x"] + sp + PIPE_W)
        self._spawn(self.pipes[1]["x"] + sp + PIPE_W)

    def _spawn(self, x: float) -> None:
        self.pipes.append(
            {
                "x": float(x),
                "height": float(self.rng.range(PIPE_H_MIN, PIPE_H_MAX)),
                "scored": False,
            }
        )

    def next_pipe(self) -> dict | None:
        best = None
        best_x = 1e9
        for p in self.pipes:
            if p["x"] + PIPE_W >= BIRD_X and p["x"] < best_x:
                best, best_x = p, p["x"]
        return best

    def features(self) -> np.ndarray:
        bird_mid = self.bird_y + BIRD_H * 0.5
        out = np.zeros(6, dtype=np.float64)
        out[0] = np.clip(bird_mid / GROUND_Y, 0, 1)
        out[1] = np.clip((self.bird_vy + 5.0) / 13.0, 0, 1)
        p = self.next_pipe()
        if not p:
            out[2], out[3], out[4], out[5] = 0.5, 0.0, PIPE_GAP / GROUND_Y, 0.5
            return out
        gap = p["height"] - PIPE_GAP * 0.5
        out[2] = np.clip(gap / GROUND_Y, 0, 1)
        dist = p["x"] - (BIRD_X + BIRD_W)
        out[3] = np.clip(1.0 - dist / W, 0, 1)
        out[4] = np.clip((PIPE_GAP * 0.5) / GROUND_Y, 0, 1)
        out[5] = np.clip(0.5 + (bird_mid - gap) / GROUND_Y, 0, 1)
        return out

    def rule_flap(self) -> int:
        p = self.next_pipe()
        mid = self.bird_y + BIRD_H / 2
        gap = p["height"] - PIPE_GAP / 2 if p else GROUND_Y / 2
        err = mid - gap
        if err > 4:
            return 1
        if err > -8 and self.bird_vy > 1:
            return 1
        return 0

    def step(self, flap: int) -> None:
        if not self.alive:
            return
        if flap:
            self.bird_vy = FLAP_V
        self.bird_vy += GRAVITY
        if self.bird_vy > MAX_FALL:
            self.bird_vy = MAX_FALL
        self.bird_y += self.bird_vy
        for p in self.pipes:
            p["x"] -= SCROLL
            if not p["scored"] and (p["x"] == BIRD_X or p["x"] == BIRD_X - 1):
                p["scored"] = True
                self.score += 1
        self.pipes = [p for p in self.pipes if p["x"] >= -PIPE_W]
        while len(self.pipes) < 3:
            maxx = max(p["x"] for p in self.pipes)
            self._spawn(maxx + pipe_spacing() + PIPE_W)
        by = int(self.bird_y)
        if by >= GROUND_Y - BIRD_H:
            self.alive = False
            return
        if self.bird_y < 0:
            self.bird_y = 0.0
        for p in self.pipes:
            top_y = (p["height"] - PIPE_H) - PIPE_GAP
            bot_y = p["height"]
            if (
                BIRD_X + BIRD_W >= p["x"]
                and BIRD_X <= p["x"] + PIPE_W
                and by + BIRD_H >= top_y
                and by <= top_y + PIPE_H
            ) or (
                BIRD_X + BIRD_W >= p["x"]
                and BIRD_X <= p["x"] + PIPE_W
                and by + BIRD_H >= bot_y
                and by <= bot_y + PIPE_H
            ):
                self.alive = False
                return
        self.frame += 1


def collect_imitation(brain: Brain, n_courses: int, seed0: int) -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(n_courses):
        g = Game(seed0 + i)
        brain.reset()
        max_frames = 60 * PHYS_HZ
        while g.alive and g.frame < max_frames:
            if g.frame % 2 == 0:
                feat = g.features()
                acts = brain.dn_acts(feat)
                label = g.rule_flap()
                xs.append(acts)
                ys.append(label)
                g.step(label)
            else:
                g.step(0)
    return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)


def collect_dagger(
    brain: Brain, n_courses: int, seed0: int, beta: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float]:
    """Roll out mostly with learner; label every state with the rule teacher."""
    xs, ys = [], []
    pipes = []
    for i in range(n_courses):
        g = Game(seed0 + i)
        brain.reset()
        max_frames = 60 * PHYS_HZ
        while g.alive and g.frame < max_frames:
            if g.frame % 2 == 0:
                feat = g.features()
                acts = brain.dn_acts(feat)
                teacher = g.rule_flap()
                xs.append(acts.copy())
                ys.append(teacher)
                if rng.random() < beta:
                    action = teacher
                else:
                    action = brain.decide_from_acts(acts)
                g.step(action)
            else:
                g.step(0)
        pipes.append(g.score)
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.int64),
        float(np.mean(pipes)),
    )


def eval_pipes(brain: Brain, params: np.ndarray, seeds: list[int]) -> float:
    brain.set_params(params)
    scores = []
    for s in seeds:
        g = Game(s)
        brain.reset()
        while g.alive and g.frame < 60 * PHYS_HZ:
            if g.frame % 2 == 0:
                acts = brain.dn_acts(g.features())
                g.step(brain.decide_from_acts(acts))
            else:
                g.step(0)
        scores.append(g.score)
    return float(np.mean(scores))


def fit_readout(
    brain: Brain, X: np.ndarray, y: np.ndarray, steps: int = 800, init: np.ndarray | None = None
) -> np.ndarray:
    """SGD on MLP readout to classify flap from DN activities."""
    nd = brain.n_dn
    rng = np.random.default_rng(0)
    if init is not None:
        p = np.asarray(init, dtype=np.float64).copy()
    else:
        p = rng.normal(0, 0.15, size=brain.n_params)
    lr = 0.05
    n = len(y)
    for step in range(steps):
        idx = rng.integers(0, n, size=min(256, n))
        xb, yb = X[idx], y[idx]
        W1 = p[: HIDDEN * nd].reshape(HIDDEN, nd)
        b1 = p[HIDDEN * nd : HIDDEN * nd + HIDDEN]
        W2 = p[HIDDEN * nd + HIDDEN : HIDDEN * nd + HIDDEN + 2 * HIDDEN].reshape(2, HIDDEN)
        b2 = p[-2:]
        pre = W1 @ xb.T + b1[:, None]  # H x B
        hid = np.tanh(pre)
        logits = W2 @ hid + b2[:, None]  # 2 x B  (0=flap, 1=idle)
        # softmax CE
        logits -= logits.max(axis=0, keepdims=True)
        exp = np.exp(logits)
        prob = exp / exp.sum(axis=0, keepdims=True)
        target = np.zeros_like(prob)
        target[0, np.arange(len(yb))] = yb  # flap
        target[1, np.arange(len(yb))] = 1 - yb  # idle
        # Upweight rare flaps so we don't collapse to always-idle.
        w = np.where(yb == 1, 4.0, 1.0)
        dlogits = (prob - target) * w[None, :]
        dlogits /= w.sum()
        dW2 = dlogits @ hid.T
        db2 = dlogits.sum(axis=1)
        dhid = W2.T @ dlogits
        dpre = dhid * (1 - hid**2)
        dW1 = dpre @ xb
        db1 = dpre.sum(axis=1)
        grad = np.concatenate([dW1.ravel(), db1, dW2.ravel(), db2])
        p -= lr * grad
        if step % 200 == 0:
            lr *= 0.85
            z0, z1 = logits[0], logits[1]
            pred = (z0 > z1).astype(np.int64)
            acc = float((pred == yb).mean())
            print(f"  imitate step {step:4d} batch_acc={acc:.3f} lr={lr:.4f}")
    brain.set_params(p)
    correct = 0
    for j in range(len(X)):
        z = brain.logits(X[j])
        pred = 1 if z[0] > z[1] else 0
        correct += int(pred == y[j])
    print(f"  imitate acc={correct / len(y):.3f} on {len(y)} samples")
    return brain.params.copy()


def write_params(path: Path, vec: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack(f"<{len(vec)}f", *np.asarray(vec, dtype=np.float32).tolist()))


def cem_finetune(
    binary: Path,
    graph: Path,
    init: np.ndarray,
    *,
    generations: int,
    population: int,
    elites: int,
    courses: int,
    seed: int,
    out: Path,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    dim = len(init)
    mean = init.astype(np.float64).copy()
    sigma = np.full(dim, 0.25, dtype=np.float64)
    best = mean.copy()
    best_fit = -1.0
    validation = [1100001, 1100002, 1100003, 1100004, 1100005, 1100006]

    def eval_vec(vec: np.ndarray, seeds: list[int]) -> float:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
            p = Path(f.name)
        try:
            write_params(p, vec)
            fits = []
            for s in seeds:
                outp = subprocess.check_output(
                    [
                        str(binary),
                        "rollout",
                        "--seed",
                        str(s),
                        "--policy",
                        "brain",
                        "--graph",
                        str(graph),
                        "--params",
                        str(p),
                        "--quiet",
                    ],
                    text=True,
                ).split()
                pipes, seconds = int(outp[1]), float(outp[2])
                fits.append(100.0 * pipes + seconds)
            return float(np.mean(fits))
        finally:
            p.unlink(missing_ok=True)

    for gen in range(generations):
        mean_prev = mean.copy()
        train_seeds = [int(rng.integers(1, 900001)) for _ in range(courses)]
        pop = mean_prev + sigma * rng.standard_normal((population, dim))
        fits = np.array([eval_vec(pop[i], train_seeds) for i in range(population)])
        order = np.argsort(-fits)
        elite = pop[order[:elites]]
        mean = 0.3 * mean_prev + 0.7 * elite.mean(axis=0)
        sigma = np.maximum(elite.std(axis=0), 0.05)
        val = eval_vec(elite[0], validation)
        if val > best_fit:
            best_fit = val
            best = elite[0].copy()
            write_params(out / "champion_params.bin", best)
            write_params(ROOT / "play" / "brain" / "champion_params.bin", best)
        print(
            f"cem gen {gen:03d} train={fits[order[:elites]].mean():.1f} "
            f"val={val:.1f} best={best_fit:.1f} sigma={sigma.mean():.3f}"
        )
    write_params(out / "champion_params.bin", best)
    write_params(ROOT / "play" / "brain" / "champion_params.bin", best)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=ROOT / "data" / "processed" / "subgraph_v1.bin")
    ap.add_argument("--bin", type=Path, default=ROOT / "c" / "flappy_fly")
    ap.add_argument("--courses", type=int, default=40, help="courses per DAgger round")
    ap.add_argument("--dagger-rounds", type=int, default=8)
    ap.add_argument("--imitate-steps", type=int, default=600)
    ap.add_argument("--cem-gens", type=int, default=40)
    ap.add_argument("--cem-pop", type=int, default=32)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "play_train")
    ap.add_argument("--seed", type=int, default=20260916)
    args = ap.parse_args()

    if not args.graph.exists():
        raise SystemExit(f"missing {args.graph}; run scripts/build_subgraph.py")
    if not args.bin.exists():
        raise SystemExit(f"missing {args.bin}; run make -C c")

    args.out.mkdir(parents=True, exist_ok=True)
    play_brain = ROOT / "play" / "brain"
    play_brain.mkdir(parents=True, exist_ok=True)

    print(f"loading {args.graph}")
    g = load_ffsg(args.graph)
    print(f"  n={g['n']} edges={g['n_edges']} dn={g['n_dn']}")
    brain = Brain(g)
    rng = np.random.default_rng(args.seed)

    # Seed with teacher-only trajectories, then DAgger closed-loop rounds.
    print(f"bootstrap imitation ({args.courses} courses)…")
    X, y = collect_imitation(brain, args.courses, args.seed)
    print(f"  samples={len(y)} flap_rate={y.mean():.3f}")
    params = fit_readout(brain, X, y, steps=args.imitate_steps)
    write_params(play_brain / "champion_params.bin", params)

    for rnd in range(args.dagger_rounds):
        beta = max(0.05, 0.6 * (0.7**rnd))  # mix of teacher actions while collecting
        print(f"DAgger round {rnd + 1}/{args.dagger_rounds} beta={beta:.2f}")
        Xn, yn, mean_pipes = collect_dagger(
            brain, args.courses, args.seed + 10_000 * (rnd + 1), beta, rng
        )
        X = np.concatenate([X, Xn], axis=0)
        y = np.concatenate([y, yn], axis=0)
        # Cap dataset size
        if len(y) > 80_000:
            keep = rng.choice(len(y), 80_000, replace=False)
            X, y = X[keep], y[keep]
        print(f"  dataset={len(y)} collect_pipes={mean_pipes:.2f}")
        params = fit_readout(
            brain, X, y, steps=args.imitate_steps, init=params
        )
        pipes = eval_pipes(
            brain, params, list(range(args.seed + 2000, args.seed + 2012))
        )
        print(f"  eval pipes mean={pipes:.2f}")
        write_params(args.out / "imitate_params.bin", params)
        write_params(play_brain / "champion_params.bin", params)

    print("CEM fine-tune via C…")
    cem_finetune(
        args.bin,
        args.graph,
        params,
        generations=args.cem_gens,
        population=args.cem_pop,
        elites=6,
        courses=4,
        seed=args.seed + 7,
        out=args.out,
    )
    print(f"done — refresh http://127.0.0.1:8765/play/")


if __name__ == "__main__":
    main()
