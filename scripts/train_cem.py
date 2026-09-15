#!/usr/bin/env python3
"""CEM trainer for Flappy Fly readout weights.

Optimizes only the MLP readout (graph frozen). Calls the C binary for rollouts.

Usage:
  .venv/bin/python scripts/train_cem.py \\
    --graph data/processed/subgraph_v1.bin \\
    --arm real --generations 40

Arms: real | shuffle | silence (graph + optional --silence)
"""

from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def n_params(binary: Path, graph: Path | None) -> int:
    cmd = [str(binary), "nparams"]
    if graph is not None:
        cmd += ["--graph", str(graph)]
    out = subprocess.check_output(cmd, text=True).strip()
    return int(out)


def write_params(path: Path, vec: np.ndarray) -> None:
    path.write_bytes(struct.pack(f"<{len(vec)}f", *vec.astype(np.float32).tolist()))


def run_quiet_rollout(
    binary: Path,
    seed: int,
    *,
    graph: Path | None,
    params: Path | None,
    silence: bool,
) -> tuple[int, float]:
    cmd = [
        str(binary),
        "rollout",
        "--seed",
        str(seed),
        "--policy",
        "brain",
        "--quiet",
    ]
    if graph is not None:
        cmd += ["--graph", str(graph)]
    if params is not None:
        cmd += ["--params", str(params)]
    if silence:
        cmd.append("--silence")
    out = subprocess.check_output(cmd, text=True).strip()
    parts = out.split()
    pipes = int(parts[1])
    seconds = float(parts[2])
    return pipes, seconds


def fitness(pipes: float, seconds: float) -> float:
    """Dense demo score: reward surviving and clearing pipes."""
    return 100.0 * pipes + seconds


def evaluate(
    binary: Path,
    vec: np.ndarray,
    courses: list[int],
    *,
    graph: Path | None,
    silence: bool,
    work: Path,
) -> dict:
    pfile = work / "params.bin"
    write_params(pfile, vec)
    scores = []
    for s in courses:
        pipes, seconds = run_quiet_rollout(
            binary, s, graph=graph, params=pfile, silence=silence
        )
        scores.append(
            {
                "seed": s,
                "pipes": pipes,
                "seconds": seconds,
                "fitness": fitness(pipes, seconds),
            }
        )
    mean_pipes = float(np.mean([x["pipes"] for x in scores]))
    mean_seconds = float(np.mean([x["seconds"] for x in scores]))
    mean_fitness = float(np.mean([x["fitness"] for x in scores]))
    return {
        "mean_pipes": mean_pipes,
        "mean_seconds": mean_seconds,
        "mean_fitness": mean_fitness,
        "episodes": scores,
    }


def run_cem(args: argparse.Namespace) -> dict:
    binary = args.bin.resolve()
    if not binary.exists():
        raise SystemExit(f"missing binary {binary}; run: make -C c")

    silence = args.arm == "silence" or args.silence

    if args.no_graph:
        graph = None
    elif args.arm == "shuffle":
        graph = args.shuffle_graph.resolve()
        if not graph.exists():
            raise SystemExit(f"missing shuffle graph {graph}")
    else:
        graph = args.graph.resolve() if args.graph else None
        if graph is not None and not graph.exists():
            raise SystemExit(
                f"missing graph {graph}; run: scripts/build_subgraph.py "
                "(or pass --no-graph for identity stub)"
            )

    rng = np.random.default_rng(args.seed)
    dim = n_params(binary, graph)
    mean = np.zeros(dim, dtype=np.float64)
    if args.init_params is not None and args.init_params.exists():
        raw = np.frombuffer(args.init_params.read_bytes(), dtype=np.float32)
        if len(raw) >= dim:
            mean = raw[:dim].astype(np.float64).copy()
            print(f"warm-start from {args.init_params} ({dim} params)")
    sigma = np.full(dim, args.init_sigma, dtype=np.float64)

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    best_val = -1.0
    best_pipes = -1.0
    champion = mean.copy()

    with tempfile.TemporaryDirectory(prefix="ff_cem_") as tmp:
        work = Path(tmp)
        for gen in range(args.generations):
            mean_prev = mean.copy()
            courses = [
                int(rng.integers(args.train_lo, args.train_hi + 1))
                for _ in range(args.courses_per_gen)
            ]
            noise = rng.standard_normal((args.population, dim))
            pop = mean_prev + sigma * noise
            fits = np.empty(args.population, dtype=np.float64)
            for i in range(args.population):
                res = evaluate(
                    binary,
                    pop[i],
                    courses,
                    graph=graph,
                    silence=silence,
                    work=work,
                )
                fits[i] = res["mean_fitness"]

            order = np.argsort(-fits)
            elite_idx = order[: args.elites]
            elites_x = pop[elite_idx]
            elite_mean = elites_x.mean(axis=0)
            elite_std = elites_x.std(axis=0)

            # mean_mix = fraction of OLD mean kept
            mean = args.mean_mix * mean_prev + (1.0 - args.mean_mix) * elite_mean
            sigma = np.maximum(elite_std, args.sigma_floor)

            train_fit = float(fits[elite_idx].mean())
            val = evaluate(
                binary,
                elites_x[0],
                args.validation,
                graph=graph,
                silence=silence,
                work=work,
            )
            val_fit = val["mean_fitness"]
            val_pipes = val["mean_pipes"]
            if val_fit > best_val:
                best_val = val_fit
                best_pipes = val_pipes
                champion = elites_x[0].copy()
                write_params(out_dir / "champion_params.bin", champion)
                # Keep play/ viewer in sync when training for demo.
                play_dir = ROOT / "play" / "brain"
                if play_dir.is_dir():
                    write_params(play_dir / "champion_params.bin", champion)

            row = {
                "generation": gen,
                "courses": courses,
                "train_elite_mean_fitness": train_fit,
                "val_mean_fitness": val_fit,
                "val_mean_pipes": val_pipes,
                "best_val_mean_fitness": best_val,
                "best_val_mean_pipes": best_pipes,
                "sigma_mean": float(sigma.mean()),
            }
            history.append(row)
            print(
                f"gen {gen:03d} train_fit={train_fit:.2f} "
                f"val_fit={val_fit:.2f} val_pipes={val_pipes:.3f} "
                f"best_fit={best_val:.2f} best_pipes={best_pipes:.3f} "
                f"sigma={sigma.mean():.4f}"
            )

    write_params(out_dir / "final_mean_params.bin", mean)
    write_params(out_dir / "champion_params.bin", champion)
    play_dir = ROOT / "play" / "brain"
    if play_dir.is_dir():
        write_params(play_dir / "champion_params.bin", champion)

    summary = {
        "arm": args.arm,
        "seed": args.seed,
        "dim": dim,
        "graph": str(graph) if graph else None,
        "silence": silence,
        "fitness": "100*pipes + seconds",
        "generations": args.generations,
        "population": args.population,
        "elites": args.elites,
        "best_val_mean_fitness": best_val,
        "best_val_mean_pipes": best_pipes,
        "history": history,
        "champion_params": "champion_params.bin",
    }
    (out_dir / "train.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"wrote {out_dir / 'train.json'} "
        f"best_fit={best_val:.2f} best_pipes={best_pipes:.3f}"
    )
    return summary


def baselines(args: argparse.Namespace) -> dict:
    binary = args.bin.resolve()
    rng = np.random.default_rng(args.seed)
    courses = [int(rng.integers(1, 900001)) for _ in range(3)]
    summary: dict = {"courses": courses, "baselines": {}}
    for policy in ("idle", "random", "rule", "brain"):
        scores = []
        for s in courses:
            cmd = [
                str(binary),
                "rollout",
                "--seed",
                str(s),
                "--policy",
                policy,
                "--quiet",
            ]
            out = subprocess.check_output(cmd, text=True).strip().split()
            scores.append(
                {"seed": s, "pipes": int(out[1]), "seconds": float(out[2])}
            )
        mean_pipes = sum(x["pipes"] for x in scores) / len(scores)
        summary["baselines"][policy] = {
            "mean_pipes": mean_pipes,
            "episodes": scores,
        }
        print(f"{policy:8s} mean_pipes={mean_pipes:.3f}")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "baselines.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {path}")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", type=Path, default=ROOT / "c" / "flappy_fly")
    ap.add_argument(
        "--graph",
        type=Path,
        default=ROOT / "data" / "processed" / "subgraph_v1.bin",
    )
    ap.add_argument(
        "--shuffle-graph",
        type=Path,
        default=ROOT / "data" / "processed" / "subgraph_v1_shuffle.bin",
    )
    ap.add_argument(
        "--arm",
        choices=("real", "shuffle", "silence", "baselines"),
        default="real",
    )
    ap.add_argument("--silence", action="store_true")
    ap.add_argument(
        "--no-graph",
        action="store_true",
        help="Use identity-stub circuit (no MaleCNS .bin)",
    )
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--elites", type=int, default=4)
    ap.add_argument("--courses-per-gen", type=int, default=3)
    ap.add_argument("--init-sigma", type=float, default=0.8)
    ap.add_argument("--sigma-floor", type=float, default=0.07)
    ap.add_argument("--mean-mix", type=float, default=0.3)
    ap.add_argument("--train-lo", type=int, default=1)
    ap.add_argument("--train-hi", type=int, default=900000)
    ap.add_argument(
        "--validation",
        type=int,
        nargs="+",
        default=[1100001, 1100002, 1100003, 1100004],
    )
    ap.add_argument("--seed", type=int, default=20260914)
    ap.add_argument(
        "--init-params",
        type=Path,
        default=None,
        help="Optional float32 params.bin warm-start",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="results directory (default: results/cem_<arm>_<seed>)",
    )
    args = ap.parse_args()
    if args.out is None:
        args.out = ROOT / "results" / f"cem_{args.arm}_{args.seed}"

    if args.arm == "baselines":
        baselines(args)
    else:
        run_cem(args)


if __name__ == "__main__":
    main()
