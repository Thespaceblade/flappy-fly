#!/usr/bin/env python3
"""Train optical Flappy Fly: pixels → CNN → MaleCNS inject cells → DN readout.

Encoder learns to match engineered features (esp. vy / height-error) from a
readable 4×64² frame stack. Readout stays frozen at the feature champion.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ff_obs import render_obs  # noqa: E402
from optical_model import OpticalFly, load_ffsg  # noqa: E402
from train_play import Brain, Game, load_ffsg as load_ffsg_np  # noqa: E402


def feature_dn(model: OpticalFly, feats: torch.Tensor, feature_to_cell: list[int]) -> torch.Tensor:
    b = feats.shape[0]
    h = feats.new_zeros(b, model.n)
    u = feats.new_zeros(b, model.n)
    for f, cell in enumerate(feature_to_cell):
        if 0 <= cell < model.n:
            u[:, cell] = 2.0 * (feats[:, f] - 0.5)
    for _ in range(3):
        h = model.step_graph(h, u)
    return 4.0 * h[:, model.dn_idx]


@torch.no_grad()
def rollout_pipes(model: OpticalFly, seeds: list[int], device: torch.device) -> float:
    model.eval()
    scores = []
    for s in seeds:
        g = Game(s)
        h = None
        stack = None
        while g.alive and g.frame < 60 * 60:
            if g.frame % 2 == 0:
                obs_np, stack = render_obs(g, stack)
                obs = torch.tensor(obs_np[None], device=device)
                logits, h, _ = model(obs, h)
                g.step(1 if logits[0, 0] > logits[0, 1] else 0)
            else:
                g.step(0)
        scores.append(g.score)
    return float(np.mean(scores))


@torch.no_grad()
def agreement_with_features(
    model: OpticalFly, feat_brain: Brain, seeds: list[int], device: torch.device
) -> float:
    model.eval()
    agree, total = 0, 0
    for s in seeds:
        g = Game(s)
        h = None
        stack = None
        feat_brain.reset()
        while g.alive and g.frame < 60 * 30:
            if g.frame % 2 == 0:
                obs_np, stack = render_obs(g, stack)
                feat = g.features()
                logits, h, _ = model(torch.tensor(obs_np[None], device=device), h)
                o = 1 if logits[0, 0] > logits[0, 1] else 0
                f = feat_brain.decide_from_acts(feat_brain.dn_acts(feat))
                agree += int(o == f)
                total += 1
                g.step(f)  # follow feature teacher trajectory
            else:
                g.step(0)
    return agree / max(1, total)


def collect_champion(
    model: OpticalFly,
    feat_brain: Brain,
    n_courses: int,
    seed0: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Roll out feature champion; store obs + features + champion flap labels."""
    model.eval()
    obs_list, feat_list, y_list = [], [], []
    pipes = []
    with torch.no_grad():
        for i in range(n_courses):
            g = Game(seed0 + i)
            h = None
            stack = None
            feat_brain.reset()
            while g.alive and g.frame < 60 * 60:
                if g.frame % 2 == 0:
                    obs_np, stack = render_obs(g, stack)
                    feat = g.features().astype(np.float32)
                    acts = feat_brain.dn_acts(feat)
                    label = feat_brain.decide_from_acts(acts)
                    obs_list.append(obs_np)
                    feat_list.append(feat)
                    y_list.append(label)
                    # Mix a little of optical action for DAgger coverage later
                    g.step(label)
                else:
                    g.step(0)
            pipes.append(g.score)
    return (
        torch.tensor(np.stack(obs_list), dtype=torch.float32, device=device),
        torch.tensor(np.stack(feat_list), dtype=torch.float32, device=device),
        torch.tensor(y_list, dtype=torch.long, device=device),
        float(np.mean(pipes)),
    )


def collect_dagger(
    model: OpticalFly,
    feat_brain: Brain,
    n_courses: int,
    seed0: int,
    beta: float,
    rng: np.random.Generator,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    model.eval()
    obs_list, feat_list, y_list = [], [], []
    pipes = []
    with torch.no_grad():
        for i in range(n_courses):
            g = Game(seed0 + i)
            h = None
            stack = None
            feat_brain.reset()
            while g.alive and g.frame < 60 * 60:
                if g.frame % 2 == 0:
                    obs_np, stack = render_obs(g, stack)
                    feat = g.features().astype(np.float32)
                    acts = feat_brain.dn_acts(feat)
                    teacher = feat_brain.decide_from_acts(acts)
                    logits, h, _ = model(torch.tensor(obs_np[None], device=device), h)
                    learner = 1 if logits[0, 0] > logits[0, 1] else 0
                    obs_list.append(obs_np)
                    feat_list.append(feat)
                    y_list.append(teacher)
                    g.step(teacher if rng.random() < beta else learner)
                else:
                    g.step(0)
            pipes.append(g.score)
    return (
        torch.tensor(np.stack(obs_list), dtype=torch.float32, device=device),
        torch.tensor(np.stack(feat_list), dtype=torch.float32, device=device),
        torch.tensor(y_list, dtype=torch.long, device=device),
        float(np.mean(pipes)),
    )


def fit_encoder(
    model: OpticalFly,
    obs: torch.Tensor,
    feats: torch.Tensor,
    y: torch.Tensor,
    feature_to_cell: list[int],
    steps: int,
    lr: float,
) -> None:
    model.train()
    for p in model.readout.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(model.encoder.parameters(), lr=lr)
    n = obs.shape[0]
    ch_w = torch.tensor([1.0, 6.0, 2.0, 1.0, 0.2, 6.0], device=obs.device)
    for step in range(steps):
        idx = torch.randint(0, n, (min(512, n),), device=obs.device)
        drive_t = (feats[idx] - 0.5).clamp(-1, 1)
        logits, h, drives = model(obs[idx], None)
        with torch.no_grad():
            dn_t = feature_dn(model, feats[idx], feature_to_cell)
            teacher_y = model.readout(dn_t).argmax(-1)
        dn_p = 4.0 * h[:, model.dn_idx]
        inj = F.smooth_l1_loss(drives, drive_t, reduction="none")
        inj = (inj * ch_w).mean()
        dn_loss = F.smooth_l1_loss(dn_p, dn_t)
        # Strong action match vs feature champion (through frozen readout)
        flap_w = torch.tensor([1.0, 3.0], device=obs.device)  # upweight flap class
        ce = F.cross_entropy(logits, teacher_y, weight=flap_w)
        # Also match labels collected on trajectory
        ce2 = F.cross_entropy(logits, y[idx], weight=flap_w)
        loss = 8.0 * inj + 4.0 * dn_loss + 1.5 * ce + 0.5 * ce2
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 150 == 0:
            with torch.no_grad():
                acc = (logits.argmax(-1) == teacher_y).float().mean().item()
                # per-channel mse for vy / err
                mse = ((drives - drive_t) ** 2).mean(0)
            print(
                f"  bc {step:4d} loss={loss.item():.3f} inj={inj.item():.4f} "
                f"dn={dn_loss.item():.4f} ce={ce.item():.3f} acc={acc:.3f} "
                f"vy_mse={mse[1].item():.4f} err_mse={mse[5].item():.4f}"
            )


def load_readout_init(model: OpticalFly, path: Path) -> None:
    if not path.exists():
        return
    raw = np.fromfile(path, dtype=np.float32)
    nd, H = model.n_dn, 16
    need = H * nd + H + 2 * H + 2
    if raw.size < need:
        return
    w1 = raw[: H * nd].reshape(H, nd)
    b1 = raw[H * nd : H * nd + H]
    w2 = raw[H * nd + H : H * nd + H + 2 * H].reshape(2, H)
    b2 = raw[-2:]
    with torch.no_grad():
        model.readout[0].weight.copy_(torch.tensor(w1))
        model.readout[0].bias.copy_(torch.tensor(b1))
        model.readout[2].weight.copy_(torch.tensor(w2))
        model.readout[2].bias.copy_(torch.tensor(b2))
    print(f"  seeded readout from {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph", type=Path, default=ROOT / "data" / "processed" / "subgraph_v1.bin")
    ap.add_argument("--meta", type=Path, default=ROOT / "data" / "processed" / "subgraph_v1.json")
    ap.add_argument(
        "--readout-init",
        type=Path,
        default=ROOT / "results" / "play_train" / "champion_params.bin",
    )
    ap.add_argument("--courses", type=int, default=48)
    ap.add_argument("--dagger-rounds", type=int, default=6)
    ap.add_argument("--bc-steps", type=int, default=1200)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "optical")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    device = torch.device(args.device)
    g = load_ffsg(args.graph)
    meta = json.loads(args.meta.read_text())
    inject = [c for c in g["feature_to_cell"] if c >= 0]
    meta["visual_to_cell"] = inject
    meta["optical_inject"] = "feature_to_cell"
    print(f"graph n={g['n']} inject={len(inject)} obs=4x64x64")

    model = OpticalFly(g, inject).to(device)
    load_readout_init(model, args.readout_init)
    for p in model.readout.parameters():
        p.requires_grad_(False)

    feat_brain = Brain(load_ffsg_np(args.graph))
    feat_brain.set_params(np.fromfile(args.readout_init, dtype=np.float32).astype(np.float64))

    rng = np.random.default_rng(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    play = ROOT / "play" / "brain"
    play.mkdir(parents=True, exist_ok=True)

    eval_seeds = list(range(args.seed + 300_000, args.seed + 300_016))
    best_pipes = -1.0
    best_agree = -1.0
    best_state = None

    def save(tag: str, pipes: float, agree: float) -> None:
        nonlocal best_pipes, best_agree, best_state
        best_pipes, best_agree = pipes, agree
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.export_bundle(args.out / "optical.json", meta)
        # Force champion readout bytes into export
        shutil.copy2(args.readout_init, args.out / "optical_readout.bin")
        for name in ("optical.json", "optical_encoder.bin", "optical_readout.bin"):
            shutil.copy2(args.out / name, play / name)
        shutil.copy2(args.graph, play / "subgraph_v1.bin")
        meta["optical_best_pipes"] = pipes
        meta["optical_agree"] = agree
        (play / "subgraph_v1.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"  saved ({tag}) pipes={pipes:.2f} agree={agree:.3f} → play/brain/")

    print("bootstrap on feature-champion rollouts…")
    obs, feats, y, mean_p = collect_champion(
        model, feat_brain, args.courses, args.seed, device
    )
    print(f"  samples={len(y)} champ_pipes={mean_p:.2f} flap_rate={y.float().mean():.3f}")
    fit_encoder(model, obs, feats, y, g["feature_to_cell"], args.bc_steps, args.lr)

    X_all, F_all, y_all = obs, feats, y
    for rnd in range(args.dagger_rounds):
        beta = max(0.2, 1.0 - (rnd + 1) / (args.dagger_rounds + 1))
        print(f"DAgger round {rnd + 1}/{args.dagger_rounds} beta={beta:.2f}")
        obs, feats, y, mean_p = collect_dagger(
            model,
            feat_brain,
            args.courses,
            args.seed + 10_000 * (rnd + 1),
            beta,
            rng,
            device,
        )
        print(f"  collect_pipes={mean_p:.2f} new={len(y)}")
        X_all = torch.cat([X_all, obs], 0)
        F_all = torch.cat([F_all, feats], 0)
        y_all = torch.cat([y_all, y], 0)
        if X_all.shape[0] > 80_000:
            keep = torch.randperm(X_all.shape[0], device=device)[:80_000]
            X_all, F_all, y_all = X_all[keep], F_all[keep], y_all[keep]
        fit_encoder(
            model, X_all, F_all, y_all, g["feature_to_cell"], args.bc_steps, args.lr * 0.7
        )

        pipes = rollout_pipes(model, eval_seeds, device)
        agree = agreement_with_features(model, feat_brain, eval_seeds[:8], device)
        print(f"  eval pipes={pipes:.2f} agree={agree:.3f}")
        if pipes > best_pipes + 1e-6 or (pipes >= best_pipes and agree > best_agree):
            save(f"round{rnd+1}", pipes, agree)

    if best_state is None:
        pipes = rollout_pipes(model, eval_seeds, device)
        agree = agreement_with_features(model, feat_brain, eval_seeds[:8], device)
        save("final", pipes, agree)
    else:
        model.load_state_dict(best_state)
        model.export_bundle(args.out / "optical.json", meta)
        shutil.copy2(args.readout_init, args.out / "optical_readout.bin")
        for name in ("optical.json", "optical_encoder.bin", "optical_readout.bin"):
            shutil.copy2(args.out / name, play / name)

    (args.out / "train.json").write_text(
        json.dumps(
            {
                "best_pipes": best_pipes,
                "best_agree": best_agree,
                "eval_seeds": eval_seeds,
                "obs": [4, 64, 64],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"done best_pipes={best_pipes:.2f} agree={best_agree:.3f}")


if __name__ == "__main__":
    main()
