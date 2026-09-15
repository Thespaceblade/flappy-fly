# Research protocol: does MaleCNS topology help Flappy Bird?

**Status:** Phase 1 complete (C Flappy env + baselines); subgraph / CEM next  
**Environment ID:** `flappy-fly-malecns-topology-v1`

## Our claim

A trainable flap readout whose only inputs are activities of a fixed MaleCNS-derived circuit will outperform **topology-shuffled** and **circuit-silenced** controls under matched training budgets on Flappy Bird.


## Hypothesis

**H1 (primary):** With only the readout trained, real MaleCNS wiring yields higher held-out survival than an edge-shuffled graph with the same cells, same degree sequence (or same weight multiset), same dynamics, same encoder, same training budget.

**H0:** Real and shuffled topologies are statistically indistinguishable on held-out pipes / survival.

**Secondary checks:**
- Silenced circuit + trained readout ≈ untrained / idle (causal path through the circuit)
- Direct engineered features → same readout (no circuit) as an upper-bound engineered baseline (not a topology claim)

## Architecture

```
Flappy state (engineered features)
  → fixed encoder onto visual cells
  → fixed sparse MaleCNS subgraph dynamics
  → descending-cell activities
  → trainable readout → flap | no-flap
```

Only readout parameters learn. Graph edges, signs, and recurrence stay frozen during training (except in the shuffled *control*, where edges are randomized once before training and then frozen).

## Locked design choices (v1)

| Choice | Decision |
| --- | --- |
| Game | Original Flappy Bird measurements via [THEN00P/reFlappy](https://github.com/THEN00P/reFlappy); sprites via [nebez/floppybird](https://github.com/nebez/floppybird). See [THIRD_PARTY.md](THIRD_PARTY.md). |
| Action | Binary flap at decision rate **30 Hz** (hold physics at 60 Hz) |
| Episode | Ends on collision or **60 s** cap |
| Fitness | Mean pipes cleared over training courses (primary); survival seconds logged |
| Graph | Bounded MaleCNS visual→descending subgraph, target **~80 cells** (Fly Dino–scale) |
| Dynamics | Signed leaky-tanh recurrence (same family as Fly Dino); 3 steps per decision |
| Readout | Small MLP: `n_DN → 12 tanh → 2 logits` (flap / idle); argmax or Bernoulli |
| Trainer | Cross-entropy method (CEM) / neuroevolution — score-based, no backprop through graph |
| Seeds | **Generalize across seeds** (not Beat Saber–style single-track overfitting). Train on many random courses; validate on a fixed held-out set; report once on a predeclared test set. Disjoint train / val / test ranges. |

### Observation encoder (engineered, not pixels)

Channels drive assigned visual cell types (assignment fixed before training; no biological claim):

| # | Feature | Encoding |
| --- | --- | --- |
| 0 | Bird y (normalized) | `[0,1]` playfield fraction |
| 1 | Bird vy | mapped to `[0,1]` |
| 2 | Next pipe gap y | `[0,1]` |
| 3 | Horizontal distance to pipe | near→1, far→0 |
| 4 | Gap half-height / playfield | scale cue |
| 5 | Bird vs gap vertical error | signed, mapped to `[0,1]` |

Exact formulas live in code and must match train / eval / C runtime.

### Subgraph selection (anatomy only, before any training)

Deterministic procedure (to be implemented exactly and hashed):

1. Start from MaleCNS v1.0 flat tables (min confidence 0.5), pin SHA-256.
2. Prefer visual types with strong direct contacts onto descending neurons (LC\* / LPLC\* family as available in v1.0 annotations).
3. Take a small set of descending targets (prefer motor-relevant DN types when annotated).
4. Fill intermediates by two-hop bridge strength until ~80 cells.
5. Keep **every** measured directed edge among selected cells; do not invent edges.
6. Transmitter signs: ACh +, GABA/Glu −, unclear 0 (document counts).

Export: `data/processed/subgraph_v1.{npz,json}` + `manifest.json` with source hashes.

### Controls (same budget each)

| Arm | What changes |
| --- | --- |
| **Real** | Measured edges |
| **Shuffle** | Edges rewired (preserve cell set; prefer degree-preserving shuffle); weights remapped; then freeze |
| **Silence** | Circuit activity forced to 0; readout still trained |
| **No-circuit** | Same features → readout directly (optional ceiling baseline) |
| **Untrained** | Random readout, no CEM |
| **Idle / random flap** | Non-learning baselines |

Primary comparison: **Real vs Shuffle** on held-out test seeds.

## Evaluation protocol

1. **Train** on shared courses per CEM generation (N candidates × K courses).
2. **Validate** each generation on fixed validation seeds; keep best validation champion.
3. **Test once** on predeclared held-out seeds (e.g. 100 episodes). No test-based selection.
4. Report: pipes cleared, survival seconds, flaps, completion rate; mean ± across training replicas (min 3 seeds).
5. Publish checkpoints, training curves, and per-seed JSON.

### Success criteria (v1)

- **Learning works:** Real arm ≫ idle / untrained / silence on held-out.
- **Topology claim supported:** Real mean pipes **significantly > Shuffle** under matched seeds/budget (predeclare test; report effect size; do not p-hack).
- **Topology claim unsupported:** Real ≈ Shuffle → publish as negative / null result (still valuable).

## Implementation plan

| Phase | Deliverable | Language |
| --- | --- | --- |
| 0 | This protocol + repo layout | — |
| 1 | Flappy env (seeded, headless + window) | C (SDL optional) |
| 2 | Fetch MaleCNS + build/hash subgraph | Python |
| 3 | Sparse dynamics + readout step | C |
| 4 | CEM trainer calling C via CLI/FFI | Python or C |
| 5 | Ablation harness + JSON reports | Python/shell |
| 6 | Replicate seeds + write results | — |

## Provenance

- Connectome: MaleCNS v1.0, FlyEM / HHMI Janelia + collaborators, CC BY 4.0 — https://male-cns.janelia.org/
- Method cousins: [cobanov/flyjump](https://github.com/cobanov/flyjump) (circuit + CEM readout), [nftechie/doomfly](https://github.com/nftechie/doomfly) (full-graph honesty), [cobanov/awesome-fly](https://github.com/cobanov/awesome-fly)

## Changelog

- **2026-09-14:** v1 protocol locked (topology Real vs Shuffle primary).
