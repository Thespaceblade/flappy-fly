# Notes: how the viral fly-brain game demos work

## The dataset (what Google actually released)

**MaleCNS v1.0** is a *connectome*: a synaptic wiring map of an adult male fruit fly's central nervous system (~166,700 neurons, ~125M synaptic contacts). It is not a pretrained "AI model" that already knows how to play games. You (or a simulator) still have to:

- pick neuron dynamics (how activity evolves over time)
- decide how game pixels become sensory input
- decide how neural activity becomes button presses
- optionally add plasticity / RL so weights or policies change with experience

Official portals: https://male-cns.janelia.org/ and neuPrint at Janelia.

## Beat Saber demo (Lyra Bubbles / @_lyraaaa_)

**No public GitHub repo found** as of scaffold time. Understanding comes from the author's X thread clarifying the viral clip.

Pipeline (as described by the author):

1. **Motor stage first.** The "motor" side of the network was overfit to a recorded play / movement sequence (teacher forcing / replay distillation). Motion is still *produced by the connectome*, not a hard-coded animation of saber swings.
2. **Not pure replay.** Replay / teacher signals were mixed into inputs early in training, then ramped down so the network had to generate motion itself.
3. **Vision still in progress.** Visual reactivity was being trained separately; play quality was poor while that was incomplete.
4. **Next step: RL.** Reinforcement learning to improve reactive play patterns with less reliance on external teacher signals.
5. **Caveat.** Overfit to one track at the time of the demo — not a general Beat Saber agent.

Useful mental model: *distill a motor policy into the connectome, then wean it onto sensory-driven control + RL.*

## Open reference: DOOMFLY

Repo: https://github.com/nftechie/doomfly

This is the clearest open implementation of the same genre of experiment.

Closed loop each frame:

```
ViZDoom pixels
  → proxy retinal inputs (thousands of R1–R6 / R8 cells)
  → full retained MaleCNS graph (~166k neurons, ~25.6M directed edges)
  → fixed readout cells (e.g. DNp20 → turn, DNpe017 → move/fire)
  → game buttons
```

Learning attempt (experimental, author reports validation failures so far):

- Nonfatal damage → brief artificial drive into dopamine cells (PPL101)
- Plasticity on a small subset of existing KC→MBON connections
- Rest of wiring + controller mapping stay fixed

Key honesty from that project: engineered sensory proxies + engineered motor mapping ≠ validated fly motor commands. Changing a few weights does not automatically equal "the fly learned Doom."

## Implications for Flappy Bird

Flappy Bird is a simpler control problem than Beat Saber or Doom (one action: flap), so it is a good first game for this idea.

Likely stages:

1. **Baseline open loop** — freeze wiring, hand-pick sensory encode + flap decode, see if anything coherent happens.
2. **Imitation / distillation** — record a competent human or scripted flapper; drive / supervise the connectome toward that motor pattern (Beat Saber-style).
3. **RL** — reward survival / pipes cleared while reducing teacher forcing (DOOMFLY-style dopamine plasticity, or an outer RL loop around the readout / a plastic subgraph).

## Related but different

- **flybody** (DeepMind / TuragaLab): biomechanical body + RL locomotion — not the MaleCNS game-control meme.
- **NeuroCraft Fly**: Minecraft entity driven by MaleCNS activity with scripted body programs.
- **flygym / NeuroMechFly**: sensorimotor digital twin for neuroscience, not Flappy Bird.
