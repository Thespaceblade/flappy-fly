# Flappy Fly

Can the Google/HHMI **MaleCNS v1.0** fruit-fly connectome learn to play Flappy Bird?

This project explores wiring a simulated adult male *Drosophila* CNS (~166k neurons) into a Flappy Bird environment: sensory frames in → connectome dynamics → flap / no-flap out → optional learning.

## What this is

An experimental research playground, not a claim that a reconstructed fly "understands" the game. The connectome is a wiring diagram; dynamics, sensory encoding, and action decoding are engineering choices we define.

## Upstream resources

| Resource | Role |
| --- | --- |
| [MaleCNS connectome](https://male-cns.janelia.org/) | Official dataset (Google Research + HHMI Janelia + collaborators) |
| [nftechie/doomfly](https://github.com/nftechie/doomfly) | Best-documented open reference: full MaleCNS loop on Doom |
| [@_lyraaaa_ Beat Saber thread](https://x.com/_lyraaaa_/status/2097527368919470162) | Viral motor-distillation + RL demo (no public repo found) |
| [TuragaLab/flybody](https://github.com/TuragaLab/flybody) | Biomechanical fly body / RL tasks in MuJoCo (different goal) |

## Approach sketch

Inspired by DOOMFLY and the Beat Saber experiment:

1. **Encode** each Flappy Bird frame into a proxy sensory drive (e.g. brightness / motion onto optic-lobe or other chosen input neurons).
2. **Simulate** spike / rate dynamics on the retained MaleCNS graph.
3. **Decode** activity from a small set of readout neurons into a binary flap action.
4. **Train** in stages: optional teacher-forced / replay distillation for motor patterning, then reinforcement learning with less external forcing.

Details and code land here as the experiment is built.

## Status

Repo scaffold only. Simulation loop, Flappy Bird env, and training pipeline are next.
