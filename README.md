# Flappy Fly

**Research question:** Does a fixed MaleCNS-derived circuit help a trained flap readout play Flappy Bird better than a topology-shuffled control?

See [RESEARCH.md](RESEARCH.md). Game **mechanics** come from the original Android Flappy Bird via [reFlappy](https://github.com/THEN00P/reFlappy); **sprites** from [floppybird](https://github.com/nebez/floppybird). Attribution: [THIRD_PARTY.md](THIRD_PARTY.md).

## Play (original look + original numbers)

```sh
python3 -m http.server 8765 --bind 127.0.0.1
# open http://127.0.0.1:8765/play/
```

**Brain (eye)** uses a learned CNN on a 4×64×64 game-view stack → MaleCNS inject cells → DN readout.  
**Brain (feat)** is the older engineered-feature champion.

```sh
.venv/bin/python scripts/train_optical.py
```

## Headless C env (same physics)

```sh
make -C c test
./c/flappy_fly rollout --seed 42 --policy rule
python3 scripts/train_cem.py
```

## Layout

```
c/                   # reFlappy-faithful simulator + brain stub
play/                # browser UI (floppybird sprites + reFlappy physics)
vendor/floppybird/   # upstream assets
RESEARCH.md          # Real vs Shuffle protocol
THIRD_PARTY.md       # attribution
```
