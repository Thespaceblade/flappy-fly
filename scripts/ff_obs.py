"""Low-res Flappy observation shared by optical train + play (must match play/optical.js)."""

from __future__ import annotations

import numpy as np

OBS_H = 64
OBS_W = 64
OBS_C = 4  # stacked frames (newest first) for motion / vy
STACK = 4
W, H = 288, 512
BIRD_W, BIRD_H = 20, 20
BIRD_X = 80
PIPE_W, PIPE_H, PIPE_GAP = 52, 320, 96
GROUND_Y = 400
# Optical bird drawn larger than physics box so motion is visible at 64².
OPT_BIRD = 8


def render_frame(game) -> np.ndarray:
    """Return float32 [H, W] in [0, 1]: playfield crop (sky/pipes/bird/ground)."""
    img = np.full((OBS_H, OBS_W), 0.12, dtype=np.float32)
    # Map full width × playfield height (to ground) into the square.
    sx = OBS_W / W
    sy = OBS_H / GROUND_Y

    ground_r = OBS_H - max(2, int(0.08 * OBS_H))
    img[ground_r:, :] = 0.45

    for p in game.pipes:
        x0 = int(p["x"] * sx)
        x1 = int((p["x"] + PIPE_W) * sx)
        x0 = max(0, min(OBS_W, x0))
        x1 = max(0, min(OBS_W, x1))
        if x1 <= x0:
            continue
        top_y = (p["height"] - PIPE_H) - PIPE_GAP
        bot_y = p["height"]
        y0 = int(top_y * sy)
        y1 = int((top_y + PIPE_H) * sy)
        y0 = max(0, min(OBS_H, y0))
        y1 = max(0, min(OBS_H, y1))
        if y1 > y0:
            img[y0:y1, x0:x1] = 0.8
        y0 = int(bot_y * sy)
        y1 = int((bot_y + PIPE_H) * sy)
        y0 = max(0, min(OBS_H, y0))
        y1 = max(0, min(OBS_H, y1))
        if y1 > y0:
            img[y0:y1, x0:x1] = 0.8
        # gap tint
        gap0 = int((bot_y - PIPE_GAP) * sy)
        gap1 = int(bot_y * sy)
        gap0 = max(0, min(OBS_H, gap0))
        gap1 = max(0, min(OBS_H, gap1))
        if gap1 > gap0:
            img[gap0:gap1, x0:x1] = np.maximum(img[gap0:gap1, x0:x1], 0.28)

    # Oversized bird blob (optical readability, not collision box).
    cy = int((game.bird_y + BIRD_H * 0.5) * sy)
    cx = int((BIRD_X + BIRD_W * 0.5) * sx)
    half = OPT_BIRD // 2
    y0, y1 = max(0, cy - half), min(OBS_H, cy + half)
    x0, x1 = max(0, cx - half), min(OBS_W, cx + half)
    if y1 > y0 and x1 > x0:
        img[y0:y1, x0:x1] = 1.0
        # Motion streak: extend opposite to velocity so rising bird leaves a trail below.
        streak = int(np.clip(game.bird_vy * 1.2, -10, 10))
        if streak > 0:
            ys0, ys1 = max(0, cy - half - streak), y0
            if ys1 > ys0:
                img[ys0:ys1, x0:x1] = np.maximum(img[ys0:ys1, x0:x1], 0.65)
        elif streak < 0:
            ys0, ys1 = y1, min(OBS_H, cy + half - streak)
            if ys1 > ys0:
                img[ys0:ys1, x0:x1] = np.maximum(img[ys0:ys1, x0:x1], 0.65)

    return img


def render_obs(game, stack: list[np.ndarray] | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return (obs[STACK,H,W], updated_stack). Newest frame in channel 0."""
    cur = render_frame(game)
    if stack is None:
        stack = []
    stack = [cur] + list(stack)
    stack = stack[:STACK]
    while len(stack) < STACK:
        stack.append(cur)
    obs = np.stack(stack[:STACK], axis=0).astype(np.float32)
    return obs, stack
