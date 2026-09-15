#!/usr/bin/env python3
"""Pixel-art fruit-fly sprite maker (Flappy Bird style, 34x24).

Fills a similar visual footprint to yellowbird-*.png. Sim hitbox stays 20x20.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "play" / "assets" / "fly"

K = (40, 28, 18, 255)
BODY = (92, 64, 40, 255)
BODY2 = (130, 92, 55, 255)
AB = (70, 48, 30, 255)
STR = (48, 32, 20, 255)
WING = (200, 220, 235, 255)
WING2 = (155, 180, 205, 255)
EYE = (255, 255, 255, 255)
PUPIL = (20, 20, 20, 255)
LEG = (200, 120, 50, 255)
CLR = (0, 0, 0, 0)


def put(pix, x, y, c) -> None:
    if 0 <= x < 34 and 0 <= y < 24:
        pix[x, y] = c


def oval(pix, cx, cy, rx, ry, c) -> None:
    for y in range(cy - ry, cy + ry + 1):
        for x in range(cx - rx, cx + rx + 1):
            if rx and ry and ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.08:
                put(pix, x, y, c)


def outline(im: Image.Image) -> None:
    pix = im.load()
    mark = []
    for y in range(24):
        for x in range(34):
            if pix[x, y][3] < 128:
                continue
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nx, ny = x + dx, y + dy
                if not (0 <= nx < 34 and 0 <= ny < 24) or pix[nx, ny][3] < 128:
                    mark.append((x, y))
                    break
    for x, y in mark:
        if pix[x, y][:3] != EYE[:3]:
            pix[x, y] = K


def draw_fly(mode: str) -> Image.Image:
    im = Image.new("RGBA", (34, 24), CLR)
    pix = im.load()
    oval(pix, 14, 13, 10, 7, AB)
    for sx in (6, 9, 12, 15):
        for y in range(8, 19):
            put(pix, sx, y, STR)
    oval(pix, 20, 12, 8, 8, BODY)
    oval(pix, 21, 11, 5, 5, BODY2)
    oval(pix, 27, 11, 6, 7, BODY)
    oval(pix, 28, 10, 4, 4, EYE)
    put(pix, 29, 10, PUPIL)
    put(pix, 29, 11, PUPIL)
    put(pix, 30, 10, PUPIL)
    for pts in (
        (12, 20),
        (11, 21),
        (10, 22),
        (17, 20),
        (17, 22),
        (16, 22),
        (23, 19),
        (24, 21),
        (25, 22),
    ):
        put(pix, *pts, LEG)
    if mode == "up":
        oval(pix, 16, 4, 10, 4, WING)
        oval(pix, 16, 3, 7, 2, WING2)
    elif mode == "mid":
        oval(pix, 11, 10, 10, 4, WING)
        oval(pix, 10, 10, 7, 2, WING2)
    else:
        oval(pix, 15, 19, 10, 4, WING)
        oval(pix, 15, 20, 7, 2, WING2)
    outline(im)
    oval(pix, 28, 10, 4, 4, EYE)
    put(pix, 29, 10, PUPIL)
    put(pix, 29, 11, PUPIL)
    put(pix, 30, 10, PUPIL)
    return im


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, mode in (
        ("0_upflap", "up"),
        ("1_midflap", "mid"),
        ("2_downflap", "down"),
    ):
        path = OUT / f"{name}.png"
        draw_fly(mode).save(path)
        print("wrote", path)


if __name__ == "__main__":
    main()
