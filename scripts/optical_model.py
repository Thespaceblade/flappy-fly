"""Differentiable MaleCNS dynamics + tiny CNN optical encoder (torch)."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HIDDEN = 16  # match c/brain.h


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


def visual_cells_from_meta(meta: dict) -> list[int]:
    if "visual_to_cell" in meta:
        return list(meta["visual_to_cell"])
    cells = [i for i, n in enumerate(meta.get("nodes", [])) if n.get("role") == "input"]
    if not cells:
        raise SystemExit("no visual/input cells in subgraph meta")
    return cells


def dense_from_csr(g: dict) -> np.ndarray:
    n = g["n"]
    W = np.zeros((n, n), dtype=np.float64)
    for post in range(n):
        a, b = g["row_ptr"][post], g["row_ptr"][post + 1]
        for k in range(a, b):
            W[post, g["col_idx"][k]] = g["weight"][k]
    return W


class OpticalEncoder(nn.Module):
    """4×64×64 frame stack → n_out drives in (-1, 1)."""

    def __init__(self, n_out: int):
        super().__init__()
        self.n_out = n_out
        self.conv1 = nn.Conv2d(4, 32, 5, stride=2, padding=2)  # 32×32
        self.conv2 = nn.Conv2d(32, 64, 5, stride=2, padding=2)  # 16×16
        self.conv3 = nn.Conv2d(64, 64, 5, stride=2, padding=2)  # 8×8
        self.fc = nn.Linear(64 * 8 * 8, 128)
        self.out = nn.Linear(128, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.flatten(1)
        h = F.relu(self.fc(h))
        return torch.tanh(self.out(h))


class OpticalFly(nn.Module):
    """obs → CNN → visual drives → frozen graph (3 steps) → DN → logits.

    By default drives the same 6 feature_to_cell injection sites as the
    engineered-feature path (optical substitute for f[0..5]). Pass
    visual_to_cell=all inputs to widen later.
    """

    def __init__(self, g: dict, inject_cells: list[int]):
        super().__init__()
        self.n = g["n"]
        self.n_dn = g["n_dn"]
        self.visual_to_cell = list(inject_cells)
        self.n_visual = len(inject_cells)
        self.encoder = OpticalEncoder(self.n_visual)

        W = dense_from_csr(g)
        self.register_buffer("W", torch.tensor(W, dtype=torch.float32))
        self.register_buffer(
            "visual_idx", torch.tensor(inject_cells, dtype=torch.long)
        )
        self.register_buffer(
            "dn_idx", torch.tensor(g["dn_index"][: self.n_dn], dtype=torch.long)
        )

        nd = self.n_dn
        self.readout = nn.Sequential(
            nn.Linear(nd, HIDDEN),
            nn.Tanh(),
            nn.Linear(HIDDEN, 2),
        )

    def step_graph(self, h: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        # h,u: [B,n]
        acc = u + 1.4 * (h @ self.W.T)
        return 0.3 * h + 0.7 * torch.tanh(acc)

    def forward(
        self, obs: torch.Tensor, h: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        obs: [B,4,64,64] frame stack
        h: [B,n] or None
        returns logits [B,2], h_new [B,n], drives [B,n_visual]
        """
        b = obs.shape[0]
        if h is None:
            h = obs.new_zeros(b, self.n)
        drives = self.encoder(obs)
        u = obs.new_zeros(b, self.n)
        u[:, self.visual_idx] = 2.0 * drives
        for _ in range(3):
            h = self.step_graph(h, u)
        dn = 4.0 * h[:, self.dn_idx]
        logits = self.readout(dn)
        return logits, h, drives

    def export_bundle(self, path: Path, meta: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        sd = self.state_dict()
        enc = {k: v.detach().cpu().numpy() for k, v in sd.items() if k.startswith("encoder.")}
        order = [
            "encoder.conv1.weight",
            "encoder.conv1.bias",
            "encoder.conv2.weight",
            "encoder.conv2.bias",
            "encoder.conv3.weight",
            "encoder.conv3.bias",
            "encoder.fc.weight",
            "encoder.fc.bias",
            "encoder.out.weight",
            "encoder.out.bias",
        ]
        flat = []
        shapes = {}
        for k in order:
            arr = enc[k]
            shapes[k] = list(arr.shape)
            flat.append(arr.astype(np.float32).ravel())
        enc_bin = path.with_name("optical_encoder.bin")
        enc_bin.write_bytes(np.concatenate(flat).tobytes())

        w1 = self.readout[0].weight.detach().cpu().numpy().astype(np.float32)
        b1 = self.readout[0].bias.detach().cpu().numpy().astype(np.float32)
        w2 = self.readout[2].weight.detach().cpu().numpy().astype(np.float32)
        b2 = self.readout[2].bias.detach().cpu().numpy().astype(np.float32)
        readout = np.concatenate([w1.ravel(), b1, w2.ravel(), b2])
        read_bin = path.with_name("optical_readout.bin")
        read_bin.write_bytes(readout.tobytes())

        bundle = {
            "version": "flappy-fly-optical-v2",
            "obs": [4, 64, 64],
            "stack": 4,
            "n_visual": self.n_visual,
            "visual_to_cell": self.visual_to_cell,
            "n_dn": self.n_dn,
            "hidden": HIDDEN,
            "encoder_shapes": shapes,
            "encoder_bin": enc_bin.name,
            "readout_bin": read_bin.name,
            "channels": meta.get("channels", []),
            "note": "4-frame 64² stack → inject cells; frozen MaleCNS + DN readout",
        }
        path.write_text(json.dumps(bundle, indent=2) + "\n")
