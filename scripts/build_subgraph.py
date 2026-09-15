#!/usr/bin/env python3
"""Build MaleCNS visual→DN subgraph for Flappy Fly.

Anatomy-only selection (Fly Dino–style), adapted to our 6 engineered features.
Default ~200 cells (within C FF_BRAIN_MAX_N=256).
Exports:
  data/processed/subgraph_v1.json
  data/processed/subgraph_v1.bin      (CSR for C)
  data/processed/subgraph_v1_shuffle.bin
  data/processed/manifest.json

Requires: pip install pyarrow numpy
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "malecns_v1"
OUT = ROOT / "data" / "processed"

ANN = RAW / "body-annotations-male-cns-v1.0-minconf-0.5.feather"
EDGES = RAW / "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
NT = RAW / "body-neurotransmitters-male-cns-v1.0.feather"

# One visual type per Flappy feature channel (engineered, not biology).
VISUAL_TYPES = ["LC4", "LC11", "LC9", "LC15", "LC16", "LC17"]
N_VISUAL_PER_TYPE = 8
N_DN = 32
N_BRIDGES = 140

SIGNS = {"acetylcholine": 1, "gaba": -1, "glutamate": -1}
MAGIC = b"FFSG"
VERSION = 1


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_graph():
    import pyarrow.feather as feather

    for p in (ANN, EDGES, NT):
        if not p.exists():
            raise SystemExit(f"missing {p}; run: python3 scripts/fetch_malecns.py")

    print("loading annotations…")
    rows = feather.read_table(
        ANN, columns=["bodyId", "type", "superclass", "somaLocation"]
    ).to_pylist()
    ann = {r["bodyId"]: r for r in rows if r.get("somaLocation")}

    print("loading edges…")
    e = feather.read_table(EDGES)
    # Column names may vary slightly across dumps — accept common aliases.
    cols = set(e.column_names)
    pre_col = "body_pre" if "body_pre" in cols else "pre"
    post_col = "body_post" if "body_post" in cols else "post"
    w_col = "weight" if "weight" in cols else "syn_count"
    pre = e[pre_col].to_numpy()
    post = e[post_col].to_numpy()
    weight = e[w_col].to_numpy().astype(np.float64)

    print("loading neurotransmitters…")
    nt_rows = feather.read_table(NT, columns=["body", "consensus_nt"]).to_pylist()
    nt = {r["body"]: r["consensus_nt"] for r in nt_rows}

    dn = np.array(
        [i for i, r in ann.items() if r.get("superclass") == "descending_neuron"],
        dtype=np.int64,
    )
    if len(dn) == 0:
        # Fallback: type name starts with DN
        dn = np.array(
            [
                i
                for i, r in ann.items()
                if isinstance(r.get("type"), str)
                and r["type"].startswith(("DN", "DNp", "DNg", "DNb"))
            ],
            dtype=np.int64,
        )
    print(f"  descending candidates: {len(dn)}")

    md = np.isin(post, dn)
    inputs: list[tuple[int, int]] = []
    targets: list[int] = []

    for channel, typ in enumerate(VISUAL_TYPES):
        ids = np.array([i for i, r in ann.items() if r.get("type") == typ], dtype=np.int64)
        if len(ids) == 0:
            raise SystemExit(f"no neurons with type {typ}")
        ix = np.flatnonzero(np.isin(pre, ids) & md)
        strength = {
            int(i): int(weight[ix[pre[ix] == i]].sum()) for i in np.unique(pre[ix])
        }
        chosen = sorted(strength, key=lambda i: (-strength[i], i))[:N_VISUAL_PER_TYPE]
        if len(chosen) < N_VISUAL_PER_TYPE:
            # Pad with any cells of that type by body id
            rest = sorted(set(ids.tolist()) - set(chosen))
            chosen = chosen + rest[: N_VISUAL_PER_TYPE - len(chosen)]
        if len(chosen) < N_VISUAL_PER_TYPE:
            raise SystemExit(f"not enough {typ} cells ({len(chosen)})")
        inputs.extend((i, channel) for i in chosen)
        jx = ix[np.isin(pre[ix], chosen)] if len(ix) else np.array([], dtype=np.int64)
        if len(jx):
            ranks = {
                int(i): int(weight[jx[post[jx] == i]].sum())
                for i in np.unique(post[jx])
            }
            targets.extend(sorted(ranks, key=lambda i: (-ranks[i], i))[:2])

    selected_inputs = [i for i, _ in inputs]
    ix = np.flatnonzero(np.isin(pre, selected_inputs) & md)
    strength = {
        int(i): int(weight[ix[post[ix] == i]].sum()) for i in np.unique(post[ix])
    } if len(ix) else {}
    targets = list(dict.fromkeys(targets))
    for i in sorted(strength, key=lambda i: (-strength[i], i)):
        if len(targets) >= N_DN:
            break
        if i not in targets:
            targets.append(i)
    while len(targets) < N_DN:
        # pad with any DN not already chosen
        for i in sorted(dn.tolist()):
            if i not in targets and i in ann:
                targets.append(int(i))
                break
        else:
            break

    imask = np.isin(pre, selected_inputs)
    omask = np.isin(post, targets)
    u, inv = np.unique(post[imask], return_inverse=True)
    incoming = dict(zip(u.tolist(), np.bincount(inv, weights=weight[imask]).tolist())) if len(u) else {}
    u, inv = np.unique(pre[omask], return_inverse=True)
    outgoing = dict(zip(u.tolist(), np.bincount(inv, weights=weight[omask]).tolist())) if len(u) else {}
    bridges = sorted(
        (
            int(i)
            for i in set(incoming) & set(outgoing)
            if i in ann and i not in selected_inputs and i not in targets
        ),
        key=lambda i: (-min(incoming[i], outgoing[i]), i),
    )[:N_BRIDGES]

    ids = sorted(set(selected_inputs + targets + bridges))
    idx = {i: j for j, i in enumerate(ids)}
    n = len(ids)

    mask = np.isin(pre, ids) & np.isin(post, ids)
    raw_edges = [
        (idx[int(a)], idx[int(b)], int(w))
        for a, b, w in zip(pre[mask], post[mask], weight[mask])
    ]

    # Signed normalized weights per postsynaptic cell (Fly Dino convention).
    # CSR: for each post, list of (pre, w_signed_norm)
    accum: list[list[tuple[int, float]]] = [[] for _ in range(n)]
    for pre_i, post_i, w in raw_edges:
        body = ids[pre_i]
        sign = SIGNS.get(nt.get(body), 0)
        if sign == 0:
            continue
        accum[post_i].append((pre_i, float(w) * sign))

    row_ptr = [0]
    col_idx: list[int] = []
    weights: list[float] = []
    for post_i in range(n):
        denom = sum(abs(w) for _, w in accum[post_i]) or 1.0
        for pre_i, w in accum[post_i]:
            col_idx.append(pre_i)
            weights.append(w / denom)
        row_ptr.append(len(col_idx))

    feature_to_cell = [-1] * 6
    # Drive the first chosen cell of each channel (strongest).
    seen_ch: set[int] = set()
    for body, ch in inputs:
        if ch in seen_ch:
            continue
        feature_to_cell[ch] = idx[body]
        seen_ch.add(ch)

    # All visual input cells (optical encoder targets).
    visual_to_cell = sorted({idx[i] for i, _ in inputs})

    dn_index = [idx[i] for i in targets if i in idx]

    nodes = []
    for i in ids:
        role = (
            "input"
            if i in selected_inputs
            else "output"
            if i in targets
            else "interneuron"
        )
        nodes.append(
            {
                "id": int(i),
                "type": ann[i].get("type"),
                "nt": nt.get(i),
                "sign": SIGNS.get(nt.get(i), 0),
                "role": role,
            }
        )

    graph = {
        "version": "flappy-fly-subgraph-v1",
        "n": n,
        "n_dn": len(dn_index),
        "feature_to_cell": feature_to_cell,
        "visual_to_cell": visual_to_cell,
        "dn_index": dn_index,
        "row_ptr": row_ptr,
        "col_idx": col_idx,
        "weight": weights,
        "nodes": nodes,
        "channels": VISUAL_TYPES,
        "inputs": [[idx[i], c] for i, c in inputs],
        "selection": {
            "visual_per_type": N_VISUAL_PER_TYPE,
            "n_dn": N_DN,
            "n_bridges": N_BRIDGES,
        },
    }
    return graph, {
        "annotations": sha256_file(ANN),
        "edges": sha256_file(EDGES),
        "neurotransmitters": sha256_file(NT),
    }


def write_bin(path: Path, graph: dict, shuffle: bool = False, seed: int = 0) -> None:
    n = graph["n"]
    row_ptr = list(graph["row_ptr"])
    col_idx = list(graph["col_idx"])
    weight = list(graph["weight"])
    feature_to_cell = list(graph["feature_to_cell"])
    dn_index = list(graph["dn_index"])

    if shuffle:
        rng = np.random.default_rng(seed)
        # Degree-preserving-ish: shuffle destinations among existing edge slots
        # Keep row_ptr (in-degree structure); shuffle which presynaptic cells
        # feed each post, sampling from the global pre multiset.
        pres = col_idx.copy()
        rng.shuffle(pres)
        col_idx = list(pres)
        # Also shuffle weight magnitudes with signs remapped randomly ±
        mags = [abs(w) for w in weight]
        rng.shuffle(mags)
        signs = rng.choice([-1.0, 1.0], size=len(mags))
        weight = [float(m * s) for m, s in zip(mags, signs)]
        # Re-normalize per post
        for post in range(n):
            a, b = row_ptr[post], row_ptr[post + 1]
            denom = sum(abs(weight[k]) for k in range(a, b)) or 1.0
            for k in range(a, b):
                weight[k] /= denom

    n_dn = len(dn_index)
    n_edges = len(col_idx)
    # Pack dn_index to fixed 32 slots (-1 pad)
    dn_pad = dn_index + [-1] * (32 - n_dn)
    if len(dn_pad) > 32:
        dn_pad = dn_pad[:32]
        n_dn = 32

    with path.open("wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<IIII", n, 6, n_dn, n_edges))
        f.write(struct.pack("<6i", *feature_to_cell))
        f.write(struct.pack("<32i", *dn_pad))
        f.write(struct.pack(f"<{n + 1}i", *row_ptr))
        f.write(struct.pack(f"<{n_edges}i", *col_idx))
        f.write(struct.pack(f"<{n_edges}f", *weight))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        print(json.dumps({"raw": str(RAW), "visual_types": VISUAL_TYPES}, indent=2))
        return

    graph, sources = build_graph()
    OUT.mkdir(parents=True, exist_ok=True)

    json_path = OUT / "subgraph_v1.json"
    # JSON without huge float lists for readability — keep compact
    json_path.write_text(
        json.dumps(
            {
                "version": graph["version"],
                "n": graph["n"],
                "n_edges": len(graph["col_idx"]),
                "n_dn": graph["n_dn"],
                "feature_to_cell": graph["feature_to_cell"],
                "visual_to_cell": graph["visual_to_cell"],
                "dn_index": graph["dn_index"],
                "channels": graph["channels"],
                "nodes": graph["nodes"],
                "selection": graph["selection"],
            },
            indent=2,
        )
        + "\n"
    )

    bin_path = OUT / "subgraph_v1.bin"
    shuffle_path = OUT / "subgraph_v1_shuffle.bin"
    write_bin(bin_path, graph, shuffle=False)
    write_bin(shuffle_path, graph, shuffle=True, seed=20260914)

    manifest = {
        "dataset": "FlyEM MaleCNS v1.0, min confidence 0.5",
        "license": "CC BY 4.0",
        "source": "https://male-cns.janelia.org/download/",
        "nodes": graph["n"],
        "edges": len(graph["col_idx"]),
        "n_dn": graph["n_dn"],
        "channels": VISUAL_TYPES,
        "sources_sha256": sources,
        "files": {
            "json": "subgraph_v1.json",
            "bin": "subgraph_v1.bin",
            "shuffle_bin": "subgraph_v1_shuffle.bin",
            "bin_sha256": sha256_file(bin_path),
            "shuffle_sha256": sha256_file(shuffle_path),
        },
        "assumptions": (
            "Engineered 6-channel feature injection onto strongest cell per LC type; "
            "ACh +1, GABA/Glu -1; unclear 0; signed weights normalized per post cell; "
            "leaky-tanh dynamics in C. Not a physiological whole-brain model."
        ),
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
