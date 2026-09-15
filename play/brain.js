/** MaleCNS subgraph dynamics — mirrors c/brain.c (leaky-tanh + MLP readout). */
export class FlyBrain {
  constructor() {
    this.n = 0;
    this.nDn = 0;
    this.nEdges = 0;
    this.rowPtr = null;
    this.colIdx = null;
    this.weight = null;
    this.featureToCell = new Int32Array(6);
    this.visualToCell = [];
    this.dnIndex = new Int32Array(32);
    this.h = null;
    this.params = null;
    this.nParams = 0;
    this.hidden = 16;
    this.nodes = [];
    this.channels = [];
    this.optical = false;
  }

  static async load({ graphUrl, metaUrl, paramsUrl }) {
    const b = new FlyBrain();
    const [graphBuf, meta, paramsBuf] = await Promise.all([
      fetch(graphUrl).then((r) => {
        if (!r.ok) throw new Error(`graph ${r.status}`);
        return r.arrayBuffer();
      }),
      fetch(metaUrl).then((r) => {
        if (!r.ok) throw new Error(`meta ${r.status}`);
        return r.json();
      }),
      fetch(paramsUrl).then((r) => {
        if (!r.ok) throw new Error(`params ${r.status}`);
        return r.arrayBuffer();
      }),
    ]);
    b._loadGraph(graphBuf);
    b.nodes = meta.nodes || [];
    b.channels = meta.channels || [];
    b.visualToCell =
      meta.visual_to_cell ||
      meta.nodes
        .map((n, i) => (n.role === "input" ? i : -1))
        .filter((i) => i >= 0);
    b.setParams(new Float32Array(paramsBuf));
    b.reset();
    return b;
  }

  _loadGraph(buf) {
    const dv = new DataView(buf);
    let o = 0;
    const magic = String.fromCharCode(
      dv.getUint8(0),
      dv.getUint8(1),
      dv.getUint8(2),
      dv.getUint8(3)
    );
    if (magic !== "FFSG") throw new Error("bad FFSG magic");
    o = 4;
    const version = dv.getUint32(o, true); o += 4;
    if (version !== 1) throw new Error(`unsupported FFSG version ${version}`);
    const n = dv.getUint32(o, true); o += 4;
    const nIn = dv.getUint32(o, true); o += 4;
    const nDn = dv.getUint32(o, true); o += 4;
    const nEdges = dv.getUint32(o, true); o += 4;
    if (nIn !== 6) throw new Error("expected 6 inputs");
    this.n = n;
    this.nDn = nDn;
    this.nEdges = nEdges;
    for (let i = 0; i < 6; i++) {
      this.featureToCell[i] = dv.getInt32(o, true);
      o += 4;
    }
    for (let i = 0; i < 32; i++) {
      this.dnIndex[i] = dv.getInt32(o, true);
      o += 4;
    }
    this.rowPtr = new Int32Array(n + 1);
    for (let i = 0; i <= n; i++) {
      this.rowPtr[i] = dv.getInt32(o, true);
      o += 4;
    }
    this.colIdx = new Int32Array(nEdges);
    for (let i = 0; i < nEdges; i++) {
      this.colIdx[i] = dv.getInt32(o, true);
      o += 4;
    }
    this.weight = new Float32Array(nEdges);
    for (let i = 0; i < nEdges; i++) {
      this.weight[i] = dv.getFloat32(o, true);
      o += 4;
    }
    this.h = new Float32Array(n);
    this.nParams = this.hidden * nDn + this.hidden + 2 * this.hidden + 2;
    this.params = new Float32Array(this.nParams);
  }

  setParams(arr) {
    const n = Math.min(arr.length, this.nParams);
    this.params.set(arr.subarray(0, n));
  }

  reset() {
    this.h.fill(0);
  }

  _tanh(x) {
    if (x > 20) return 1;
    if (x < -20) return -1;
    return Math.tanh(x);
  }

  _step(u) {
    const hNew = new Float32Array(this.n);
    for (let i = 0; i < this.n; i++) {
      let acc = u[i];
      const a = this.rowPtr[i];
      const b = this.rowPtr[i + 1];
      for (let k = a; k < b; k++) {
        acc += 1.4 * this.weight[k] * this.h[this.colIdx[k]];
      }
      hNew[i] = 0.3 * this.h[i] + 0.7 * this._tanh(acc);
    }
    this.h.set(hNew);
  }

  _readoutFromState() {
    const p = this.params;
    const nd = this.nDn;
    const H = this.hidden;
    const W1 = 0;
    const b1 = W1 + H * nd;
    const W2 = b1 + H;
    const b2 = W2 + 2 * H;
    const dnActs = new Float32Array(nd);
    for (let i = 0; i < nd; i++) {
      const cell = this.dnIndex[i];
      dnActs[i] = cell >= 0 && cell < this.n ? 4 * this.h[cell] : 0;
    }
    const hidden = new Float32Array(H);
    for (let h = 0; h < H; h++) {
      let a = p[b1 + h];
      for (let i = 0; i < nd; i++) a += p[W1 + h * nd + i] * dnActs[i];
      hidden[h] = this._tanh(a);
    }
    const logits = [0, 0];
    for (let o = 0; o < 2; o++) {
      let a = p[b2 + o];
      for (let h = 0; h < H; h++) a += p[W2 + o * H + h] * hidden[h];
      logits[o] = a;
    }
    const flap = logits[0] > logits[1] ? 1 : 0;
    const m = Math.max(logits[0], logits[1]);
    const e0 = Math.exp(logits[0] - m);
    const e1 = Math.exp(logits[1] - m);
    const pFlap = e0 / (e0 + e1);
    return { flap, logits, pFlap, hidden, dnActs };
  }

  decide(features) {
    const u = new Float32Array(this.n);
    for (let f = 0; f < 6; f++) {
      const cell = this.featureToCell[f];
      if (cell >= 0 && cell < this.n) u[cell] = 2 * (features[f] - 0.5);
    }
    for (let s = 0; s < 3; s++) this._step(u);
    return {
      ...this._readoutFromState(),
      features: Float32Array.from(features),
      drives: null,
    };
  }

  /** Optical path: drives[i] ∈ (-1,1) → visualToCell[i]. */
  decideFromDrives(drives) {
    const u = new Float32Array(this.n);
    const n = Math.min(drives.length, this.visualToCell.length);
    for (let i = 0; i < n; i++) {
      const cell = this.visualToCell[i];
      if (cell >= 0 && cell < this.n) u[cell] = 2 * drives[i];
    }
    for (let s = 0; s < 3; s++) this._step(u);
    return {
      ...this._readoutFromState(),
      features: null,
      drives: Float32Array.from(drives),
    };
  }

  /** Flat readout parameter snapshot for live HUD. */
  paramSnapshot() {
    return Array.from(this.params);
  }
}

/** Draw activity map onto a canvas. */
export function drawBrainMap(canvas, brain, hoverIdx) {
  const ctx = canvas.getContext("2d");
  const n = brain.n;
  const cols = 8;
  const rows = Math.ceil(n / cols);
  const pad = 4;
  const cell = Math.floor(
    Math.min(
      (canvas.width - pad * 2) / cols,
      (canvas.height - pad * 2 - 28) / rows
    )
  );
  ctx.fillStyle = "#0b1020";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#9ab";
  ctx.font = "11px ui-monospace, Menlo, monospace";
  ctx.fillText("MaleCNS subgraph activity", pad, 14);

  const dnSet = new Set();
  for (let i = 0; i < brain.nDn; i++) dnSet.add(brain.dnIndex[i]);
  const inSet = new Set(
    brain.visualToCell && brain.visualToCell.length
      ? brain.visualToCell
      : [...brain.featureToCell]
  );

  for (let i = 0; i < n; i++) {
    const c = i % cols;
    const r = Math.floor(i / cols);
    const x = pad + c * cell;
    const y = 22 + pad + r * cell;
    const v = brain.h[i];
    const t = Math.max(-1, Math.min(1, v));
    // blue ← 0 → amber
    if (t >= 0) {
      const g = Math.floor(40 + t * 180);
      const rC = Math.floor(40 + t * 215);
      ctx.fillStyle = `rgb(${rC},${g},40)`;
    } else {
      const b = Math.floor(40 + -t * 200);
      ctx.fillStyle = `rgb(30,60,${b})`;
    }
    ctx.fillRect(x + 1, y + 1, cell - 2, cell - 2);

    let stroke = "#334";
    if (inSet.has(i)) stroke = "#4ade80";
    if (dnSet.has(i)) stroke = "#f472b6";
    ctx.strokeStyle = i === hoverIdx ? "#fff" : stroke;
    ctx.lineWidth = i === hoverIdx ? 2 : 1;
    ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
  }

  // legend
  const ly = canvas.height - 14;
  ctx.fillStyle = "#4ade80";
  ctx.fillRect(pad, ly - 8, 8, 8);
  ctx.fillStyle = "#9ab";
  ctx.fillText("input", pad + 12, ly);
  ctx.fillStyle = "#f472b6";
  ctx.fillRect(pad + 60, ly - 8, 8, 8);
  ctx.fillStyle = "#9ab";
  ctx.fillText("DN out", pad + 72, ly);
}
