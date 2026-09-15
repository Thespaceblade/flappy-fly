/** Optical encoder v2 — matches scripts/optical_model.py (4×64×64 stack). */

export class OpticalEncoder {
  constructor(bundle, weights) {
    this.nVisual = bundle.n_visual;
    this.shapes = bundle.encoder_shapes;
    this.obsShape = bundle.obs || [4, 64, 64];
    this.stackN = bundle.stack || this.obsShape[0] || 4;
    this.H = this.obsShape[1] || 64;
    this.W = this.obsShape[2] || 64;
    this.w = this._unpack(weights, bundle.encoder_shapes);
    this.stack = [];
  }

  static async load(bundleUrl) {
    const bundle = await fetch(bundleUrl).then((r) => {
      if (!r.ok) throw new Error(`optical bundle ${r.status}`);
      return r.json();
    });
    const binUrl = bundleUrl.replace(/[^/]+$/, bundle.encoder_bin);
    const buf = await fetch(binUrl).then((r) => {
      if (!r.ok) throw new Error(`encoder bin ${r.status}`);
      return r.arrayBuffer();
    });
    return { bundle, encoder: new OpticalEncoder(bundle, new Float32Array(buf)) };
  }

  reset() {
    this.stack = [];
  }

  _unpack(flat, shapes) {
    const order = [
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
    ];
    const out = {};
    let o = 0;
    for (const k of order) {
      const shape = shapes[k];
      let n = 1;
      for (const d of shape) n *= d;
      out[k] = flat.subarray(o, o + n);
      o += n;
    }
    if (o !== flat.length) throw new Error(`encoder weight size mismatch ${o} vs ${flat.length}`);
    return out;
  }

  forwardFromGame(game) {
    const cur = renderFrame(game, this.H, this.W);
    this.stack = [cur, ...this.stack].slice(0, this.stackN);
    while (this.stack.length < this.stackN) this.stack.push(cur);
    const obs = new Float32Array(this.stackN * this.H * this.W);
    for (let c = 0; c < this.stackN; c++) obs.set(this.stack[c], c * this.H * this.W);
    this.prevFrame = cur; // for eye preview
    return this.forward(obs);
  }

  forward(obs) {
    const C0 = this.stackN;
    let t = this._conv(obs, C0, this.H, this.W, this.w["encoder.conv1.weight"], this.w["encoder.conv1.bias"], 32, 5, 2, 2);
    this._relu(t);
    t = this._conv(t.data, 32, 32, 32, this.w["encoder.conv2.weight"], this.w["encoder.conv2.bias"], 64, 5, 2, 2);
    this._relu(t);
    t = this._conv(t.data, 64, 16, 16, this.w["encoder.conv3.weight"], this.w["encoder.conv3.bias"], 64, 5, 2, 2);
    this._relu(t);
    const flat = t.data;
    const inN = 64 * 8 * 8;
    const hid = new Float32Array(128);
    const Wf = this.w["encoder.fc.weight"];
    const bf = this.w["encoder.fc.bias"];
    for (let o = 0; o < 128; o++) {
      let a = bf[o];
      const row = o * inN;
      for (let i = 0; i < inN; i++) a += Wf[row + i] * flat[i];
      hid[o] = a > 0 ? a : 0;
    }
    const Wo = this.w["encoder.out.weight"];
    const bo = this.w["encoder.out.bias"];
    const out = new Float32Array(this.nVisual);
    for (let o = 0; o < this.nVisual; o++) {
      let a = bo[o];
      const row = o * 128;
      for (let i = 0; i < 128; i++) a += Wo[row + i] * hid[i];
      out[o] = Math.tanh(a);
    }
    return out;
  }

  _relu(t) {
    for (let i = 0; i < t.data.length; i++) if (t.data[i] < 0) t.data[i] = 0;
  }

  _conv(input, cIn, hIn, wIn, weight, bias, cOut, k, stride, pad) {
    const hOut = Math.floor((hIn + 2 * pad - k) / stride) + 1;
    const wOut = Math.floor((wIn + 2 * pad - k) / stride) + 1;
    const out = new Float32Array(cOut * hOut * wOut);
    for (let oc = 0; oc < cOut; oc++) {
      for (let oy = 0; oy < hOut; oy++) {
        for (let ox = 0; ox < wOut; ox++) {
          let acc = bias[oc];
          const iy0 = oy * stride - pad;
          const ix0 = ox * stride - pad;
          for (let ic = 0; ic < cIn; ic++) {
            for (let ky = 0; ky < k; ky++) {
              const iy = iy0 + ky;
              if (iy < 0 || iy >= hIn) continue;
              for (let kx = 0; kx < k; kx++) {
                const ix = ix0 + kx;
                if (ix < 0 || ix >= wIn) continue;
                const wi = (((oc * cIn + ic) * k + ky) * k + kx);
                const ii = (ic * hIn + iy) * wIn + ix;
                acc += weight[wi] * input[ii];
              }
            }
          }
          out[(oc * hOut + oy) * wOut + ox] = acc;
        }
      }
    }
    return { data: out, c: cOut, h: hOut, w: wOut };
  }
}

/** Match scripts/ff_obs.py render_frame */
export function renderFrame(game, OBS_H = 64, OBS_W = 64) {
  const W = 288;
  const GROUND_Y = 400;
  const BIRD_W = 20;
  const BIRD_H = 20;
  const BIRD_X = 80;
  const PIPE_W = 52;
  const PIPE_H = 320;
  const PIPE_GAP = 96;
  const OPT_BIRD = 8;
  const img = new Float32Array(OBS_H * OBS_W);
  img.fill(0.12);
  const sx = OBS_W / W;
  const sy = OBS_H / GROUND_Y;
  const groundR = OBS_H - Math.max(2, (0.08 * OBS_H) | 0);
  for (let y = groundR; y < OBS_H; y++) for (let x = 0; x < OBS_W; x++) img[y * OBS_W + x] = 0.45;

  for (const p of game.pipes) {
    let x0 = (p.x * sx) | 0;
    let x1 = ((p.x + PIPE_W) * sx) | 0;
    x0 = Math.max(0, Math.min(OBS_W, x0));
    x1 = Math.max(0, Math.min(OBS_W, x1));
    if (x1 <= x0) continue;
    const topY = p.height - PIPE_H - PIPE_GAP;
    const botY = p.height;
    let y0 = (topY * sy) | 0;
    let y1 = ((topY + PIPE_H) * sy) | 0;
    y0 = Math.max(0, Math.min(OBS_H, y0));
    y1 = Math.max(0, Math.min(OBS_H, y1));
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) img[y * OBS_W + x] = 0.8;
    y0 = (botY * sy) | 0;
    y1 = ((botY + PIPE_H) * sy) | 0;
    y0 = Math.max(0, Math.min(OBS_H, y0));
    y1 = Math.max(0, Math.min(OBS_H, y1));
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) img[y * OBS_W + x] = 0.8;
    let gap0 = ((botY - PIPE_GAP) * sy) | 0;
    let gap1 = (botY * sy) | 0;
    gap0 = Math.max(0, Math.min(OBS_H, gap0));
    gap1 = Math.max(0, Math.min(OBS_H, gap1));
    for (let y = gap0; y < gap1; y++) for (let x = x0; x < x1; x++) {
      if (img[y * OBS_W + x] < 0.28) img[y * OBS_W + x] = 0.28;
    }
  }

  const cy = ((game.birdY + BIRD_H * 0.5) * sy) | 0;
  const cx = ((BIRD_X + BIRD_W * 0.5) * sx) | 0;
  const half = (OPT_BIRD / 2) | 0;
  const y0 = Math.max(0, cy - half);
  const y1 = Math.min(OBS_H, cy + half);
  const x0 = Math.max(0, cx - half);
  const x1 = Math.min(OBS_W, cx + half);
  for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) img[y * OBS_W + x] = 1.0;
  const streak = Math.max(-10, Math.min(10, (game.birdVy * 1.2) | 0));
  if (streak > 0) {
    const ys0 = Math.max(0, cy - half - streak);
    for (let y = ys0; y < y0; y++) for (let x = x0; x < x1; x++) {
      if (img[y * OBS_W + x] < 0.65) img[y * OBS_W + x] = 0.65;
    }
  } else if (streak < 0) {
    const ys1 = Math.min(OBS_H, cy + half - streak);
    for (let y = y1; y < ys1; y++) for (let x = x0; x < x1; x++) {
      if (img[y * OBS_W + x] < 0.65) img[y * OBS_W + x] = 0.65;
    }
  }
  return img;
}

/** @deprecated — use OpticalEncoder.forwardFromGame */
export function renderObs(game, prevFrame) {
  const cur = renderFrame(game);
  return { obs: cur, cur };
}
