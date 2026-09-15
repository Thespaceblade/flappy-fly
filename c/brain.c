#include "brain.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static float tanhf_local(float x) {
  if (x > 20.0f)
    return 1.0f;
  if (x < -20.0f)
    return -1.0f;
  return tanhf(x);
}

static void brain_alloc_params(FfBrain *b) {
  b->n_params = FF_BRAIN_HIDDEN * b->n_dn + FF_BRAIN_HIDDEN +
                2 * FF_BRAIN_HIDDEN + 2;
  if (b->n_params > (int)(sizeof(b->params) / sizeof(b->params[0])))
    b->n_params = (int)(sizeof(b->params) / sizeof(b->params[0]));
  for (int i = 0; i < b->n_params; i++)
    b->params[i] = 0.0f;
}

void brain_init_identity_stub(FfBrain *b) {
  memset(b, 0, sizeof(*b));
  b->n_dn = FF_N_FEATURES;
  b->n = FF_N_FEATURES + FF_N_FEATURES;
  for (int i = 0; i < FF_N_FEATURES; i++) {
    b->feature_to_cell[i] = i;
    b->dn_index[i] = FF_N_FEATURES + i;
  }
  int e = 0;
  for (int post = 0; post < b->n; post++) {
    b->row_ptr[post] = e;
    if (post >= FF_N_FEATURES) {
      b->col_idx[e] = post - FF_N_FEATURES;
      b->weight[e] = 1.0f;
      e++;
    }
  }
  b->row_ptr[b->n] = e;
  b->n_edges = e;
  brain_alloc_params(b);
}

int brain_load_bin(FfBrain *b, const char *path) {
  memset(b, 0, sizeof(*b));
  FILE *f = fopen(path, "rb");
  if (!f)
    return -1;
  char magic[4];
  if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "FFSG", 4) != 0) {
    fclose(f);
    return -2;
  }
  unsigned version = 0, n = 0, n_in = 0, n_dn = 0, n_edges = 0;
  if (fread(&version, 4, 1, f) != 1 || fread(&n, 4, 1, f) != 1 ||
      fread(&n_in, 4, 1, f) != 1 || fread(&n_dn, 4, 1, f) != 1 ||
      fread(&n_edges, 4, 1, f) != 1) {
    fclose(f);
    return -3;
  }
  if (n > FF_BRAIN_MAX_N || n_edges > FF_BRAIN_MAX_E || n_dn > FF_BRAIN_MAX_DN ||
      n_in != FF_N_FEATURES) {
    fclose(f);
    return -4;
  }
  b->n = (int)n;
  b->n_dn = (int)n_dn;
  b->n_edges = (int)n_edges;
  if (fread(b->feature_to_cell, 4, FF_N_FEATURES, f) != FF_N_FEATURES) {
    fclose(f);
    return -5;
  }
  int dn_pad[FF_BRAIN_MAX_DN];
  if (fread(dn_pad, 4, FF_BRAIN_MAX_DN, f) != FF_BRAIN_MAX_DN) {
    fclose(f);
    return -6;
  }
  memcpy(b->dn_index, dn_pad, sizeof(int) * b->n_dn);
  if (fread(b->row_ptr, 4, n + 1, f) != n + 1) {
    fclose(f);
    return -7;
  }
  if (fread(b->col_idx, 4, n_edges, f) != n_edges) {
    fclose(f);
    return -8;
  }
  if (fread(b->weight, 4, n_edges, f) != n_edges) {
    fclose(f);
    return -9;
  }
  fclose(f);
  brain_alloc_params(b);
  return 0;
}

void brain_reset(FfBrain *b) {
  for (int i = 0; i < b->n; i++)
    b->h[i] = 0.0f;
}

void brain_set_silenced(FfBrain *b, bool on) { b->silenced = on; }

int brain_n_params(const FfBrain *b) { return b->n_params; }

void brain_set_params(FfBrain *b, const float *params, int n) {
  if (n > b->n_params)
    n = b->n_params;
  memcpy(b->params, params, (size_t)n * sizeof(float));
}

static void recurrent_step(FfBrain *b, const float *u) {
  float h_new[FF_BRAIN_MAX_N];
  for (int i = 0; i < b->n; i++) {
    float acc = u[i];
    for (int k = b->row_ptr[i]; k < b->row_ptr[i + 1]; k++) {
      int j = b->col_idx[k];
      acc += 1.4f * b->weight[k] * b->h[j];
    }
    h_new[i] = 0.3f * b->h[i] + 0.7f * tanhf_local(acc);
  }
  memcpy(b->h, h_new, (size_t)b->n * sizeof(float));
}

static void readout_logits(const FfBrain *b, float logits[2]) {
  const float *p = b->params;
  int nd = b->n_dn;
  const float *W1 = p;
  const float *b1 = W1 + FF_BRAIN_HIDDEN * nd;
  const float *W2 = b1 + FF_BRAIN_HIDDEN;
  const float *b2 = W2 + 2 * FF_BRAIN_HIDDEN;

  float hidden[FF_BRAIN_HIDDEN];
  for (int h = 0; h < FF_BRAIN_HIDDEN; h++) {
    float a = b1[h];
    for (int i = 0; i < nd; i++) {
      float act = 0.0f;
      if (!b->silenced) {
        int cell = b->dn_index[i];
        if (cell >= 0 && cell < b->n)
          act = 4.0f * b->h[cell];
      }
      a += W1[h * nd + i] * act;
    }
    hidden[h] = tanhf_local(a);
  }
  for (int o = 0; o < 2; o++) {
    float a = b2[o];
    for (int h = 0; h < FF_BRAIN_HIDDEN; h++)
      a += W2[o * FF_BRAIN_HIDDEN + h] * hidden[h];
    logits[o] = a;
  }
}

int brain_decide(FfBrain *b, const float features[FF_N_FEATURES]) {
  float u[FF_BRAIN_MAX_N];
  for (int i = 0; i < b->n; i++)
    u[i] = 0.0f;
  if (!b->silenced) {
    for (int f = 0; f < FF_N_FEATURES; f++) {
      int cell = b->feature_to_cell[f];
      if (cell >= 0 && cell < b->n)
        u[cell] = 2.0f * (features[f] - 0.5f);
    }
  }
  for (int s = 0; s < 3; s++)
    recurrent_step(b, u);

  float logits[2];
  readout_logits(b, logits);
  return logits[0] > logits[1] ? 1 : 0;
}
