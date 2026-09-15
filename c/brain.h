#ifndef BRAIN_H
#define BRAIN_H

#include "flappy.h"

#include <stdbool.h>

enum {
  FF_BRAIN_MAX_N = 256,
  FF_BRAIN_MAX_E = 16384,
  FF_BRAIN_HIDDEN = 16,
  FF_BRAIN_N_IN = FF_N_FEATURES,
  FF_BRAIN_MAX_DN = 32
};

typedef struct {
  int n;
  int n_dn;
  int n_edges;

  int row_ptr[FF_BRAIN_MAX_N + 1];
  int col_idx[FF_BRAIN_MAX_E];
  float weight[FF_BRAIN_MAX_E];

  int feature_to_cell[FF_BRAIN_N_IN];
  int dn_index[FF_BRAIN_MAX_DN];

  float h[FF_BRAIN_MAX_N];
  bool silenced;

  int n_params;
  float params[2048];
} FfBrain;

void brain_init_identity_stub(FfBrain *b);
/* Load FFSG binary from scripts/build_subgraph.py. Returns 0 on success. */
int brain_load_bin(FfBrain *b, const char *path);
void brain_reset(FfBrain *b);
void brain_set_silenced(FfBrain *b, bool on);

int brain_decide(FfBrain *b, const float features[FF_N_FEATURES]);

int brain_n_params(const FfBrain *b);
void brain_set_params(FfBrain *b, const float *params, int n);

#endif
