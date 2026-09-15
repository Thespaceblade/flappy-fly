#ifndef FLAPPY_H
#define FLAPPY_H

#include <stdbool.h>
#include <stdint.h>

/*
 * Physics / layout from the original Flappy Bird Android build, via
 * THEN00P/reFlappy (decompiled com.dotgears.flappy):
 *   Bird.java, GameScene.java, atlas.txt
 * Visual assets for the browser UI come from nebez/floppybird (extracted
 * original sprites). See THIRD_PARTY.md.
 */

enum {
  FF_WIDTH = 288,
  FF_HEIGHT = 512,
  /* Collision box from Bird.loadFrames(..., 20, 20, ...) — not the 48px sprite. */
  FF_BIRD_W = 20,
  FF_BIRD_H = 20,
  FF_BIRD_START_X = 80,
  FF_BIRD_START_Y = 246,
  FF_PIPE_W = 52,
  FF_PIPE_H = 320, /* atlas: pipe_up / pipe_down */
  FF_PIPE_GAP = 96, /* pixels between bottom of top pipe and top of bottom pipe */
  FF_PIPE_H_MIN = 180, /* randomRange(180, 360) → [180, 360) */
  FF_PIPE_H_MAX = 360,
  FF_GROUND_Y = 400, /* death when bird.y >= GROUND_Y - bird.height */
  FF_SCROLL = 2,     /* gameSpeed */
  FF_MAX_PIPES = 3,
  FF_PHYS_HZ = 60,
  FF_DECIDE_HZ = 30,
  FF_EPISODE_CAP_S = 60,
  FF_N_FEATURES = 6
};

typedef struct {
  int x;
  int height; /* y of bottom (pipe_up) top edge; gap sits above it by FF_PIPE_GAP */
  bool scored;
} FfPipe;

typedef struct {
  uint32_t seed;
  uint32_t rng;

  float bird_y;
  float bird_vy;
  int bird_x;

  FfPipe pipes[FF_MAX_PIPES];
  int n_pipes;
  int pipe_spacing; /* (288 - (pipe_w * 3 / 2)) / 2 */

  int frame;
  int pipes_cleared;
  bool alive;
  bool flap_pending;
} FfGame;

void ff_seed_rng(FfGame *g, uint32_t seed);
uint32_t ff_rand_u32(FfGame *g);
int ff_rand_range(FfGame *g, int min_inclusive, int max_exclusive);

void ff_reset(FfGame *g, uint32_t seed);
void ff_queue_flap(FfGame *g);
void ff_step(FfGame *g);

void ff_features(const FfGame *g, float out[FF_N_FEATURES]);

int ff_physics_frames_per_decision(void);
int ff_max_physics_frames(void);

#endif
