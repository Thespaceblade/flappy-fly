#include "flappy.h"

#include <math.h>
#include <string.h>

/*
 * Constants cited from THEN00P/reFlappy:
 *   gravity after flap = 0.3f, flap vy = -5.0f, max fall = 8.0f (Bird.java)
 *   scroll = 2 (GameScene.gameSpeed)
 *   gap = 96, pipe height randomRange(180,360) (GameScene.java)
 */

static float clampf(float x, float lo, float hi) {
  if (x < lo)
    return lo;
  if (x > hi)
    return hi;
  return x;
}

void ff_seed_rng(FfGame *g, uint32_t seed) {
  g->seed = seed;
  g->rng = seed ? seed : 1u;
}

uint32_t ff_rand_u32(FfGame *g) {
  g->rng = g->rng * 1664525u + 1013904223u;
  return g->rng;
}

/* Match MathHelper.randomRange: (nextRandom() % (max - min)) + min */
int ff_rand_range(FfGame *g, int min_inclusive, int max_exclusive) {
  int span = max_exclusive - min_inclusive;
  if (span <= 0)
    return min_inclusive;
  return (int)(ff_rand_u32(g) % (uint32_t)span) + min_inclusive;
}

static int pipe_spacing(void) {
  /* GameScene: (288 - ((pipeUp.width * 3) / 2)) / 2 with width=52 → 105 */
  return (FF_WIDTH - ((FF_PIPE_W * 3) / 2)) / 2;
}

static void spawn_pipe(FfGame *g, int x) {
  if (g->n_pipes >= FF_MAX_PIPES)
    return;
  FfPipe *p = &g->pipes[g->n_pipes++];
  p->x = x;
  p->height = ff_rand_range(g, FF_PIPE_H_MIN, FF_PIPE_H_MAX);
  p->scored = false;
}

void ff_reset(FfGame *g, uint32_t seed) {
  memset(g, 0, sizeof(*g));
  ff_seed_rng(g, seed);
  g->bird_x = FF_BIRD_START_X;
  g->bird_y = (float)FF_BIRD_START_Y;
  g->bird_vy = 0.0f;
  g->alive = true;
  g->pipe_spacing = pipe_spacing();
  g->n_pipes = 0;

  /* Active gameplay layout: three pipes like GameScene.startGame, already
   * scrolling (skip ready/menu). Initial x from startGame formulas. */
  int spacing = g->pipe_spacing;
  int x0 = spacing - (FF_PIPE_W >> 1);
  spawn_pipe(g, x0 + FF_WIDTH); /* first pipe enters from the right */
  spawn_pipe(g, g->pipes[0].x + spacing + FF_PIPE_W);
  spawn_pipe(g, g->pipes[1].x + spacing + FF_PIPE_W);
}

void ff_queue_flap(FfGame *g) { g->flap_pending = true; }

static bool aabb(int ax, int ay, int aw, int ah, int bx, int by, int bw,
                 int bh) {
  return ax + aw >= bx && ax <= bx + bw && ay + ah >= by && ay <= by + bh;
}

static void collide(FfGame *g) {
  int bx = g->bird_x;
  int by = (int)g->bird_y;
  if (by >= FF_GROUND_Y - FF_BIRD_H) {
    g->alive = false;
    return;
  }
  if (by < 0) {
    /* Original clamps ceiling visually; we treat hard ceiling as death for
     * training clarity (bird.y < 0 is rare with flap vy -5). */
    by = 0;
    g->bird_y = 0.0f;
  }
  for (int i = 0; i < g->n_pipes; i++) {
    FfPipe *p = &g->pipes[i];
    int top_y = (p->height - FF_PIPE_H) - FF_PIPE_GAP;
    int bot_y = p->height;
    if (aabb(bx, by, FF_BIRD_W, FF_BIRD_H, p->x, top_y, FF_PIPE_W, FF_PIPE_H) ||
        aabb(bx, by, FF_BIRD_W, FF_BIRD_H, p->x, bot_y, FF_PIPE_W, FF_PIPE_H)) {
      g->alive = false;
      return;
    }
  }
}

void ff_step(FfGame *g) {
  if (!g->alive)
    return;

  /* Bird.java flap() / update() */
  if (g->flap_pending) {
    g->bird_vy = -5.0f;
    g->flap_pending = false;
  }
  g->bird_vy += 0.3f;
  if (g->bird_vy > 8.0f)
    g->bird_vy = 8.0f;
  g->bird_y += g->bird_vy;

  int farthest = 0;
  for (int i = 0; i < g->n_pipes; i++) {
    g->pipes[i].x -= FF_SCROLL;
    if (g->pipes[i].x + FF_PIPE_W > farthest)
      farthest = g->pipes[i].x + FF_PIPE_W;
    /* Score when pipe x crosses bird.x (GameScene: pipe1X == bird.x || -1) */
    if (!g->pipes[i].scored &&
        (g->pipes[i].x == g->bird_x || g->pipes[i].x == g->bird_x - 1)) {
      g->pipes[i].scored = true;
      g->pipes_cleared++;
    }
  }

  for (int i = 0; i < g->n_pipes;) {
    if (g->pipes[i].x < -FF_PIPE_W) {
      g->pipes[i] = g->pipes[g->n_pipes - 1];
      g->n_pipes--;
    } else {
      i++;
    }
  }
  while (g->n_pipes < FF_MAX_PIPES) {
    int x = farthest + g->pipe_spacing;
    /* When recycling, original uses: pipe3X = pipe2X + spacing + width */
    if (g->n_pipes > 0) {
      int maxx = g->pipes[0].x;
      for (int i = 1; i < g->n_pipes; i++)
        if (g->pipes[i].x > maxx)
          maxx = g->pipes[i].x;
      x = maxx + g->pipe_spacing + FF_PIPE_W;
    }
    spawn_pipe(g, x);
    farthest = x + FF_PIPE_W;
  }

  collide(g);
  g->frame++;
}

static const FfPipe *next_pipe(const FfGame *g) {
  const FfPipe *best = NULL;
  int best_x = 1 << 30;
  for (int i = 0; i < g->n_pipes; i++) {
    const FfPipe *p = &g->pipes[i];
    if (p->x + FF_PIPE_W >= g->bird_x && p->x < best_x) {
      best_x = p->x;
      best = p;
    }
  }
  return best;
}

void ff_features(const FfGame *g, float out[FF_N_FEATURES]) {
  float bird_mid = g->bird_y + FF_BIRD_H * 0.5f;
  out[0] = clampf(bird_mid / (float)FF_GROUND_Y, 0.0f, 1.0f);
  out[1] = clampf((g->bird_vy + 5.0f) / 13.0f, 0.0f, 1.0f);

  const FfPipe *p = next_pipe(g);
  if (!p) {
    out[2] = 0.5f;
    out[3] = 0.0f;
    out[4] = (float)FF_PIPE_GAP / (float)FF_GROUND_Y;
    out[5] = 0.5f;
    return;
  }
  float gap_center = (float)p->height - (float)FF_PIPE_GAP * 0.5f;
  out[2] = clampf(gap_center / (float)FF_GROUND_Y, 0.0f, 1.0f);
  float dist = (float)(p->x - (g->bird_x + FF_BIRD_W));
  out[3] = clampf(1.0f - dist / (float)FF_WIDTH, 0.0f, 1.0f);
  out[4] = clampf((float)FF_PIPE_GAP * 0.5f / (float)FF_GROUND_Y, 0.0f, 1.0f);
  float err = bird_mid - gap_center;
  out[5] = clampf(0.5f + err / (float)FF_GROUND_Y, 0.0f, 1.0f);
}

int ff_physics_frames_per_decision(void) { return FF_PHYS_HZ / FF_DECIDE_HZ; }

int ff_max_physics_frames(void) { return FF_EPISODE_CAP_S * FF_PHYS_HZ; }
