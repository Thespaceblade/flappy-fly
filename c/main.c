#include "brain.h"
#include "flappy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
  POLICY_IDLE = 0,
  POLICY_RANDOM,
  POLICY_RULE,
  POLICY_BRAIN
} Policy;

static void usage(const char *argv0) {
  fprintf(stderr,
          "usage:\n"
          "  %s rollout --seed N [--policy idle|random|rule|brain]\n"
          "            [--graph path.bin] [--params floats.bin] [--silence]\n"
          "            [--quiet]\n"
          "  %s nparams [--graph path.bin]\n"
          "  %s features --seed N\n"
          "  %s selftest\n",
          argv0, argv0, argv0, argv0);
}

static int load_params_file(FfBrain *b, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f)
    return -1;
  float buf[2048];
  size_t n = fread(buf, sizeof(float), 2048, f);
  fclose(f);
  if (n == 0)
    return -2;
  if ((int)n > b->n_params)
    n = (size_t)b->n_params;
  brain_set_params(b, buf, (int)n);
  return 0;
}

static const FfPipe *find_next_pipe(const FfGame *g) {
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

static int policy_rule(const FfGame *g) {
  const FfPipe *p = find_next_pipe(g);
  float bird_mid = g->bird_y + FF_BIRD_H * 0.5f;
  float gap_center =
      p ? ((float)p->height - (float)FF_PIPE_GAP * 0.5f) : (float)FF_GROUND_Y * 0.5f;
  float err = bird_mid - gap_center; /* + => too low */
  if (err > 4.0f)
    return 1;
  if (err > -8.0f && g->bird_vy > 1.0f)
    return 1;
  return 0;
}

static uint32_t policy_rng;

static int policy_random(void) {
  policy_rng = policy_rng * 1664525u + 1013904223u;
  return (policy_rng >> 31) & 1;
}

/* verbose: 2=human, 1=machine line, 0=silent */
static int run_rollout(uint32_t seed, Policy pol, bool silence, int verbose,
                       const char *graph_path, const char *params_path) {
  FfGame g;
  FfBrain brain;
  ff_reset(&g, seed);
  if (graph_path && graph_path[0]) {
    int err = brain_load_bin(&brain, graph_path);
    if (err != 0) {
      fprintf(stderr, "brain_load_bin(%s) failed: %d\n", graph_path, err);
      return -1;
    }
  } else {
    brain_init_identity_stub(&brain);
  }
  if (params_path && params_path[0]) {
    int err = load_params_file(&brain, params_path);
    if (err != 0) {
      fprintf(stderr, "load_params(%s) failed: %d\n", params_path, err);
      return -1;
    }
  }
  brain_reset(&brain);
  brain_set_silenced(&brain, silence);
  policy_rng = seed ^ 0xC0FFEEu;

  int decide_every = ff_physics_frames_per_decision();
  int max_frames = ff_max_physics_frames();
  int flaps = 0;

  while (g.alive && g.frame < max_frames) {
    if (g.frame % decide_every == 0) {
      int flap = 0;
      float feat[FF_N_FEATURES];
      ff_features(&g, feat);
      switch (pol) {
      case POLICY_IDLE:
        flap = 0;
        break;
      case POLICY_RANDOM:
        flap = policy_random();
        break;
      case POLICY_RULE:
        flap = policy_rule(&g);
        break;
      case POLICY_BRAIN:
        flap = brain_decide(&brain, feat);
        break;
      }
      if (flap) {
        ff_queue_flap(&g);
        flaps++;
      }
    }
    ff_step(&g);
  }

  double seconds = g.frame / (double)FF_PHYS_HZ;
  if (verbose >= 2) {
    printf("seed=%u policy=%d silence=%d alive=%d frames=%d seconds=%.3f "
           "pipes=%d flaps=%d n=%d\n",
           seed, (int)pol, silence ? 1 : 0, g.alive ? 1 : 0, g.frame, seconds,
           g.pipes_cleared, flaps, brain.n);
  } else if (verbose == 1) {
    printf("%u %d %.6f %d %d\n", seed, g.pipes_cleared, seconds, g.frame,
           flaps);
  }
  return g.pipes_cleared;
}

static int selftest(void) {
  FfGame a, b;
  ff_reset(&a, 42);
  ff_reset(&b, 42);
  for (int i = 0; i < 200; i++) {
    if (i % 17 == 0) {
      ff_queue_flap(&a);
      ff_queue_flap(&b);
    }
    ff_step(&a);
    ff_step(&b);
    if (a.bird_y != b.bird_y || a.pipes_cleared != b.pipes_cleared) {
      fprintf(stderr, "selftest: nondeterministic at frame %d\n", i);
      return 1;
    }
  }

  /* Idle determinism across calls */
  if (run_rollout(7, POLICY_IDLE, false, 0, NULL, NULL) !=
      run_rollout(7, POLICY_IDLE, false, 0, NULL, NULL)) {
    fprintf(stderr, "selftest: idle mismatch\n");
    return 1;
  }

  /* Rule should survive longer than idle; ideally clear some pipes. */
  int idle_frames = 0, rule_frames = 0;
  int rule_pipes = 0;
  for (uint32_t s = 1; s <= 30; s++) {
    FfGame gi, gr;
    ff_reset(&gi, s);
    ff_reset(&gr, s);
    int max_frames = ff_max_physics_frames();
    int every = ff_physics_frames_per_decision();
    while (gi.alive && gi.frame < max_frames)
      ff_step(&gi);
    while (gr.alive && gr.frame < max_frames) {
      if (gr.frame % every == 0 && policy_rule(&gr))
        ff_queue_flap(&gr);
      ff_step(&gr);
    }
    idle_frames += gi.frame;
    rule_frames += gr.frame;
    rule_pipes += gr.pipes_cleared;
  }
  if (rule_frames <= idle_frames) {
    fprintf(stderr, "selftest: rule frames (%d) did not beat idle (%d)\n",
            rule_frames, idle_frames);
    return 1;
  }
  if (rule_pipes < 1) {
    fprintf(stderr, "selftest: rule cleared 0 pipes over 30 seeds\n");
    return 1;
  }

  printf("selftest ok (idle_frames=%d rule_frames=%d rule_pipes=%d over 30 "
         "seeds)\n",
         idle_frames, rule_frames, rule_pipes);
  return 0;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    usage(argv[0]);
    return 2;
  }

  if (strcmp(argv[1], "selftest") == 0)
    return selftest();

  if (strcmp(argv[1], "nparams") == 0) {
    const char *graph = NULL;
    for (int i = 2; i < argc; i++) {
      if (strcmp(argv[i], "--graph") == 0 && i + 1 < argc)
        graph = argv[++i];
    }
    FfBrain brain;
    if (graph && graph[0]) {
      int err = brain_load_bin(&brain, graph);
      if (err != 0) {
        fprintf(stderr, "brain_load_bin(%s) failed: %d\n", graph, err);
        return 1;
      }
    } else {
      brain_init_identity_stub(&brain);
    }
    printf("%d\n", brain_n_params(&brain));
    return 0;
  }

  if (strcmp(argv[1], "features") == 0) {
    uint32_t seed = 1;
    for (int i = 2; i < argc; i++) {
      if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        seed = (uint32_t)strtoul(argv[++i], NULL, 10);
    }
    FfGame g;
    ff_reset(&g, seed);
    float f[FF_N_FEATURES];
    ff_features(&g, f);
    for (int i = 0; i < FF_N_FEATURES; i++)
      printf("%.6f%s", f[i], i + 1 == FF_N_FEATURES ? "\n" : " ");
    return 0;
  }

  if (strcmp(argv[1], "rollout") == 0) {
    uint32_t seed = 1;
    Policy pol = POLICY_RULE;
    bool silence = false;
    int verbose = 2;
    const char *graph = NULL;
    const char *params = NULL;
    for (int i = 2; i < argc; i++) {
      if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
        seed = (uint32_t)strtoul(argv[++i], NULL, 10);
      else if (strcmp(argv[i], "--policy") == 0 && i + 1 < argc) {
        const char *p = argv[++i];
        if (strcmp(p, "idle") == 0)
          pol = POLICY_IDLE;
        else if (strcmp(p, "random") == 0)
          pol = POLICY_RANDOM;
        else if (strcmp(p, "rule") == 0)
          pol = POLICY_RULE;
        else if (strcmp(p, "brain") == 0)
          pol = POLICY_BRAIN;
        else {
          fprintf(stderr, "unknown policy %s\n", p);
          return 2;
        }
      } else if (strcmp(argv[i], "--graph") == 0 && i + 1 < argc)
        graph = argv[++i];
      else if (strcmp(argv[i], "--params") == 0 && i + 1 < argc)
        params = argv[++i];
      else if (strcmp(argv[i], "--silence") == 0)
        silence = true;
      else if (strcmp(argv[i], "--quiet") == 0)
        verbose = 1;
    }
    int pipes = run_rollout(seed, pol, silence, verbose, graph, params);
    return pipes < 0 ? 1 : 0;
  }

  usage(argv[0]);
  return 2;
}
