# Overnight run log

Appended after each stage. Readable without Claude; this file is the report.

| stage | status | detail |
|---|---|---|
| sim rewrite | done | Ported sim to match Chromium exactly: DINO_X 25->50, speed-dependent jump velocity (-10 - speed/10), per-frame Math.round, cactus clusters (size 1-3), real per-sprite collision boxes, Chromium gap formula. Old sim mistrained every threshold. |
| oracle fix | done | `_survives` assumed 50px obstacle width; clusters are up to 75px, so it labelled fatal frames as safe. Now tracks real width. |
| window remeasure | done | 4,588 obstacles: min 5 frames (83ms), p01 17, median 27. Model is 21.6ms, polled every 2 frames (33ms). Still fits. |
| old training | killed | Was 60% through a run on stale-sim data; that checkpoint could not transfer to the real game. |
| train numeric (faithful) | done | 60k balanced rows, 3 epochs, head-only. acc 0.976, recall 0.974, precision 0.987. ECE 0.0119 -> 0.0068 (T=1.795). Zero-shot baseline was acc 0.481. |
| browser bridge | built | Serves third_party/t-rex-runner locally; reads Runner.instance_ state (evaluate p50 0.4ms); drives via Space keypress. Validated end-to-end. |
| sim eval v1 | 4/6 | threshold sweep: 0.99 and 0.995 both give median 1987, 4/6 survived. 0.9 -> 0/6, 0.999 -> 0/6 (too strict, misses windows). |
| bug: early jumps | found | Model fired at dist=326px with p=0.997, with erratic probabilities just before. Cause was my own stratification: balancing on (obstacle,label) thinned long-range negatives, so far distances were out-of-distribution. |
| retrain v2 | running | Stratify on (obstacle, label, distance band). 100k rows, 3 epochs. |
| NOTE: power | - | Battery reached 3% before AC was connected; macOS low-power throttling cut training throughput from ~76 to ~7 rows/s for a period. Training survived; timings in this log are not clean benchmarks after that point. |
| retrain v2 | done | Distance-banded stratification, 100k rows. acc 0.982, recall 0.979, precision 0.991. ECE 0.0091 -> 0.0056 (T=1.874). |
| sim sweep v2 | 7/8 | thr 0.97 -> 3/8; 0.99 -> 5/8; **0.995 -> 7/8 (median 1987 = frame cap, worst 1254)**; 0.999 -> 3/8 (rarely exceeds it, misses windows). Worst-case score went 39 -> 1254 vs v1. |
| REALITY GAP found | fixed | Ran on real Chrome: scores 248/124/420 vs 1987 in sim. Cause was OUR physics, not the bridge: Chromium's `updateJump` calls `endJump()` once yPos < MAX_JUMP_HEIGHT, clamping velocity to DROP_VELOCITY and truncating the ascent. Sim modelled a free parabola (peak 99px, 36 frames) vs real (87px, 33 frames). Sim thought a 51px cluster was clearable from 201px; Chrome disagreed. After the fix the sim agrees: window opens at ~190, not ~220. |
| browser bridge works | confirmed | Real Chrome dino driven at 61fps, decision latency p50 22.5ms, evaluate round-trip 0.4ms. |
| retrain v3 (correct physics) | done | acc 0.964, recall 0.963, precision 0.967. ECE 0.0217 -> 0.0133 (T=1.568). Lower than v2's 0.982 because the corrected game is genuinely harder. |
| sim sweep v3 | 4/8 | 0.97 -> 1/8; 0.99 -> 3/8; **0.995 -> 4/8 (median 1674)**; 0.998 -> 0/8. |
| window remeasure (final) | done | 3,822 obstacles: min 12 frames (200ms), p01 16, median 22 (367ms). Capped jump widened the worst case vs the uncapped sim. |
| restart bug | fixed | 2/5 Chrome games reported 0 decisions and carried the previous game's score. `Runner.restart()` is guarded by `if (!this.raqId)` so it no-ops on a run that ended by time limit. Now reloads the page per game; invalid runs are excluded from the summary rather than silently averaged in. |
| **REAL CHROME DINO** | **done** | 5 runs, 180s cap, thr 0.995: **3209, 3208, 368, 2552, 1112**. Median 2552, best 3209. 4/5 reached max speed 13.0; 2/5 never crashed. Latency p50 29.7ms / p90 37.0ms over 9,190 decisions. |
| video | done | results/laya_plays_chrome_dino.mp4 (5.1MB, 190s, real game + real sprite). Raw webm recordings in results/video/. |
| README | rewritten | All numbers refreshed to final measured values; stale claims removed. |
| NOT DONE | - | Bucketed-encoding ablation (Phase C). Code supports `--encoding bucketed`; never trained. |
| head-to-head | done | laya vs tuned one-liner on 3 disjoint seed ranges. Seeds 50000-07 (baseline tuned here): laya 4/8, baseline 6/8. Seeds 60000-07 (held out): laya 8/8, baseline 3/8. Seeds 70000-07 (held out): laya 6/8, baseline 2/8. **Total laya 18/24, baseline 11/24.** The one-liner only wins where it was fitted. |
| CORRECTION: Chrome stats | done | Morning report said "median 2552, best 3209" from a 5-run sample. Ran 12 more + a 3-run 300s batch. Pooled over **20 valid runs**: median 1062, mean 1322, quartiles 431/1062/2190, best 3209, worst 240. Survived full limit 3/20; reached max speed 13/20. Machine load was ~1.0, so the spread is genuine level-to-level variance, not contention. README corrected. |
| video per run | done | Rewrote recording to use one browser context per game, so each run gets its own self-describing file (game07_score241.webm). Slicing one long recording by accumulated durations failed, inter-game gaps are not constant and offsets drift, so the extracted clip showed the wrong game entirely. |
| FINAL Chrome stats | done | **28 runs: median 976, mean 1206, IQR 428-2086, range 81-3209. Max speed 17/28, survived full limit 3/28.** Clips: results/chrome_best.mp4 (2178), results/chrome_typical.mp4 (896). |

## Where this stands (2026-09-23)

Published: [repo](https://github.com/that-daniel/laya-dino) and
[writeup](https://bookofdaniel.in/posts/2026-09-23-i-taught-a-one-decision-model-to-play-chrome-dino/).

Headline numbers, all reproducible from this repo:

- Real Chrome dino, 28 runs: median 976, best 3209, worst 81. Max speed 17/28,
  clean runs 3/28. Decision latency p50 23-30ms.
- Model: acc 0.964, recall 0.963, precision 0.991, ECE 0.0217 -> 0.0133 at
  T=1.568. Zero-shot baseline was 0.468, below chance.
- Oracle ceiling is ~19,000, so the model is at roughly 1/20th of perfect play.

### Agreed and NOT yet done

1. **Widen the head-to-head.** `results/head_to_head.json` used 8 seeds per
   range, while the Chrome figures used 28 runs. The published post claims
   laya 18/24 vs the tuned threshold rule 11/24 on that thin sample. Pure
   simulation, ~40 min, no retraining needed. This is the weakest published
   claim.

2. **Bucketed-encoding ablation.** `scripts/train.py --encoding bucketed` is
   supported and has never been run. It answers whether the model learned the
   numeric threshold or the prompt pre-solved it. ~70 min.

### Cheap gains left on the table

- Only 100k of 1.3M generated rows were used for training.
- Only the decision head (15M params, 4.7%) was trained; the encoder is frozen.
- `laya` (ModernBERT-large, 421M) was never tried; all results use
  `laya-multilingual` (mmBERT-base, 322M) because it is 1.5x faster.
