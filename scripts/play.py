"""Let the model play, and measure whether it actually survives.

Accuracy on a held-out set is not the deliverable — surviving is. A model can
score 99% on frames and still die in ten seconds, because the 1% it gets wrong
are the frames that matter. This script reports scores from real runs on unseen
levels.

Three controllers are compared, deliberately:

* **oracle**    — brute-force perfect play. The ceiling.
* **threshold** — `if dist < k * speed: jump`, the three-line if-statement.
  Included because it is the honest control: if a 322M-parameter model cannot
  beat a one-liner, the reader deserves to know that.
* **laya**      — the fine-tuned decision head.

The model is polled every `--poll` frames rather than every frame. That is not
a shortcut: the measured jump window is 10-22 frames wide, so a decision every
2 frames cannot miss it, and it keeps the decision rate inside the model's
~30ms latency.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

import laya
from laya.common import QTYPES, build_sequence
from laya_dino import state
from laya_dino.game import DINO_H, JUMP, RUN, Game
from laya_dino.oracle import best_action

QUESTION = {"t": "noul", "ins": state.JUMP_QUESTION, "crit": None}


class LayaController:
    """Wraps the fine-tuned head as a policy. Batches nothing: this is the
    real-time path, one decision at a time, exactly as it would run in a demo."""

    def __init__(self, ckpt_path: Path, threshold: float = 0.5):
        blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        self.agent = laya.load(blob["checkpoint"])
        m = self.agent.model
        m.head.load_state_dict(blob["head"])
        m.type_emb.load_state_dict(blob["type_emb"])
        m.scorer.load_state_dict(blob["scorer"])
        m.act_head.load_state_dict(blob["act_head"])
        m.eval()
        self.temperature = blob.get("temperature", 1.0)
        self.encoding = blob["encoding"]
        self.max_len = blob.get("max_len", 160)
        self.threshold = threshold
        self.device = self.agent.device
        self.tok = self.agent.tok
        self.latencies: list[float] = []
        self.probs: list[float] = []

    @torch.no_grad()
    def jump_prob(self, game: Game) -> float:
        return self.jump_prob_from_obs(game.observe())

    @torch.no_grad()
    def jump_prob_from_obs(self, obs) -> float:
        """Decide from an Observation rather than a Game.

        The browser bridge has no `Game` object — it reconstructs an
        Observation from the live page. Routing both paths through this one
        method guarantees the real game and the sim get byte-identical prompts;
        any divergence here would be invisible and would look like the model
        failing to transfer.
        """
        text = state.encode(obs, self.encoding)
        ids, markers = build_sequence(self.tok, text, QUESTION, max_len=self.max_len)
        t0 = time.perf_counter()
        input_ids = torch.tensor([ids], device=self.device)
        attn = torch.ones_like(input_ids)
        mpos = torch.tensor([markers], device=self.device)
        mmask = torch.ones((1, 2), dtype=torch.bool, device=self.device)
        qt = torch.tensor([QTYPES["noul"]], device=self.device)
        logits, _ = self.agent.model(input_ids, attn, mpos, mmask, qt)
        p = torch.softmax(logits.float() / self.temperature, -1)[0, 1].item()
        self.latencies.append((time.perf_counter() - t0) * 1000.0)
        self.probs.append(p)
        return p


def threshold_controller(game: Game, k: float = 9.0, c: float = 0.5) -> int:
    """The one-liner every reader will think of. It is NOT enough.

    On the earlier approximate sim this rule beat the game for any k in
    [6.5, 14.75] — an enormous tolerance. Once the sim was made faithful to
    Chromium it stopped working: best configuration over a 2D sweep of (k, c)
    survives only 3 of 8 levels, dying inside size-3 cactus clusters.

    Why: the jump arc is fixed once launched, and a 75px cluster has to be
    fully traversed mid-air. Whether that happens is a function of the arc,
    the obstacle width and the speed together — not a linear threshold on
    distance. The oracle solves it by simulating the arc; a genuinely correct
    hand-written controller would have to do the same.

    So the honest framing for the writeup is not "a three-line if beats a 322M
    model". It is: the naive rule beats a simplified dino and fails the real
    one, and the gap between those two facts is the part worth writing about.

    The ground-offset check must compare against the dino's full height: a
    pterodactyl at +25px still overlaps a standing dino (0-47px) and must be
    jumped; only the +50px one can be run under.
    """
    obs = game.observe()
    if obs.obstacle is None or game.jumping:
        return RUN
    if obs.obstacle_ground_offset >= DINO_H:
        return RUN  # flies high enough to run under
    return JUMP if obs.distance < k * obs.speed + c * obs.obstacle_w else RUN


def run_game(seed: int, policy, max_frames: int,
             trace: list | None = None) -> dict:
    """Play one game. If `trace` is given, append each frame's action so the
    run can be replayed and rendered later — the sim is deterministic, so the
    action list plus the seed reproduces the run exactly."""
    g = Game(seed)
    while g.frame < max_frames:
        a = policy(g)
        if trace is not None:
            trace.append(a)
        if not g.tick(a):
            break
    return {"seed": seed, "score": g.score(), "frames": g.frame,
            "speed": round(g.speed, 2), "crashed": g.crashed,
            "reached_max_speed": g.at_max_speed()}


def make_laya_policy(jump_prob, poll: int, threshold: float,
                     on_frame=None):
    """Build a fresh policy for ONE game.

    The state here — poll phase and the held decision — must not outlive a
    single run. An earlier version kept it in a mutable default argument,
    which Python evaluates once at definition time, so all eight games shared
    one latch. A stale JUMP leaked from the end of one game into frame 0 of
    the next and killed it on the first obstacle. That looked exactly like a
    model failure and wasn't. Hence the factory, and hence
    `test_policy_state_does_not_leak_between_games`.

    `jump_prob(game) -> float` is injected so this is testable without loading
    a 322M-parameter model.
    """
    st = {"held": RUN, "n": 0}

    def policy(game: Game) -> int:
        # Poll every `poll` frames; hold the decision in between. The jump
        # window is 10-22 frames, so this cannot miss one.
        if game.jumping:
            if on_frame:
                on_frame(None)
            return RUN
        if st["n"] % poll == 0:
            pr = jump_prob(game)
            st["held"] = JUMP if pr >= threshold else RUN
            if on_frame:
                on_frame(pr)
        elif on_frame:
            on_frame(...)  # sentinel: "held, no new decision"
        st["n"] += 1
        action = st["held"]
        # Clear the latch once a jump is issued. Otherwise the held JUMP
        # survives the ~33 airborne frames and fires again the instant the
        # dino lands, turning one decision into a permanent hop.
        if action == JUMP:
            st["held"] = RUN
        return action

    return policy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/dino_head_numeric.pt"))
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=40_000)
    p.add_argument("--poll", type=int, default=2, help="decide every N frames")
    # Not 0.5. A false positive is fatal — jump too early and the dino lands
    # on the cactus — while a false negative costs nothing, because the window
    # stays open for another 10-22 frames and we poll every 2. The costs are
    # wildly asymmetric, so the operating point should be too.
    #
    # Measured: at 0.5 the model dies on the first obstacle every run despite
    # 96.8% frame accuracy. At 0.99 it survives. This only works because the
    # probabilities are calibrated (ECE 0.0068) — on an uncalibrated model
    # "0.99" would be a meaningless dial.
    p.add_argument("--threshold", type=float, default=0.99)
    p.add_argument("--controllers", default="oracle,threshold,laya")
    p.add_argument("--first-seed", type=int, default=50_000)
    p.add_argument("--out", type=Path, default=Path("results"))
    args = p.parse_args()

    # Seeds far from both train (0..200) and eval (10_000..) ranges.
    seeds = [args.first_seed + i for i in range(args.games)]
    wanted = args.controllers.split(",")
    results: dict[str, list[dict]] = {}

    if "oracle" in wanted:
        results["oracle"] = [run_game(s, best_action, args.max_frames) for s in seeds]

    if "threshold" in wanted:
        results["threshold"] = [run_game(s, threshold_controller, args.max_frames)
                                for s in seeds]

    ctrl = None
    if "laya" in wanted:
        ctrl = LayaController(args.checkpoint, args.threshold)
        print(f"loaded {args.checkpoint} (encoding={ctrl.encoding}, "
              f"T={ctrl.temperature:.3f}, device={ctrl.device})", flush=True)

        # Per-frame record for the renderer: what the model said, and how long
        # it took. `None` on frames where we held the previous decision.
        frame_probs: list[float | None] = []
        frame_lats: list[float | None] = []

        def record(pr):
            """Per-frame bookkeeping for the renderer."""
            if pr is None:                      # airborne
                frame_probs.append(None); frame_lats.append(None)
            elif pr is ...:                     # held, no new decision
                frame_probs.append(frame_probs[-1] if frame_probs else None)
                frame_lats.append(None)
            else:
                frame_probs.append(pr)
                frame_lats.append(ctrl.latencies[-1] if ctrl.latencies else None)

        runs = []
        best_trace, best_run = None, None
        for i, s in enumerate(seeds):
            frame_probs.clear(); frame_lats.clear()
            trace: list[int] = []
            policy = make_laya_policy(ctrl.jump_prob, args.poll,
                                      args.threshold, on_frame=record)
            r = run_game(s, policy, args.max_frames, trace)
            runs.append(r)
            if best_run is None or r["score"] > best_run["score"]:
                best_run, best_trace = r, {
                    "seed": s, "actions": list(trace),
                    "probs": list(frame_probs), "latencies": list(frame_lats),
                    "score": r["score"], "poll": args.poll,
                }
            print(f"  game {i+1}/{len(seeds)} seed={s} score={r['score']} "
                  f"crashed={r['crashed']}", flush=True)
        results["laya"] = runs

        if best_trace is not None:
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / "replay.json").write_text(json.dumps(best_trace))
            print(f"saved best replay (seed={best_trace['seed']}, "
                  f"score={best_trace['score']}) -> {args.out / 'replay.json'}")

    print(f"\n{'controller':<12} {'median':>8} {'best':>8} {'worst':>8} "
          f"{'survived':>10} {'max speed':>10}")
    print("-" * 62)
    summary = {}
    for name in wanted:
        rs = results.get(name)
        if not rs:
            continue
        scores = sorted(r["score"] for r in rs)
        survived = sum(not r["crashed"] for r in rs)
        maxed = sum(r["reached_max_speed"] for r in rs)
        summary[name] = {
            "median": statistics.median(scores), "best": scores[-1],
            "worst": scores[0], "survived": survived, "games": len(rs),
            "reached_max_speed": maxed,
        }
        print(f"{name:<12} {statistics.median(scores):>8.0f} {scores[-1]:>8} "
              f"{scores[0]:>8} {survived:>6}/{len(rs):<3} {maxed:>7}/{len(rs):<3}")

    if ctrl and ctrl.latencies:
        lat = sorted(ctrl.latencies)
        print(f"\ndecisions: {len(lat)}  p50={lat[len(lat)//2]:.1f}ms  "
              f"p90={lat[int(len(lat)*0.9)]:.1f}ms  max={lat[-1]:.1f}ms")
        print(f"jump window worst case measured earlier: 167ms")
        summary["latency_ms"] = {"p50": lat[len(lat) // 2],
                                 "p90": lat[int(len(lat) * 0.9)], "max": lat[-1]}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "play.json").write_text(json.dumps(
        {"summary": summary, "runs": results, "poll": args.poll}, indent=2))
    print(f"\nwrote {args.out / 'play.json'}")


if __name__ == "__main__":
    main()
