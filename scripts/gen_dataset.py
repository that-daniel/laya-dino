"""Generate training data by watching a perfect player.

Walks the oracle's own trajectory and labels every grounded frame with the
ground-truth answer to the one question the model will be asked:
"should the dino jump right now?"

Two sampling decisions matter:

* **Class imbalance.** Jumps are ~2-4% of frames. Training on the raw stream
  teaches the model to always say "no", which scores 97% and dies instantly.
  We keep every positive and subsample negatives.
* **Hard negatives.** Most negatives are trivial ("obstacle 400px away"). The
  ones that teach the threshold are the frames *just* outside the jump window
  — one frame too early, or already too late. We oversample those deliberately,
  because they are the entire decision boundary.

* **Off-trajectory positives.** The oracle jumps on the *first* frame of the
  window, so following it blindly yields exactly one positive per obstacle.
  But the window is 10-22 frames wide and jumping is correct throughout. At
  inference the model will routinely be 2-5 frames deep into a window (it
  hesitated, or we polled late) — states an on-trajectory dataset never
  contains. So at each window we fork the sim, hold RUN, and record the whole
  window. Without this the model is trained on a distribution it will never
  actually see, and one hesitation becomes unrecoverable.

Seeds are split disjointly between train and eval, so the model is always
evaluated on levels it has never seen.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from laya_dino import state
from laya_dino.game import JUMP, Game
from laya_dino.oracle import best_action, must_jump

# A frame is a "hard negative" if a jump window opens within this many frames.
HARD_NEGATIVE_LOOKAHEAD = 28


def rollout(seed: int, max_frames: int, rng: random.Random,
            easy_negative_rate: float) -> list[dict]:
    """One pass over the oracle's trajectory.

    Performance note: labelling is dominated by the forward simulations inside
    `must_jump`, so we call it exactly once per grounded frame and derive
    everything else from the resulting sequence. An earlier version probed
    28 frames ahead *per frame* to find hard negatives, which was ~1000x more
    work for information already implied by the labels.
    """
    g = Game(seed)
    pending: list[dict] = []   # candidate rows, in frame order
    extra: list[dict] = []     # off-trajectory window interiors

    while g.frame < max_frames:
        if not g.jumping:
            obs = g.observe()
            label = must_jump(g)
            pending.append(_row(g, obs, label, "positive" if label else "negative",
                                0, seed))
            if label:
                # Walk the rest of the window off-trajectory: same obstacle,
                # but having hesitated 1..n frames. All still correctly "jump".
                interior = _window_interior(g, seed)
                extra.extend(interior)
                pending[-1]["window"] = len(interior) + 1

        if not g.tick(best_action(g)):
            break

    # A negative is "hard" if a jump window opens within the next N grounded
    # frames. That's just proximity to the next positive in this list — no
    # extra simulation needed.
    next_pos = math.inf
    for r in reversed(pending):
        if r["label"]:
            next_pos = 0
        else:
            next_pos += 1
            r["kind"] = "hard_negative" if next_pos <= HARD_NEGATIVE_LOOKAHEAD else "easy_negative"

    rows = [r for r in pending
            if r["label"] or r["kind"] == "hard_negative"
            or rng.random() < easy_negative_rate]
    rows.extend(extra)
    return rows


def _row(g: Game, obs, label: bool, kind: str, window: int, seed: int) -> dict:
    return {
        "seed": seed,
        "frame": g.frame,
        "label": int(label),
        "kind": kind,
        "window": window,
        "speed": round(obs.speed, 3),
        "distance": None if obs.distance == float("inf") else round(obs.distance, 1),
        "numeric": state.encode(obs, state.NUMERIC),
        "bucketed": state.encode(obs, state.BUCKETED),
    }


def _window_interior(game: Game, seed: int) -> list[dict]:
    """Every remaining frame of an open jump window, reached by holding RUN.

    The final frame of the window is the most valuable row in the dataset: it
    is the last instant a jump still clears. One frame later the answer flips
    to "no" and the run is lost. That boundary is exactly what we need the
    model to learn.
    """
    rows = []
    g = game.clone()
    offset = 0
    while offset < 60:
        if not g.tick(0):  # RUN — deliberately hesitate
            break
        offset += 1
        if g.jumping or not must_jump(g):
            break
        rows.append(_row(g, g.observe(), True, f"window_interior+{offset}",
                         0, seed))
    return rows



def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-seeds", type=int, default=120)
    p.add_argument("--eval-seeds", type=int, default=30)
    p.add_argument("--max-frames", type=int, default=6000)
    p.add_argument("--easy-negative-rate", type=float, default=0.30)
    p.add_argument("--out", type=Path, default=Path("data"))
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(0)

    # Disjoint seed ranges: the eval levels are genuinely unseen.
    splits = {
        "train": range(0, args.train_seeds),
        "eval": range(10_000, 10_000 + args.eval_seeds),
    }

    for split, seeds in splits.items():
        rows: list[dict] = []
        for i, seed in enumerate(seeds):
            rows.extend(rollout(seed, args.max_frames, rng, args.easy_negative_rate))
            if (i + 1) % 20 == 0:
                print(f"  {split}: {i + 1} seeds, {len(rows)} rows", flush=True)

        rng.shuffle(rows)
        path = args.out / f"{split}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

        pos = sum(r["label"] for r in rows)
        hard = sum(r["kind"] == "hard_negative" for r in rows)
        print(
            f"{split}: {len(rows)} rows -> {path}  "
            f"positives={pos} ({pos / len(rows):.1%})  hard_negatives={hard}"
        )

    sample = json.loads((args.out / "train.jsonl").read_text().splitlines()[0])
    print("\nexample row:")
    print(f"  numeric : {sample['numeric']}")
    print(f"  bucketed: {sample['bucketed']}")
    print(f"  label   : {sample['label']}  ({sample['kind']})")


if __name__ == "__main__":
    main()
