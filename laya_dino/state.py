"""Turning a game frame into the text `state` laya reads.

laya is an encoder over text, so *how* we write the frame is a real design
variable, not a formatting detail. Two things are at stake:

1. Token count dominates inference latency — measured 29.6ms for a short state
   vs 148ms for a long one on the same model. Every token costs.
2. Encoders are historically weak at numeric thresholds, which is exactly what
   this task is. Bucketing the distance makes the problem trivial — but then
   the model is only reading back a label the sim already computed, which
   proves nothing. Numeric is the honest test.

So we ship both and measure the gap. That gap is the interesting result.
"""

from __future__ import annotations

import math

from .game import Observation

NUMERIC = "numeric"
BUCKETED = "bucketed"
ENCODINGS = (NUMERIC, BUCKETED)

# The single question the agent asks, ~30 times a second.
JUMP_QUESTION = "Should the dino jump right now to clear the next obstacle?"

QUESTIONS = {
    "jump": {"type": "noul", "instructions": JUMP_QUESTION},
}


def _bucket_distance(d: float) -> str:
    if math.isinf(d):
        return "none"
    if d < 0:
        return "passed"
    if d < 40:
        return "imminent"
    if d < 90:
        return "close"
    if d < 160:
        return "near"
    if d < 300:
        return "far"
    return "distant"


def _bucket_width(w: float) -> str:
    """Obstacle width classes. Chromium widths are 17/34/51 (small cactus
    clusters), 25/50/75 (large) and 46 (pterodactyl)."""
    if w <= 20:
        return "narrow"
    if w <= 40:
        return "medium"
    if w <= 55:
        return "wide"
    return "very_wide"


def _bucket_speed(s: float) -> str:
    if s < 7.5:
        return "slow"
    if s < 10.0:
        return "medium"
    if s < 12.0:
        return "fast"
    return "max"


def encode(obs: Observation, encoding: str = NUMERIC) -> str:
    """Render the observation as a compact key=value line.

    Deliberately not JSON: braces, quotes and colons all cost tokens and add
    nothing the model needs. This form is roughly half the tokens of the
    equivalent JSON object.
    """
    airborne = int(obs.airborne)

    if encoding == NUMERIC:
        if obs.obstacle is None:
            return f"obstacle=none speed={obs.speed:.1f} airborne={airborne}"
        return (
            f"obstacle={obs.obstacle} dist={obs.distance:.0f} "
            f"speed={obs.speed:.1f} w={obs.obstacle_w:.0f} h={obs.obstacle_h:.0f} "
            f"alt={obs.obstacle_ground_offset:.0f} airborne={airborne} "
            f"vy={obs.dino_y_velocity:.1f}"
        )

    if encoding == BUCKETED:
        if obs.obstacle is None:
            return (
                f"obstacle=none speed={_bucket_speed(obs.speed)} "
                f"airborne={airborne}"
            )
        height = "high" if obs.obstacle_ground_offset > 20.0 else "ground"
        # Width must survive bucketing. Chromium spawns cacti in clusters of
        # up to 3, and a 75px obstacle needs a very different jump from a 17px
        # one. Dropping it would make the bucketed arm lose on missing
        # information rather than on encoding style, which is not the
        # comparison we want to run.
        return (
            f"obstacle={obs.obstacle} dist={_bucket_distance(obs.distance)} "
            f"speed={_bucket_speed(obs.speed)} height={height} "
            f"width={_bucket_width(obs.obstacle_w)} airborne={airborne}"
        )

    raise ValueError(f"unknown encoding: {encoding!r}")
