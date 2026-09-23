"""Tests for the sim, the oracle, and the state encoder.

These mirror the Rust prototype's tests one-for-one. If the port drifted, the
oracle's labels would silently become wrong and every downstream number —
training data, accuracy, calibration — would be junk without any visible error.
"""

import math

import pytest

from laya_dino import state
from laya_dino.game import JUMP, MAX_SPEED, RUN, Game
from laya_dino.oracle import best_action, jump_window, must_jump


# --- sim ---------------------------------------------------------------

def test_replays_are_deterministic():
    def run(seed):
        g = Game(seed)
        from laya_dino.game import Rng
        r = Rng(7)
        trace = []
        while g.tick(JUMP if r.next_f32() < 0.02 else RUN):
            trace.append(round(g.distance_ran, 3))
            if g.frame > 5_000:
                break
        return g.frame, g.score(), trace

    assert run(42) == run(42)


def test_different_seeds_diverge():
    # Time-to-first-crash is *not* a valid signal: the opening obstacle always
    # spawns at the right edge, so every seed dies on the same frame if you
    # never jump. Compare the generated layout instead.
    def layout(seed):
        g = Game(seed)
        kinds = []
        for _ in range(3_000):
            g.crashed = False
            g.tick(RUN)
            if g.obstacles:
                o = g.obstacles[-1]
                if not kinds or kinds[-1] != (o.kind, int(o.w)):
                    kinds.append((o.kind, int(o.w)))
        return kinds

    assert layout(1) != layout(2)


def test_standing_still_eventually_crashes():
    g = Game(3)
    frames = 0
    while g.tick(RUN):
        frames += 1
        assert frames < 10_000, "never hit an obstacle while standing still"
    assert g.crashed


def test_a_jump_returns_to_ground():
    g = Game(9)
    g.tick(JUMP)
    assert g.jumping
    frames = 0
    while g.jumping and g.tick(RUN):
        frames += 1
        assert frames < 200, "jump never landed"
    # ~33 frames of hang time at gravity 0.6 / v0 -10.
    assert 30 <= frames < 40, f"unexpected hang time: {frames}"


def test_speed_ramps_to_cap_and_holds():
    g = Game(5)
    for _ in range(20_000):
        g.crashed = False  # ignore collisions; we only care about the ramp
        g.tick(RUN)
    assert abs(g.speed - MAX_SPEED) < 1e-3, f"speed = {g.speed}"


def test_clone_is_independent():
    g = Game(11)
    for _ in range(50):
        g.tick(RUN)
    c = g.clone()
    for _ in range(50):
        c.tick(JUMP)
    assert c.frame != g.frame
    assert c.obstacles is not g.obstacles
    assert all(a is not b for a, b in zip(c.obstacles, g.obstacles))


# --- oracle ------------------------------------------------------------

def test_oracle_survives_to_max_speed():
    g = Game(1234)
    while g.frame < 40_000:
        if not g.tick(best_action(g)):
            break
    assert not g.crashed, f"oracle crashed at score {g.score()}"
    assert abs(g.speed - MAX_SPEED) < 1e-3, "never reached max speed"
    assert g.score() > 10_000, f"oracle only scored {g.score()}"


@pytest.mark.parametrize("seed", range(10))
def test_oracle_survives_many_seeds(seed):
    g = Game(seed)
    while g.frame < 10_000:
        if not g.tick(best_action(g)):
            break
    assert not g.crashed, f"seed {seed} crashed at score {g.score()}"


def test_oracle_does_not_jump_constantly():
    # A degenerate always-jump oracle would teach the model nothing.
    g = Game(77)
    jumps = frames = 0
    while g.frame < 6_000:
        a = best_action(g)
        jumps += a == JUMP
        frames += 1
        if not g.tick(a):
            break
    rate = jumps / frames
    assert rate < 0.10, f"jump rate {rate} — oracle is spamming jumps"
    assert jumps > 0, "oracle never jumped"


def test_must_jump_agrees_with_best_action():
    g = Game(31)
    while g.frame < 4_000:
        a = best_action(g)
        if not g.jumping:
            assert must_jump(g) == (a == JUMP)
        if not g.tick(a):
            break


def test_jump_window_is_wide_enough_for_the_model():
    """The load-bearing measurement: the model has ~30ms, the window is wider."""
    worst = math.inf
    for seed in range(8):
        g = Game(seed)
        while g.frame < 8_000:
            if not g.jumping:
                w = jump_window(g)
                if w > 0:
                    worst = min(worst, w)
            if not g.tick(best_action(g)):
                break
    ms = worst * 1000.0 / 60.0
    # Measured over 4,588 obstacles on the faithful sim: min 5 frames, p01 17,
    # median 27. The rare 5-frame window is a single-cactus pterodactyl; the
    # model runs at ~22ms and we poll every 2 frames (33ms), so even the worst
    # case has room. This guards against a change that collapses it further.
    assert worst >= 4, f"window collapsed to {worst} frames"
    assert ms > 21.6 * 2, f"only {ms:.0f}ms of slack for a 21.6ms model"


# --- state encoding ----------------------------------------------------

@pytest.mark.parametrize("enc", state.ENCODINGS)
def test_encodings_are_compact(enc):
    g = Game(1)
    s = state.encode(g.observe(), enc)
    assert len(s) < 120, f"{enc} encoding too long: {s!r}"
    assert "\n" not in s


def test_handles_empty_field():
    g = Game(2)
    g.obstacles.clear()
    s = state.encode(g.observe(), state.NUMERIC)
    assert "obstacle=none" in s
    assert "inf" not in s, f"infinity leaked into the prompt: {s}"


def test_airborne_is_visible_to_the_model():
    g = Game(3)
    g.tick(JUMP)
    s = state.encode(g.observe(), state.NUMERIC)
    assert "airborne=1" in s
