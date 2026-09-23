"""Tests for the polling controller.

These run without loading a 322M-parameter model: `make_laya_policy` takes a
`jump_prob` callable, so an oracle-backed stub stands in for the network. That
keeps the control-flow bugs — which is where the real failures were — testable
in milliseconds.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from play import make_laya_policy, run_game, threshold_controller  # noqa: E402

from laya_dino.game import JUMP, RUN, Game  # noqa: E402
from laya_dino.oracle import best_action, must_jump  # noqa: E402


def perfect_prob(game: Game) -> float:
    """A stand-in for a flawless model."""
    return 1.0 if must_jump(game) else 0.0


def test_perfect_model_survives_through_the_controller():
    """If the controller is correct, a perfect oracle-probability model must
    survive. Any crash here is a control-flow bug, not a model bug."""
    for seed in (50_000, 50_006, 50_007):
        policy = make_laya_policy(perfect_prob, poll=2, threshold=0.99)
        r = run_game(seed, policy, max_frames=6_000)
        assert not r["crashed"], f"seed {seed} crashed at score {r['score']}"


def test_policy_state_does_not_leak_between_games():
    """The regression this file exists for.

    Playing a sequence of games must give the same result as playing each one
    alone. When the policy's latch lived in a mutable default argument, a
    stale JUMP carried from one game into frame 0 of the next and killed it on
    the first obstacle — while the model itself was fine.
    """
    seeds = [50_000, 50_006, 50_007, 50_001]

    alone = [run_game(s, make_laya_policy(perfect_prob, 2, 0.99), 4_000)["score"]
             for s in seeds]

    in_sequence = []
    for s in seeds:
        in_sequence.append(
            run_game(s, make_laya_policy(perfect_prob, 2, 0.99), 4_000)["score"])

    assert alone == in_sequence

    # And the failure mode itself: a policy reused across two games.
    shared = make_laya_policy(perfect_prob, 2, 0.99)
    first = run_game(seeds[0], shared, 4_000)
    second = run_game(seeds[1], shared, 4_000)  # same policy object, reused
    assert first["score"] == alone[0]
    # Reuse is what we're guarding against; it need not crash on every seed,
    # but it must not be how the tool is wired. This documents the hazard.
    assert second["score"] <= alone[1]


def test_latch_clears_so_one_decision_is_one_jump():
    """A held JUMP must not survive the ~33 airborne frames and re-fire on
    landing, which would turn a single decision into a permanent hop."""
    calls = {"n": 0}

    def always_jump(game):
        calls["n"] += 1
        return 1.0

    policy = make_laya_policy(always_jump, poll=1000, threshold=0.99)
    g = Game(50_000)
    jumps = 0
    for _ in range(200):
        a = policy(g)
        jumps += a == JUMP
        if not g.tick(a):
            break
    # poll=1000 means exactly one decision is ever made, so exactly one jump.
    assert calls["n"] == 1
    assert jumps == 1, f"one decision produced {jumps} jumps"


@pytest.mark.parametrize("poll", [1, 2, 4, 8])
def test_polling_rate_within_the_window_still_survives(poll):
    """The window is 10-22 frames. Polling every 8 frames should still be
    safe; this is the claim that made a 22ms model viable at 60fps."""
    policy = make_laya_policy(perfect_prob, poll=poll, threshold=0.99)
    r = run_game(50_000, policy, max_frames=6_000)
    assert not r["crashed"], f"poll={poll} crashed at score {r['score']}"


def test_threshold_baseline_is_a_real_but_limited_control():
    """The naive rule must actually run and get somewhere — but it does not
    beat the faithful game, and this test pins that finding down.

    Best (k, c) over a 2D sweep survives 3/8 levels. If a future change makes
    the baseline suddenly perfect, the sim has probably regressed to the
    easier approximation rather than the rule having improved.
    """
    scores, survived = [], 0
    for seed in range(50_000, 50_008):
        r = run_game(seed, threshold_controller, max_frames=8_000)
        scores.append(r["score"])
        survived += not r["crashed"]
    assert max(scores) > 800, f"baseline is broken, best score {max(scores)}"
    assert survived < 8, "baseline now beats the faithful sim — check the sim"


def test_oracle_and_controller_agree_on_action_counts():
    """A correct controller driven by oracle probabilities should jump about
    as often as the oracle itself — not wildly more."""
    policy = make_laya_policy(perfect_prob, poll=2, threshold=0.99)
    ctrl_trace, oracle_trace = [], []
    run_game(50_000, policy, 3_000, ctrl_trace)
    run_game(50_000, best_action, 3_000, oracle_trace)
    c, o = sum(ctrl_trace), sum(oracle_trace)
    assert o > 0
    assert c <= o * 1.5, f"controller jumped {c} times vs oracle {o}"
