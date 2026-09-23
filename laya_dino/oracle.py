"""A perfect controller, by brute force.

This is why dino is a good testbed for a one-decision model: ground truth is
free and exact. No human annotators, no teacher LLM grading the student. We
fork the deterministic sim, play both branches out, and see which survives.
Every label is a fact, not an opinion — which is precisely what the published
one-decision benchmarks lack.
"""

from __future__ import annotations

from .game import DINO_X, JUMP, RUN, Game

# A jump arc is ~33 frames; the widest obstacle gap is comfortably under this.
HORIZON = 240


def _survives(game: Game, first: int) -> bool:
    """Fork the sim, commit to `first` this frame, then coast. True if the dino
    is still alive once the obstacle it was facing is behind it.

    The clearance test must use the obstacle's real width. An earlier version
    assumed a fixed 50px, which was true only while obstacles came one at a
    time; with Chromium's cactus clusters a single obstacle is up to 75px wide,
    so the oracle declared victory while the dino was still inside it and
    happily labelled fatal frames as safe.
    """
    g = game.clone()
    pending = [o for o in g.obstacles if o.x + o.w > DINO_X]
    if not pending:
        return True
    target = min(pending, key=lambda o: o.x)
    tracked_x, tracked_w = target.x, target.w

    speed = g.speed
    if not g.tick(first):
        return False
    tracked_x -= speed

    for _ in range(HORIZON):
        # Cleared it: the obstacle's trailing edge is behind the dino's front.
        if tracked_x + tracked_w < DINO_X:
            return True
        speed = g.speed
        if not g.tick(RUN):
            return False
        tracked_x -= speed
    return True


def best_action(game: Game) -> int:
    """The optimal action for this frame.

    Stay grounded as long as possible, then jump on the *first* frame from
    which a jump is verified to clear. Jumping earlier means landing on the
    obstacle. High-flying pterodactyls are handled for free: running is already
    safe, so we never jump into one.
    """
    if _survives(game, RUN):
        return RUN
    if _survives(game, JUMP):
        return JUMP
    return RUN  # already doomed; recorded rather than raised


def is_decidable(game: Game) -> bool:
    """True when at least one action survives. Undecidable frames are excluded
    from training data so the model is never punished for an unwinnable state."""
    return _survives(game, RUN) or _survives(game, JUMP)


def must_jump(game: Game) -> bool:
    """Ground-truth label for the `noul` question: coasting no longer survives,
    and a jump started right now does."""
    if game.jumping:
        return False
    return (not _survives(game, RUN)) and _survives(game, JUMP)


def jump_window(game: Game) -> int:
    """How many consecutive frames, starting now, still permit a successful jump.

    This is the number that made the project viable. The frame budget (16.7ms)
    is *not* the decision budget: the dino only has to decide once per obstacle,
    within the window where a jump still clears. Measured worst case is 10
    frames (167ms) at max speed, against a 29.6ms model — 5.6x of margin.

    Frames where running is already safe don't count; there's no decision to
    miss. Airborne frames don't count either, since a second jump is a no-op
    and would read as spuriously "safe".
    """
    if game.jumping:
        return 0
    count = 0
    g = game.clone()
    for _ in range(HORIZON):
        if g.jumping:
            break
        if (not _survives(g, RUN)) and _survives(g, JUMP):
            count += 1
        elif count > 0:
            break  # window has opened and closed
        if not g.tick(RUN):
            break
    return count
