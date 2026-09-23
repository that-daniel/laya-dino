"""Faithful Chrome-dino simulation.

Ported directly from the Chromium t-rex runner (`third_party/t-rex-runner/
index.js`) so that a policy trained here transfers to the real game in a
browser. The first version of this file approximated the game; that was fine
for measuring the decision window but would have mistrained every threshold.
The differences that mattered:

* **Dino x position** is 50, not 25. A 25px shift is 2-4 frames of decision
  time at speed — larger than the margin we're optimising.
* **Jump velocity depends on speed**: `-10 - speed/10`, so a jump at max speed
  is meaningfully higher and longer than one at the start.
* **Per-frame rounding.** The real game does `yPos += Math.round(velocity)`,
  which is not the same trajectory as unrounded float accumulation.
* **Cacti spawn in clusters** of up to 3 (`MAX_OBSTACLE_LENGTH`), gated on
  speed. A 75px-wide obstacle needs an earlier jump than a 25px one, and a
  model that has never seen one will misjudge it.
* **Real collision boxes.** Chromium uses 6 boxes for the dino and 3 per
  cactus rather than one rectangle, which moves the last-safe-jump frame.
* **Gap formula** is `round(width*speed + minGap*0.6)` to 1.5x that, per type.

Two properties are load-bearing and preserved:

* **Determinism.** Same seed + same actions produce an identical replay, which
  is what makes the oracle's labels exact and evaluation reproducible.
* **No wall-clock coupling.** The game advances only when `tick()` is called,
  so a 22ms model decision cannot corrupt the physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

# --- Runner.config ---------------------------------------------------------
CANVAS_W = 600.0
CANVAS_H = 150.0
ACCELERATION = 0.001
GAP_COEFFICIENT = 0.6
GRAVITY = 0.6
START_SPEED = 6.0
MAX_SPEED = 13.0
MAX_OBSTACLE_LENGTH = 3
MAX_OBSTACLE_DUPLICATION = 2
BOTTOM_PAD = 10.0
MAX_GAP_COEFFICIENT = 1.5

# --- Trex.config -----------------------------------------------------------
DINO_X = 50.0          # START_X_POS
DINO_W = 44.0
DINO_H = 47.0
INITIAL_JUMP_VELOCITY = -10.0
GROUND_Y = CANVAS_H - DINO_H - BOTTOM_PAD   # 93.0

# The jump is *capped*, which matters more than anything else in this file.
# Chromium's `updateJump` calls `endJump()` once `yPos < MAX_JUMP_HEIGHT`, and
# `endJump` clamps the velocity to DROP_VELOCITY, cutting the ascent short.
# Without this the dino appears to follow a full parabola and reach ~94px,
# when the real game tops out far lower. Modelling the uncapped arc made the
# sim believe a 51px cactus cluster was clearable from 201px away; in the real
# browser that same jump hits the cactus. Every trained threshold inherits
# this, so it has to be exact.
MAX_JUMP_HEIGHT = 30.0      # absolute yPos, not a height above ground
MIN_JUMP_HEIGHT = 30.0      # offset below groundY at which the jump may be cut
DROP_VELOCITY = -5.0
MIN_JUMP_Y = GROUND_Y - MIN_JUMP_HEIGHT     # 63.0

SCORE_COEFFICIENT = 0.025

# Trex.collisionBoxes.RUNNING — (x, y, w, h) relative to the dino's top-left.
TREX_BOXES = (
    (22.0, 0.0, 17.0, 16.0),
    (1.0, 18.0, 30.0, 9.0),
    (10.0, 35.0, 14.0, 8.0),
    (1.0, 24.0, 29.0, 5.0),
    (5.0, 30.0, 21.0, 4.0),
    (9.0, 34.0, 15.0, 4.0),
)

CACTUS_SMALL = "cactus_small"
CACTUS_LARGE = "cactus_large"
PTERODACTYL = "pterodactyl"

# Obstacle.types, keyed by our names.
TYPES = {
    CACTUS_SMALL: {
        "width": 17.0, "height": 35.0, "y_pos": (105.0,),
        "multiple_speed": 4.0, "min_gap": 120.0, "min_speed": 0.0,
        "boxes": ((0.0, 7.0, 5.0, 27.0), (4.0, 0.0, 6.0, 34.0), (10.0, 4.0, 7.0, 14.0)),
    },
    CACTUS_LARGE: {
        "width": 25.0, "height": 50.0, "y_pos": (90.0,),
        "multiple_speed": 7.0, "min_gap": 120.0, "min_speed": 0.0,
        "boxes": ((0.0, 12.0, 7.0, 38.0), (8.0, 0.0, 7.0, 49.0), (13.0, 10.0, 10.0, 38.0)),
    },
    PTERODACTYL: {
        "width": 46.0, "height": 40.0, "y_pos": (100.0, 75.0, 50.0),
        "multiple_speed": 999.0, "min_gap": 150.0, "min_speed": 8.5,
        "boxes": ((15.0, 15.0, 16.0, 5.0), (18.0, 21.0, 24.0, 6.0),
                  (2.0, 14.0, 4.0, 3.0), (6.0, 10.0, 4.0, 7.0), (10.0, 8.0, 6.0, 9.0)),
    },
}
TYPE_ORDER = (CACTUS_SMALL, CACTUS_LARGE, PTERODACTYL)

RUN = 0
JUMP = 1

_MASK64 = (1 << 64) - 1


class Rng:
    """PCG-XSH-RR 32. Deterministic and dependency-free.

    The real game uses `Math.random()`, which we cannot reproduce — so levels
    here are *statistically* like Chrome's, not identical to any particular
    Chrome session. That's fine: the policy must generalise over levels
    anyway, and determinism is what we need for labelling.
    """

    __slots__ = ("state",)

    def __init__(self, seed: int):
        self.state = (seed * 6364136223846793005 + 1) & _MASK64
        self.next_u32()

    def clone(self) -> "Rng":
        r = Rng.__new__(Rng)
        r.state = self.state
        return r

    def next_u32(self) -> int:
        old = self.state
        self.state = (old * 6364136223846793005 + 1442695040888963407) & _MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = (old >> 59) & 31
        return ((xorshifted >> rot) | (xorshifted << (32 - rot))) & 0xFFFFFFFF

    def next_f32(self) -> float:
        return (self.next_u32() >> 8) / float(1 << 24)

    def randint(self, lo: int, hi: int) -> int:
        """Inclusive, matching the game's `getRandomNum(min, max)`."""
        return lo + self.next_u32() % (hi - lo + 1)


@dataclass(slots=True)
class Obstacle:
    kind: str
    size: int
    x: float
    y: float
    w: float
    h: float
    gap: float
    boxes: tuple
    following_created: bool = False

    def distance_from_dino(self) -> float:
        return self.x - (DINO_X + DINO_W)


@dataclass(slots=True)
class Observation:
    """Everything a controller may see, and exactly what gets serialized for
    the model. Keep it small: token count dominates inference latency."""

    distance: float
    speed: float
    obstacle: Optional[str]
    obstacle_w: float
    obstacle_h: float
    obstacle_ground_offset: float
    size: int
    airborne: bool
    dino_y_velocity: float
    frames_to_impact: float


def _boxes_for(kind: str, size: int, width: float) -> tuple:
    """Collision boxes, widened for clustered cacti exactly as the game does:
    the middle box stretches and the right box slides to the new edge."""
    b = [list(x) for x in TYPES[kind]["boxes"]]
    if size > 1:
        b[1][2] = width - b[0][2] - b[2][2]
        b[2][0] = width - b[2][2]
    return tuple(tuple(x) for x in b)


class Game:
    __slots__ = (
        "rng", "speed", "distance_ran", "frame", "dino_y", "dino_vy",
        "jumping", "obstacles", "crashed", "history", "reached_min_height",
    )

    def __init__(self, seed: int):
        self.rng = Rng(seed)
        self.speed = START_SPEED
        self.distance_ran = 0.0
        self.frame = 0
        self.dino_y = GROUND_Y
        self.dino_vy = 0.0
        self.jumping = False
        self.reached_min_height = False
        self.obstacles: list[Obstacle] = []
        self.crashed = False
        self.history: list[str] = []
        self._add_obstacle()

    def clone(self) -> "Game":
        """Fast fork for oracle lookahead; `copy.deepcopy` is ~20x slower and
        this runs tens of millions of times during dataset generation."""
        g = Game.__new__(Game)
        g.rng = self.rng.clone()
        g.speed = self.speed
        g.distance_ran = self.distance_ran
        g.frame = self.frame
        g.dino_y = self.dino_y
        g.dino_vy = self.dino_vy
        g.jumping = self.jumping
        g.reached_min_height = self.reached_min_height
        g.obstacles = [
            Obstacle(o.kind, o.size, o.x, o.y, o.w, o.h, o.gap, o.boxes,
                     o.following_created)
            for o in self.obstacles
        ]
        g.crashed = self.crashed
        g.history = list(self.history)
        return g

    def score(self) -> int:
        return int(self.distance_ran * SCORE_COEFFICIENT)

    def at_max_speed(self) -> bool:
        return self.speed >= MAX_SPEED

    def _next_obstacle(self) -> Optional[Obstacle]:
        pending = [o for o in self.obstacles if o.x + o.w > DINO_X]
        return min(pending, key=lambda o: o.x) if pending else None

    def observe(self) -> Observation:
        o = self._next_obstacle()
        if o is None:
            return Observation(math.inf, self.speed, None, 0.0, 0.0, 0.0, 0,
                               self.jumping, self.dino_vy, math.inf)
        d = o.distance_from_dino()
        return Observation(
            distance=d,
            speed=self.speed,
            obstacle=o.kind,
            obstacle_w=o.w,
            obstacle_h=o.h,
            # Height of the obstacle's bottom edge above the ground line.
            obstacle_ground_offset=(GROUND_Y + DINO_H) - (o.y + o.h),
            size=o.size,
            airborne=self.jumping,
            dino_y_velocity=self.dino_vy,
            frames_to_impact=d / self.speed,
        )

    def _duplicate_check(self, kind: str) -> bool:
        return (len(self.history) >= MAX_OBSTACLE_DUPLICATION
                and all(h == kind for h in self.history[:MAX_OBSTACLE_DUPLICATION]))

    def _add_obstacle(self) -> None:
        # The game retries type selection until one is allowed; bound the loop
        # so a pathological state can't hang the simulator.
        for _ in range(32):
            kind = TYPE_ORDER[self.rng.randint(0, len(TYPE_ORDER) - 1)]
            cfg = TYPES[kind]
            if self._duplicate_check(kind) or self.speed < cfg["min_speed"]:
                continue
            break
        else:
            kind, cfg = CACTUS_SMALL, TYPES[CACTUS_SMALL]

        size = self.rng.randint(1, MAX_OBSTACLE_LENGTH)
        if size > 1 and cfg["multiple_speed"] > self.speed:
            size = 1
        width = cfg["width"] * size

        y_opts = cfg["y_pos"]
        y = y_opts[self.rng.randint(0, len(y_opts) - 1)] if len(y_opts) > 1 else y_opts[0]

        min_gap = round(width * self.speed + cfg["min_gap"] * GAP_COEFFICIENT)
        max_gap = round(min_gap * MAX_GAP_COEFFICIENT)
        gap = float(self.rng.randint(int(min_gap), int(max_gap)))

        self.obstacles.append(Obstacle(
            kind=kind, size=size, x=CANVAS_W, y=y, w=width, h=cfg["height"],
            gap=gap, boxes=_boxes_for(kind, size, width),
        ))
        self.history.insert(0, kind)
        del self.history[MAX_OBSTACLE_DUPLICATION:]

    def _collides(self) -> bool:
        """Chromium's two-stage check: cheap outer-box test, then the detailed
        per-sprite boxes. Using one rectangle instead shifts the last-safe-jump
        frame, which is exactly the boundary we train on."""
        dx, dy = DINO_X, self.dino_y
        for o in self.obstacles:
            # Outer box first.
            if not (dx < o.x + o.w and dx + DINO_W > o.x
                    and dy < o.y + o.h and dy + DINO_H > o.y):
                continue
            for tb in TREX_BOXES:
                tx0, ty0 = dx + tb[0], dy + tb[1]
                tx1, ty1 = tx0 + tb[2], ty0 + tb[3]
                for ob in o.boxes:
                    ox0, oy0 = o.x + ob[0], o.y + ob[1]
                    ox1, oy1 = ox0 + ob[2], oy0 + ob[3]
                    if tx0 < ox1 and tx1 > ox0 and ty0 < oy1 and ty1 > oy0:
                        return True
        return False

    def tick(self, action: int) -> bool:
        """Advance exactly one frame. Returns False once the run has ended."""
        if self.crashed:
            return False

        if action == JUMP and not self.jumping:
            self.jumping = True
            self.reached_min_height = False
            # Chromium: jumpVelocity = INITIAL_JUMP_VELOCITY - (speed / 10)
            self.dino_vy = INITIAL_JUMP_VELOCITY - (self.speed / 10.0)

        if self.jumping:
            # Chromium rounds the per-frame displacement.
            self.dino_y += round(self.dino_vy)
            self.dino_vy += GRAVITY

            if self.dino_y < MIN_JUMP_Y:
                self.reached_min_height = True
            # Reached max height -> endJump(): clamp the ascent. This is what
            # makes the real jump much shorter than a free parabola.
            if self.dino_y < MAX_JUMP_HEIGHT:
                if self.reached_min_height and self.dino_vy < DROP_VELOCITY:
                    self.dino_vy = DROP_VELOCITY

            if self.dino_y > GROUND_Y:
                self.dino_y = GROUND_Y
                self.dino_vy = 0.0
                self.jumping = False
                self.reached_min_height = False

        for o in self.obstacles:
            o.x -= self.speed
        self.obstacles = [o for o in self.obstacles if o.x + o.w > 0]

        if self.obstacles:
            last = self.obstacles[-1]
            if (not last.following_created
                    and last.x + last.w + last.gap < CANVAS_W):
                self._add_obstacle()
                last.following_created = True
        else:
            self._add_obstacle()

        if self._collides():
            self.crashed = True
            return False

        self.distance_ran += self.speed
        if self.speed < MAX_SPEED:
            self.speed += ACCELERATION
        self.frame += 1
        return True
