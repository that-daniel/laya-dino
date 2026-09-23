//! Deterministic Chrome-dino simulation.
//!
//! Constants follow the Chromium t-rex runner (`offline/`), so behaviour and
//! difficulty ramp match the real game. The sim is a pure fixed-timestep state
//! machine: no clock, no rendering, no I/O. One `tick` == one frame at 60fps.
//! Given the same seed and the same action sequence it always replays
//! identically, which is what makes it usable for dataset generation and for
//! decoupling the game clock from inference latency.

pub const CANVAS_W: f32 = 600.0;
pub const CANVAS_H: f32 = 150.0;

pub const GRAVITY: f32 = 0.6;
pub const INITIAL_JUMP_VELOCITY: f32 = -10.0;
pub const START_SPEED: f32 = 6.0;
pub const MAX_SPEED: f32 = 13.0;
pub const ACCELERATION: f32 = 0.001;

pub const DINO_X: f32 = 25.0;
pub const DINO_W: f32 = 44.0;
pub const DINO_H: f32 = 47.0;
pub const GROUND_Y: f32 = CANVAS_H - DINO_H - 10.0;

/// Chromium multiplies distance by this to get the displayed score.
pub const SCORE_COEFFICIENT: f32 = 0.025;

/// Collision boxes are inset slightly, mirroring the real game's forgiving
/// per-sprite hitboxes without modelling each one.
const HITBOX_INSET: f32 = 4.0;

const MIN_GAP_COEFFICIENT: f32 = 0.6;
const MAX_GAP_COEFFICIENT: f32 = 1.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObstacleKind {
    CactusSmall,
    CactusLarge,
    Pterodactyl,
}

impl ObstacleKind {
    pub fn size(self) -> (f32, f32) {
        match self {
            ObstacleKind::CactusSmall => (17.0, 35.0),
            ObstacleKind::CactusLarge => (25.0, 50.0),
            ObstacleKind::Pterodactyl => (46.0, 40.0),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            ObstacleKind::CactusSmall => "cactus_small",
            ObstacleKind::CactusLarge => "cactus_large",
            ObstacleKind::Pterodactyl => "pterodactyl",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Obstacle {
    pub kind: ObstacleKind,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
}

impl Obstacle {
    /// Horizontal gap between the dino's leading edge and this obstacle.
    /// Negative once the dino has passed it.
    pub fn distance_from_dino(&self) -> f32 {
        self.x - (DINO_X + DINO_W)
    }
}

/// Small deterministic PRNG (PCG-XSH-RR 32). Avoids a dependency and
/// guarantees byte-identical replays across platforms and Rust versions.
#[derive(Clone, Copy, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut r = Rng { state: seed.wrapping_mul(6364136223846793005).wrapping_add(1) };
        r.next_u32();
        r
    }

    pub fn next_u32(&mut self) -> u32 {
        let old = self.state;
        self.state = old
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let xorshifted = (((old >> 18) ^ old) >> 27) as u32;
        let rot = (old >> 59) as u32;
        xorshifted.rotate_right(rot)
    }

    /// Uniform in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }

    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }
}

/// The action the controller may take on a given frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Action {
    Run,
    Jump,
}

/// Everything a controller is allowed to see. This is also exactly what gets
/// serialized and handed to the model, so it must stay small — token count is
/// the dominant term in inference latency.
#[derive(Clone, Copy, Debug)]
pub struct Observation {
    /// Pixels between the dino and the next obstacle. `f32::INFINITY` if none.
    pub distance: f32,
    pub speed: f32,
    pub obstacle: Option<ObstacleKind>,
    pub obstacle_w: f32,
    pub obstacle_h: f32,
    /// Height of the obstacle's bottom edge above ground (pterodactyls fly).
    pub obstacle_ground_offset: f32,
    pub airborne: bool,
    pub dino_y_velocity: f32,
    /// Frames until the obstacle reaches the dino at the current speed.
    pub frames_to_impact: f32,
}

#[derive(Clone, Debug)]
pub struct Game {
    pub rng: Rng,
    pub speed: f32,
    pub distance_ran: f32,
    pub frame: u64,
    pub dino_y: f32,
    pub dino_vy: f32,
    pub jumping: bool,
    pub obstacles: Vec<Obstacle>,
    pub crashed: bool,
    next_spawn_gap: f32,
}

impl Game {
    pub fn new(seed: u64) -> Self {
        let mut g = Game {
            rng: Rng::new(seed),
            speed: START_SPEED,
            distance_ran: 0.0,
            frame: 0,
            dino_y: GROUND_Y,
            dino_vy: 0.0,
            jumping: false,
            obstacles: Vec::with_capacity(8),
            crashed: false,
            next_spawn_gap: 0.0,
        };
        g.spawn_obstacle();
        g
    }

    pub fn score(&self) -> u32 {
        (self.distance_ran * SCORE_COEFFICIENT) as u32
    }

    pub fn at_max_speed(&self) -> bool {
        self.speed >= MAX_SPEED
    }

    /// Obstacles the dino has not yet cleared, nearest first.
    fn next_obstacle(&self) -> Option<&Obstacle> {
        self.obstacles
            .iter()
            .filter(|o| o.x + o.w > DINO_X)
            .min_by(|a, b| a.x.partial_cmp(&b.x).unwrap())
    }

    pub fn observe(&self) -> Observation {
        match self.next_obstacle() {
            Some(o) => Observation {
                distance: o.distance_from_dino(),
                speed: self.speed,
                obstacle: Some(o.kind),
                obstacle_w: o.w,
                obstacle_h: o.h,
                obstacle_ground_offset: (GROUND_Y + DINO_H) - (o.y + o.h),
                airborne: self.jumping,
                dino_y_velocity: self.dino_vy,
                frames_to_impact: o.distance_from_dino() / self.speed,
            },
            None => Observation {
                distance: f32::INFINITY,
                speed: self.speed,
                obstacle: None,
                obstacle_w: 0.0,
                obstacle_h: 0.0,
                obstacle_ground_offset: 0.0,
                airborne: self.jumping,
                dino_y_velocity: self.dino_vy,
                frames_to_impact: f32::INFINITY,
            },
        }
    }

    fn spawn_obstacle(&mut self) {
        // Pterodactyls only appear later, as in the real game.
        let kind = if self.score() > 450 && self.rng.next_f32() < 0.2 {
            ObstacleKind::Pterodactyl
        } else if self.rng.next_f32() < 0.5 {
            ObstacleKind::CactusLarge
        } else {
            ObstacleKind::CactusSmall
        };
        let (w, h) = kind.size();

        // Pterodactyls spawn at one of three altitudes; the lowest is jumpable,
        // the highest can be run under.
        let ground_offset = if kind == ObstacleKind::Pterodactyl {
            [0.0, 25.0, 50.0][(self.rng.next_u32() % 3) as usize]
        } else {
            0.0
        };
        let y = (GROUND_Y + DINO_H) - h - ground_offset;

        self.obstacles.push(Obstacle { kind, x: CANVAS_W, y, w, h });

        let gap_base = w * self.speed + 40.0 * self.speed;
        self.next_spawn_gap = gap_base * self.rng.range(MIN_GAP_COEFFICIENT, MAX_GAP_COEFFICIENT);
    }

    fn collides(&self) -> bool {
        let dx0 = DINO_X + HITBOX_INSET;
        let dx1 = DINO_X + DINO_W - HITBOX_INSET;
        let dy0 = self.dino_y + HITBOX_INSET;
        let dy1 = self.dino_y + DINO_H - HITBOX_INSET;
        self.obstacles.iter().any(|o| {
            let ox0 = o.x + HITBOX_INSET;
            let ox1 = o.x + o.w - HITBOX_INSET;
            let oy0 = o.y + HITBOX_INSET;
            let oy1 = o.y + o.h - HITBOX_INSET;
            dx0 < ox1 && dx1 > ox0 && dy0 < oy1 && dy1 > oy0
        })
    }

    /// Advance exactly one frame. Returns `false` once the run has ended.
    pub fn tick(&mut self, action: Action) -> bool {
        if self.crashed {
            return false;
        }

        if action == Action::Jump && !self.jumping {
            self.jumping = true;
            self.dino_vy = INITIAL_JUMP_VELOCITY;
        }

        if self.jumping {
            self.dino_y += self.dino_vy;
            self.dino_vy += GRAVITY;
            if self.dino_y >= GROUND_Y {
                self.dino_y = GROUND_Y;
                self.dino_vy = 0.0;
                self.jumping = false;
            }
        }

        for o in self.obstacles.iter_mut() {
            o.x -= self.speed;
        }
        self.obstacles.retain(|o| o.x + o.w > -50.0);

        let rightmost = self.obstacles.iter().map(|o| o.x).fold(f32::MIN, f32::max);
        if self.obstacles.is_empty() || CANVAS_W - rightmost >= self.next_spawn_gap {
            self.spawn_obstacle();
        }

        if self.collides() {
            self.crashed = true;
            return false;
        }

        self.distance_ran += self.speed;
        if self.speed < MAX_SPEED {
            self.speed += ACCELERATION;
        }
        self.frame += 1;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replays_are_deterministic() {
        let run = |seed| {
            let mut g = Game::new(seed);
            let mut acts = vec![];
            let mut r = Rng::new(7);
            while g.tick(if r.next_f32() < 0.02 { Action::Jump } else { Action::Run }) {
                acts.push(g.distance_ran);
                if g.frame > 5_000 {
                    break;
                }
            }
            (g.frame, g.score(), acts.len())
        };
        assert_eq!(run(42), run(42));
    }

    #[test]
    fn different_seeds_diverge() {
        // Note: time-to-first-crash is *not* a valid signal here — the opening
        // obstacle always spawns at the right edge, so every seed dies on the
        // same frame if you never jump. Compare the generated layout instead.
        let layout = |seed| {
            let mut g = Game::new(seed);
            let mut kinds = vec![];
            for _ in 0..3_000 {
                g.crashed = false;
                g.tick(Action::Run);
                if let Some(o) = g.obstacles.last() {
                    if kinds.last() != Some(&(o.kind, o.w as i32)) {
                        kinds.push((o.kind, o.w as i32));
                    }
                }
            }
            kinds
        };
        assert_ne!(layout(1), layout(2), "seeds should produce different obstacle layouts");
    }

    #[test]
    fn standing_still_eventually_crashes() {
        let mut g = Game::new(3);
        let mut frames = 0;
        while g.tick(Action::Run) {
            frames += 1;
            assert!(frames < 10_000, "never hit an obstacle while standing still");
        }
        assert!(g.crashed);
    }

    #[test]
    fn a_jump_returns_to_ground() {
        let mut g = Game::new(9);
        g.tick(Action::Jump);
        assert!(g.jumping);
        let mut frames = 0;
        while g.jumping && g.tick(Action::Run) {
            frames += 1;
            assert!(frames < 200, "jump never landed");
        }
        // ~33 frames of hang time at gravity 0.6 / v0 -10.
        assert!((30..40).contains(&frames), "unexpected hang time: {frames}");
    }

    #[test]
    fn speed_ramps_to_cap_and_holds() {
        let mut g = Game::new(5);
        for _ in 0..20_000 {
            g.crashed = false; // ignore collisions; we only care about the ramp
            g.tick(Action::Run);
        }
        assert!((g.speed - MAX_SPEED).abs() < 1e-3, "speed = {}", g.speed);
    }
}
