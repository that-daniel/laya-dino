//! A perfect controller, by brute force.
//!
//! This is the whole reason dino is a good testbed for a one-decision model:
//! ground truth is free and exact. We don't need human annotators or a teacher
//! LLM to tell us whether jumping was correct — we fork the deterministic sim,
//! play both branches out, and see which one survives. Every label is a fact.

use crate::game::{Action, Game, DINO_X};

/// How far ahead to look before giving up. A jump arc is ~33 frames; the
/// widest obstacle gap at minimum speed is comfortably under this.
const HORIZON: u32 = 240;

/// Fork the sim, commit to `first` on this frame, then coast. Returns true if
/// the dino is still alive once the obstacle it was facing is behind it.
fn survives(game: &Game, first: Action) -> bool {
    let mut g = game.clone();

    // Identify the obstacle we're currently deciding about, by position. We
    // follow it rather than "the nearest" so that a newly spawned obstacle
    // doesn't silently change the question mid-simulation.
    let target_x = match g.observe().obstacle {
        Some(_) => g
            .obstacles
            .iter()
            .filter(|o| o.x + o.w > DINO_X)
            .map(|o| o.x)
            .fold(f32::INFINITY, f32::min),
        None => return true,
    };
    let mut tracked = target_x;

    if !g.tick(first) {
        return false;
    }
    tracked -= game.speed;

    for _ in 0..HORIZON {
        // Cleared it: the obstacle's trailing edge is behind the dino.
        if tracked + 50.0 < DINO_X {
            return true;
        }
        let speed = g.speed;
        if !g.tick(Action::Run) {
            return false;
        }
        tracked -= speed;
    }
    true
}

/// The optimal action for this frame.
///
/// Strategy: stay on the ground as long as possible, and jump on the *first*
/// frame from which a jump is verified to clear. Jumping earlier than that
/// means landing on the obstacle; later is unnecessary risk. High-flying
/// pterodactyls are handled for free — running is already safe, so we never
/// jump into them.
pub fn best_action(game: &Game) -> Action {
    if survives(game, Action::Run) {
        Action::Run
    } else if survives(game, Action::Jump) {
        Action::Jump
    } else {
        // Already doomed (shouldn't happen from a reachable state, but if the
        // sim ever generates an impossible gap we'd rather record it than panic).
        Action::Run
    }
}

/// True when this frame is genuinely decidable — i.e. at least one action
/// survives. Undecidable frames are excluded from training data so the model
/// is never punished for an unwinnable state.
pub fn is_decidable(game: &Game) -> bool {
    survives(game, Action::Run) || survives(game, Action::Jump)
}

/// How many consecutive frames, starting now, still permit a successful jump.
///
/// This is the number that decides whether a slow model can play at all. The
/// frame budget (16.7ms) is *not* the decision budget: the dino only has to
/// decide once per obstacle, within the window where a jump still clears. If
/// that window is W frames wide, a controller polling every W frames never
/// misses it — so the real latency target is `W * 16.7ms`, not 16.7ms.
/// The window is the run of consecutive frames in which the dino *must* act
/// (coasting no longer survives) and *can still* act (a jump started now
/// clears). Frames where running is already safe don't count — there is no
/// decision to miss. Frames spent airborne don't count either, since a second
/// jump is a no-op and would read as spuriously "safe".
pub fn jump_window(game: &Game) -> u32 {
    if game.jumping {
        return 0;
    }
    let mut count = 0;
    let mut g = game.clone();
    for _ in 0..HORIZON {
        if g.jumping {
            break;
        }
        let must_act = !survives(&g, Action::Run);
        if must_act && survives(&g, Action::Jump) {
            count += 1;
        } else if count > 0 {
            break; // window has opened and closed
        }
        if !g.tick(Action::Run) {
            break;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Game, MAX_SPEED};

    #[test]
    fn oracle_survives_to_max_speed() {
        let mut g = Game::new(1234);
        while g.frame < 60_000 {
            let a = best_action(&g);
            if !g.tick(a) {
                break;
            }
        }
        assert!(!g.crashed, "oracle crashed at score {}", g.score());
        assert!((g.speed - MAX_SPEED).abs() < 1e-3, "never reached max speed");
        assert!(g.score() > 10_000, "oracle only scored {}", g.score());
    }

    #[test]
    fn oracle_survives_many_seeds() {
        for seed in 0..25u64 {
            let mut g = Game::new(seed);
            while g.frame < 20_000 {
                let a = best_action(&g);
                if !g.tick(a) {
                    break;
                }
            }
            assert!(!g.crashed, "seed {seed} crashed at score {}", g.score());
        }
    }

    #[test]
    fn oracle_does_not_jump_constantly() {
        // If the oracle were degenerate (always jump) the model would learn
        // nothing. Jumps should be a small minority of frames.
        let mut g = Game::new(77);
        let (mut jumps, mut frames) = (0u32, 0u32);
        while g.frame < 10_000 {
            let a = best_action(&g);
            if a == Action::Jump {
                jumps += 1;
            }
            frames += 1;
            if !g.tick(a) {
                break;
            }
        }
        let rate = jumps as f32 / frames as f32;
        assert!(rate < 0.10, "jump rate {rate} — oracle is spamming jumps");
        assert!(jumps > 0, "oracle never jumped");
    }
}
