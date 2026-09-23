//! Turning a game frame into the text `state` laya reads.
//!
//! laya is an encoder over text, so *how* we write the frame is a real design
//! variable, not a formatting detail. Two things are at stake:
//!
//!   1. Token count is the dominant term in inference latency, and we are
//!      fighting for a 16.7ms frame budget. Every token costs.
//!   2. Encoders are historically weak at numeric thresholds, which is exactly
//!      what this task is. Bucketing the distance makes the problem trivial —
//!      but then the model is only reading a label the sim already computed,
//!      which proves nothing. Keeping it numeric is the honest test.
//!
//! So we ship both and measure the difference.

use crate::game::Observation;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Encoding {
    /// Raw numbers. The honest, hard version.
    Numeric,
    /// Pre-bucketed qualitative labels. The easy version — included as a
    /// control, to show how much of any success is the model versus the
    /// encoding doing the work for it.
    Bucketed,
}

impl Encoding {
    pub fn as_str(self) -> &'static str {
        match self {
            Encoding::Numeric => "numeric",
            Encoding::Bucketed => "bucketed",
        }
    }
}

fn bucket_distance(d: f32) -> &'static str {
    match d {
        d if d.is_infinite() => "none",
        d if d < 0.0 => "passed",
        d if d < 40.0 => "imminent",
        d if d < 90.0 => "close",
        d if d < 160.0 => "near",
        d if d < 300.0 => "far",
        _ => "distant",
    }
}

fn bucket_speed(s: f32) -> &'static str {
    match s {
        s if s < 7.5 => "slow",
        s if s < 10.0 => "medium",
        s if s < 12.0 => "fast",
        _ => "max",
    }
}

/// Render the observation as a compact key=value line.
///
/// Deliberately not JSON: braces, quotes and colons all cost tokens and add
/// nothing the model needs. This form is roughly half the tokens of the
/// equivalent JSON object.
pub fn encode(obs: &Observation, enc: Encoding) -> String {
    let obstacle = obs.obstacle.map(|k| k.as_str()).unwrap_or("none");

    match enc {
        Encoding::Numeric => {
            if obs.obstacle.is_none() {
                return format!(
                    "obstacle=none speed={:.1} airborne={}",
                    obs.speed,
                    obs.airborne as u8
                );
            }
            format!(
                "obstacle={} dist={:.0} speed={:.1} w={:.0} h={:.0} alt={:.0} airborne={} vy={:.1}",
                obstacle,
                obs.distance,
                obs.speed,
                obs.obstacle_w,
                obs.obstacle_h,
                obs.obstacle_ground_offset,
                obs.airborne as u8,
                obs.dino_y_velocity,
            )
        }
        Encoding::Bucketed => {
            if obs.obstacle.is_none() {
                return format!(
                    "obstacle=none speed={} airborne={}",
                    bucket_speed(obs.speed),
                    obs.airborne as u8
                );
            }
            let height = if obs.obstacle_ground_offset > 20.0 {
                "high"
            } else {
                "ground"
            };
            format!(
                "obstacle={} dist={} speed={} height={} airborne={}",
                obstacle,
                bucket_distance(obs.distance),
                bucket_speed(obs.speed),
                height,
                obs.airborne as u8,
            )
        }
    }
}

/// The single question the agent asks, 60 times a second.
pub const JUMP_QUESTION: &str =
    "Should the dino jump right now to clear the next obstacle?";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game::{Action, Game};

    #[test]
    fn encodings_are_compact() {
        let g = Game::new(1);
        for enc in [Encoding::Numeric, Encoding::Bucketed] {
            let s = encode(&g.observe(), enc);
            // Rough proxy for tokens; we verify the real count in the spike.
            assert!(s.len() < 120, "{} encoding too long: {s:?}", enc.as_str());
            assert!(!s.contains('\n'));
        }
    }

    #[test]
    fn handles_empty_field() {
        let mut g = Game::new(2);
        g.obstacles.clear();
        let s = encode(&g.observe(), Encoding::Numeric);
        assert!(s.contains("obstacle=none"), "{s}");
        assert!(!s.contains("inf"), "infinity leaked into the prompt: {s}");
    }

    #[test]
    fn airborne_is_visible_to_the_model() {
        let mut g = Game::new(3);
        g.tick(Action::Jump);
        let s = encode(&g.observe(), Encoding::Numeric);
        assert!(s.contains("airborne=1"), "{s}");
    }
}
