//! Measures the decision budget: how long the dino may dither and still live.
//!
//! Reports, per speed band, the width of the window in which a jump still
//! clears the obstacle — in frames and in milliseconds at 60fps. The p01 /
//! minimum is the number that matters, since one missed window ends the run.

use laya_dino::game::{Action, Game, MAX_SPEED, START_SPEED};
use laya_dino::oracle::{best_action, jump_window};

const MS_PER_FRAME: f32 = 1000.0 / 60.0;
const BANDS: usize = 8;

fn main() {
    let mut windows: Vec<Vec<u32>> = vec![Vec::new(); BANDS];
    let mut total_obstacles = 0u32;

    for seed in 0..40u64 {
        let mut g = Game::new(seed);
        let mut in_window = false;
        while g.frame < 40_000 {
            let w = if g.jumping { 0 } else { jump_window(&g) };
            // Record once per obstacle, at the moment the window opens.
            if w > 0 && !in_window {
                let frac = (g.speed - START_SPEED) / (MAX_SPEED - START_SPEED);
                let band = ((frac * BANDS as f32) as usize).min(BANDS - 1);
                windows[band].push(w);
                total_obstacles += 1;
                in_window = true;
            } else if w == 0 {
                in_window = false;
            }
            if !g.tick(best_action(&g)) {
                break;
            }
        }
        if g.crashed {
            eprintln!("warning: oracle crashed on seed {seed} at score {}", g.score());
        }
    }

    println!("Jump-window width by speed band ({total_obstacles} obstacles, 40 seeds)\n");
    println!(
        "{:<14} {:>6} {:>8} {:>8} {:>10}",
        "speed", "n", "min", "median", "budget@min"
    );
    println!("{}", "-".repeat(50));

    for (i, w) in windows.iter().enumerate() {
        if w.is_empty() {
            continue;
        }
        let mut s = w.clone();
        s.sort_unstable();
        let lo = START_SPEED + (MAX_SPEED - START_SPEED) * (i as f32 / BANDS as f32);
        let hi = START_SPEED + (MAX_SPEED - START_SPEED) * ((i + 1) as f32 / BANDS as f32);
        let min = s[0];
        let med = s[s.len() / 2];
        println!(
            "{:<14} {:>6} {:>8} {:>8} {:>9.0}ms",
            format!("{lo:.1}-{hi:.1}"),
            s.len(),
            min,
            med,
            min as f32 * MS_PER_FRAME
        );
    }

    let mut all: Vec<u32> = windows.concat();
    all.sort_unstable();
    let worst = all[0];
    let p01 = all[all.len() / 100];
    println!(
        "\nworst window overall: {worst} frames ({:.0}ms)   p01: {p01} frames ({:.0}ms)",
        worst as f32 * MS_PER_FRAME,
        p01 as f32 * MS_PER_FRAME
    );
    println!("laya best measured p50: 29.6ms (mmBERT-base, MPS, short state)");
    let verdict = if worst as f32 * MS_PER_FRAME > 29.6 {
        "FITS — a 29.6ms decision never misses a window"
    } else {
        "TOO SLOW — some windows close before inference returns"
    };
    println!("verdict: {verdict}");

    // Sanity: the oracle itself must be flawless, or every number above is junk.
    let mut g = Game::new(999);
    while g.frame < 60_000 && g.tick(best_action(&g)) {}
    println!(
        "\noracle control run: score {} at speed {:.2}, crashed={}",
        g.score(),
        g.speed,
        g.crashed
    );
    let _ = Action::Run;
}
