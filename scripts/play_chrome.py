"""Play the *real* Chrome dino game in a real browser, with the fine-tuned model.

This is the test the sim can't do. In the sim the game only advances when we
call `tick()`, so inference latency is free. Here the game runs on the
browser's own `requestAnimationFrame` clock and does not wait for anything.
Every decision has to arrive while the jump window is still open, for real.

`chrome://dino` itself cannot be automated — Chrome blocks DevTools control of
`chrome://` URLs — so we serve the Chromium t-rex runner source locally
(`third_party/t-rex-runner`). It is the same game code, same constants, just
reachable.

State comes straight out of the game's own globals rather than from pixels:

    Runner.instance_.currentSpeed
    Runner.instance_.horizon.obstacles[0].xPos
    Runner.instance_.tRex.jumping

which is exactly the handful of numbers our prompt needs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from laya_dino import state
from laya_dino.game import DINO_H, DINO_W, GROUND_Y, Observation

ROOT = Path(__file__).resolve().parents[1]
GAME_DIR = ROOT / "third_party" / "t-rex-runner"

# One evaluate() round-trip instead of several: every millisecond here comes
# straight out of the jump window.
READ_STATE_JS = """
() => {
  const R = window.Runner && window.Runner.instance_;
  if (!R) return null;
  const t = R.tRex;
  const obs = R.horizon.obstacles;
  let best = null;
  for (const o of obs) {
    if (o.xPos + o.width > t.xPos) {
      if (!best || o.xPos < best.xPos) best = o;
    }
  }
  return {
    speed: R.currentSpeed,
    distance: R.distanceRan,
    crashed: R.crashed,
    playing: R.playing,
    activated: R.activated,
    trexX: t.xPos, trexY: t.yPos,
    jumping: t.jumping,
    jumpVelocity: t.jumpVelocity,
    obstacle: best ? {
      type: best.typeConfig.type,
      x: best.xPos, y: best.yPos,
      w: best.width, h: best.typeConfig.height,
      size: best.size,
    } : null,
  };
}
"""

TYPE_MAP = {
    "CACTUS_SMALL": "cactus_small",
    "CACTUS_LARGE": "cactus_large",
    "PTERODACTYL": "pterodactyl",
}


def to_observation(s: dict) -> Observation:
    """Map the live browser state onto the same Observation the model trained on.

    The arithmetic must match `Game.observe()` exactly, or the model sees a
    subtly different world than the one it learned. Ground level is
    `GROUND_Y + DINO_H` = 140 in both.
    """
    o = s["obstacle"]
    if o is None:
        return Observation(float("inf"), s["speed"], None, 0.0, 0.0, 0.0, 0,
                           bool(s["jumping"]), s["jumpVelocity"] or 0.0,
                           float("inf"))
    dist = o["x"] - (s["trexX"] + DINO_W)
    return Observation(
        distance=dist,
        speed=s["speed"],
        obstacle=TYPE_MAP.get(o["type"], "cactus_small"),
        obstacle_w=float(o["w"]),
        obstacle_h=float(o["h"]),
        obstacle_ground_offset=(GROUND_Y + DINO_H) - (o["y"] + o["h"]),
        size=int(o["size"]),
        airborne=bool(s["jumping"]),
        dino_y_velocity=float(s["jumpVelocity"] or 0.0),
        frames_to_impact=dist / max(s["speed"], 1e-6),
    )


def serve_game(port: int) -> ThreadingHTTPServer:
    handler = partial(SimpleHTTPRequestHandler, directory=str(GAME_DIR))
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path,
                   default=Path("checkpoints/dino_head_numeric.pt"))
    p.add_argument("--games", type=int, default=3)
    p.add_argument("--threshold", type=float, default=0.99)
    p.add_argument("--port", type=int, default=8777)
    p.add_argument("--max-seconds", type=float, default=180.0)
    p.add_argument("--headed", action="store_true")
    p.add_argument("--video-dir", type=Path, default=Path("results/video"))
    p.add_argument("--out", type=Path, default=Path("results/chrome.json"))
    args = p.parse_args()

    from playwright.sync_api import sync_playwright

    from play import LayaController  # local import: needs torch loaded first

    ctrl = LayaController(args.checkpoint, args.threshold)
    print(f"model: {args.checkpoint} enc={ctrl.encoding} T={ctrl.temperature:.3f} "
          f"device={ctrl.device}", flush=True)

    httpd = serve_game(args.port)
    url = f"http://127.0.0.1:{args.port}/index.html"
    print(f"serving {GAME_DIR.name} at {url}", flush=True)

    runs, all_lat, decisions_log = [], [], []

    with sync_playwright() as pw:
        browser = pw.chromium.launch(
            headless=not args.headed,
            args=["--disable-background-timer-throttling",
                  "--disable-renderer-backgrounding",
                  "--disable-backgrounding-occluded-windows"],
        )
        args.video_dir.mkdir(parents=True, exist_ok=True)

        for gi in range(args.games):
            # One browser context per run, so each game records to its own
            # video file. Slicing a single long recording by accumulated
            # durations does not work: the gaps between games are not
            # constant, the offsets drift, and you end up cutting the wrong
            # game entirely.
            context = browser.new_context(
                viewport={"width": 900, "height": 400},
                record_video_dir=str(args.video_dir),
                record_video_size={"width": 900, "height": 400},
            )
            page = context.new_page()

            # Fresh page per run.
            #
            # `Runner.restart()` is guarded by `if (!this.raqId)`, so it only
            # works once the game loop has actually stopped — i.e. after a
            # crash. Calling it on a run that ended on our time limit is a
            # no-op, which previously left the next game reading the old
            # session and reporting zero decisions. Reloading costs ~1s and
            # removes every restart edge case.
            page.goto(url)
            page.wait_for_function(
                "() => window.Runner && window.Runner.instance_", timeout=20_000)
            page.keyboard.press("Space")
            page.wait_for_function(
                "() => { const R = window.Runner.instance_;"
                " return R.playing && !R.crashed; }", timeout=20_000)
            time.sleep(0.5)

            t_start = time.perf_counter()
            lat, probs, n_dec = [], [], 0
            last_state = None

            while time.perf_counter() - t_start < args.max_seconds:
                s = page.evaluate(READ_STATE_JS)
                if s is None:
                    continue
                last_state = s
                if s["crashed"]:
                    break
                if not s["jumping"]:
                    t0 = time.perf_counter()
                    pr = ctrl.jump_prob_from_obs(to_observation(s))
                    lat.append((time.perf_counter() - t0) * 1000.0)
                    probs.append(pr)
                    n_dec += 1
                    if pr >= args.threshold:
                        page.keyboard.press("Space")

            score = int((last_state or {}).get("distance", 0) * 0.025)
            crashed = bool((last_state or {}).get("crashed", True))
            speed = (last_state or {}).get("speed", 0)
            # A run with no decisions is a harness failure, not a result.
            # Recording it as a score would silently poison the median.
            valid = n_dec > 0
            runs.append({"game": gi + 1, "score": score, "crashed": crashed,
                         "speed": round(speed, 2), "decisions": n_dec,
                         "valid": valid,
                         "seconds": round(time.perf_counter() - t_start, 1)})
            if not valid:
                print(f"  game {gi+1}: INVALID (0 decisions) — excluded",
                      flush=True)
            all_lat.extend(lat)
            decisions_log.append({"game": gi + 1, "probs": probs[-4000:]})
            print(f"  game {gi+1}/{args.games}: score={score} crashed={crashed} "
                  f"speed={speed:.2f} decisions={n_dec}", flush=True)

            # Close the context to flush this run's video, then rename it
            # after the score so the file is self-describing.
            video = page.video
            vpath = video.path() if video else None
            context.close()
            if vpath:
                src = Path(vpath)
                if src.exists():
                    dst = src.with_name(
                        f"game{gi+1:02d}_score{score}"
                        f"{'_survived' if not crashed else ''}.webm")
                    src.rename(dst)
                    runs[-1]["video"] = str(dst)

        browser.close()

    httpd.shutdown()

    valid_runs = [r for r in runs if r.get("valid", True)]
    scores = sorted(r["score"] for r in valid_runs)
    lat_s = sorted(all_lat)
    summary = {
        "games": len(valid_runs),
        "games_attempted": len(runs),
        "median_score": statistics.median(scores) if scores else 0,
        "best_score": scores[-1] if scores else 0,
        "survived": sum(not r["crashed"] for r in valid_runs),
        "latency_ms": {
            "p50": lat_s[len(lat_s) // 2] if lat_s else None,
            "p90": lat_s[int(len(lat_s) * 0.9)] if lat_s else None,
        },
        "runs": runs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"summary": summary, "decisions": decisions_log}, indent=2))

    print(f"\nreal Chrome dino: median={summary['median_score']} "
          f"best={summary['best_score']} survived={summary['survived']}/{len(valid_runs)}")
    if lat_s:
        print(f"decision latency p50={summary['latency_ms']['p50']:.1f}ms "
              f"p90={summary['latency_ms']['p90']:.1f}ms")
    print(f"wrote {args.out}; video in {args.video_dir}")


if __name__ == "__main__":
    main()
