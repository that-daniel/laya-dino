"""Render a run to video.

Rendering is separated from playing on purpose. `play.py` produces a replay —
a seed plus the action taken on every frame — and this script redraws it at a
true 60fps. Because the sim is deterministic, replaying the action list
reproduces the run exactly, frame for frame.

That separation is what makes the video honest. Inference takes ~30ms while a
frame lasts 16.7ms, so drawing live would either stutter or silently drop
frames. Instead the model plays at its own pace, and we render the resulting
run at the speed the game actually runs at. Nothing is sped up and no decision
is faked: the overlay shows the model's real probability and real latency for
every frame.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from laya_dino.game import (CANVAS_H, CANVAS_W, DINO_H, DINO_W, DINO_X,
                            GROUND_Y, Game)

SCALE = 2
W, H = int(CANVAS_W * SCALE), int(CANVAS_H * SCALE) + 60

BG = (247, 247, 247)
FG = (83, 83, 83)
ACCENT = (200, 60, 60)
DIM = (170, 170, 170)


def draw_frame(g: Game, prob: float | None, latency: float | None,
               decided: bool) -> Image.Image:
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    ground = (GROUND_Y + DINO_H) * SCALE
    d.line([(0, ground), (W, ground)], fill=FG, width=2)

    # Dino: a blocky stand-in. Chrome's sprite is Google's; this keeps the
    # repo shareable without a licensing question.
    dx, dy = DINO_X * SCALE, g.dino_y * SCALE
    dw, dh = DINO_W * SCALE, DINO_H * SCALE
    d.rectangle([dx, dy + dh * 0.35, dx + dw * 0.62, dy + dh], fill=FG)       # body
    d.rectangle([dx + dw * 0.55, dy, dx + dw, dy + dh * 0.42], fill=FG)       # head
    d.rectangle([dx + dw * 0.86, dy + dh * 0.12,
                 dx + dw * 0.94, dy + dh * 0.2], fill=BG)                     # eye

    for o in g.obstacles:
        x0, y0 = o.x * SCALE, o.y * SCALE
        x1, y1 = (o.x + o.w) * SCALE, (o.y + o.h) * SCALE
        if x1 < 0 or x0 > W:
            continue
        d.rectangle([x0, y0, x1, y1], fill=FG)

    # Overlay: score, speed, and the model's actual output this frame.
    y = int(CANVAS_H * SCALE) + 8
    d.text((10, y), f"score {g.score():>6}", fill=FG)
    d.text((140, y), f"speed {g.speed:>5.2f}", fill=FG)

    if prob is not None:
        col = ACCENT if prob >= 0.5 else FG
        d.text((280, y), f"P(jump) {prob:>5.3f}", fill=col)
        bar_x, bar_w = 280, 200
        d.rectangle([bar_x, y + 18, bar_x + bar_w, y + 28], outline=DIM)
        d.rectangle([bar_x, y + 18, bar_x + bar_w * prob, y + 28], fill=col)
        d.line([(bar_x + bar_w * 0.5, y + 15),
                (bar_x + bar_w * 0.5, y + 31)], fill=FG, width=1)
    if latency is not None:
        d.text((520, y), f"{latency:>5.1f} ms", fill=FG if decided else DIM)
    d.text((520, y + 18), "laya decision" if decided else "held", fill=DIM)
    return img


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--replay", type=Path, default=Path("results/replay.json"))
    p.add_argument("--out", type=Path, default=Path("results/dino.mp4"))
    p.add_argument("--max-frames", type=int, default=3600, help="60 = 1 second")
    p.add_argument("--fps", type=int, default=60)
    args = p.parse_args()

    replay = json.loads(args.replay.read_text())
    actions = replay["actions"]
    probs = replay.get("probs", [])
    lats = replay.get("latencies", [])
    g = Game(replay["seed"])

    n = min(len(actions), args.max_frames)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        for i in range(n):
            pr = probs[i] if i < len(probs) else None
            la = lats[i] if i < len(lats) else None
            draw_frame(g, pr, la, decided=la is not None).save(
                tmpdir / f"f{i:06d}.png")
            if not g.tick(actions[i]):
                # Hold the crash frame briefly so the ending is visible.
                for k in range(args.fps):
                    draw_frame(g, pr, la, decided=False).save(
                        tmpdir / f"f{i + 1 + k:06d}.png")
                break

        args.out.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["ffmpeg", "-y", "-framerate", str(args.fps),
               "-i", str(tmpdir / "f%06d.png"),
               "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
               str(args.out)]
        subprocess.run(cmd, check=True, capture_output=True)

    print(f"wrote {args.out} ({n} frames, {n / args.fps:.1f}s, "
          f"final score {g.score()})")


if __name__ == "__main__":
    main()
