"""Latency spike: can laya make a jump/no-jump decision inside a 16.7ms frame budget?

Measures p50/p90 latency across checkpoint x device x state size, for the single
binary `noul` question the dino agent would actually ask.
"""

import os, sys, time, json, statistics

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import laya

FRAME_BUDGET_MS = 1000.0 / 60.0

# The one decision the dino agent makes, 60x a second.
JUMP_Q = {
    "jump": {
        "type": "noul",
        "instructions": "Should the dino jump right now to clear the next obstacle?",
    }
}

# Realistic dino state: ~30 tokens.
TINY_STATE = {
    "dist": 118,
    "speed": 13.2,
    "obstacle": "cactus_large",
    "airborne": False,
}

# Padding to simulate the ticket/email-sized inputs laya was benchmarked on.
def padded_state(n_words: int) -> dict:
    s = dict(TINY_STATE)
    if n_words:
        s["log"] = " ".join(f"frame{i} ok" for i in range(n_words // 2))
    return s


STATES = {
    "tiny(~30tok)": padded_state(0),
    "med(~150tok)": padded_state(120),
    "large(~500tok)": padded_state(450),
}


def n_tokens(agent, state) -> int:
    """Best-effort token count of the serialized state."""
    try:
        tok = agent.tokenizer
        return len(tok(str(state))["input_ids"])
    except Exception:
        return -1


def bench(agent, state, iters=30, warmup=5):
    for _ in range(warmup):
        agent.predict(state, JUMP_Q)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        agent.predict(state, JUMP_Q)
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    return {
        "p50": statistics.median(ts),
        "p90": ts[int(len(ts) * 0.9)],
        "min": ts[0],
    }


def main():
    checkpoints = [
        ("laya (ModernBERT-large 421M)", "convaiinnovations/laya"),
        ("laya-multilingual (mmBERT-base 322M)", "convaiinnovations/laya-multilingual"),
    ]
    devices = ["cpu", "mps"]

    rows = []
    for ck_name, ck_id in checkpoints:
        for dev in devices:
            try:
                t0 = time.perf_counter()
                agent = laya.load(ck_id, device=dev)
                load_s = time.perf_counter() - t0
            except Exception as e:
                print(f"SKIP {ck_name} on {dev}: {type(e).__name__}: {e}", flush=True)
                continue
            print(f"\n=== {ck_name} | device={dev} | loaded in {load_s:.1f}s ===", flush=True)
            for sname, state in STATES.items():
                try:
                    r = bench(agent, state)
                except Exception as e:
                    print(f"  {sname:16s} FAILED {type(e).__name__}: {e}", flush=True)
                    continue
                ntok = n_tokens(agent, state)
                verdict = "REAL-TIME" if r["p50"] < FRAME_BUDGET_MS else f"{r['p50']/FRAME_BUDGET_MS:.1f}x over"
                print(
                    f"  {sname:16s} tok={ntok:4d}  p50={r['p50']:7.1f}ms  "
                    f"p90={r['p90']:7.1f}ms  min={r['min']:7.1f}ms  [{verdict}]",
                    flush=True,
                )
                rows.append({"ckpt": ck_name, "device": dev, "state": sname,
                             "tokens": ntok, **r})
            del agent

    with open("bench_results.json", "w") as f:
        json.dump({"frame_budget_ms": FRAME_BUDGET_MS, "rows": rows}, f, indent=2)
    print(f"\nFrame budget @60fps = {FRAME_BUDGET_MS:.1f}ms")
    print("wrote bench_results.json")


if __name__ == "__main__":
    main()
