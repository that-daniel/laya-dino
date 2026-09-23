"""Fine-tune laya's decision head to play dino.

We train the *head* only and leave the 322M-parameter encoder frozen. Reasons:

* The encoder already knows how to read `dist=118 speed=13.2`. What it doesn't
  know is the threshold rule that turns those numbers into a jump. That rule
  lives in the head.
* A full fine-tune needs optimizer state for every parameter — roughly 4-5GB on
  top of the weights, which is uncomfortable on a 16GB machine shared with a
  browser. Head-only trains in minutes instead of hours.
* `DecisionModel.forward` already accepts `detach_encoder`, so this is a
  supported path rather than a hack.

If head-only underperforms, the fallback is a full fine-tune on Kaggle's free
2xT4 using laya's own RLCD notebook.

The model is a `noul` (binary) question, so the two marker logits are
[false, true] and plain cross-entropy is the right objective. We additionally
fit a temperature afterwards, because laya ships uncalibrated confidences — it
says so in a runtime warning on load — and calibration is half the point of
this experiment.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import laya
from laya.common import QTYPES, build_sequence, ece_score

QUESTION = {"t": "noul", "ins": "Should the dino jump right now to clear the next obstacle?",
            "crit": None}


def obstacle_class(text: str) -> str:
    """Coarse obstacle identity, parsed back out of the encoded state.

    Pterodactyl altitude is part of the identity, not a detail: at alt 0 and 25
    the bird must be jumped, at alt 50 it must be run under, and the jump
    timing differs from a ground cactus because the obstacle is raised.
    """
    parts = text.split()
    kind = parts[0].removeprefix("obstacle=")
    if kind != "pterodactyl":
        return kind
    alt = next((p[4:] for p in parts if p.startswith("alt=")), "?")
    return f"pterodactyl@{alt}"


def _dist_band(d) -> str:
    """Coarse distance bands for stratified sampling.

    Boundaries are deliberately finer near the decision point (where the label
    flips) and coarser far away (where the answer is obviously "no" but the
    model still has to have seen examples to say so confidently)."""
    if d is None:
        return "none"
    for hi in (40, 80, 120, 160, 220, 300, 420):
        if d < hi:
            return f"<{hi}"
    return ">=420"


def load_rows(path: Path, field: str, limit: int | None,
              balance: bool = False, seed: int = 0) -> list[dict]:
    """Load rows, optionally stratified by obstacle class.

    Natural frequencies are badly skewed: cacti are 87% of the data and
    mid-altitude pterodactyls only 4%. Taking the first N rows inherits that
    skew, and the model duly learned cacti well and birds poorly — the one
    surviving crash was a pterodactyl@25 that it jumped at too early.
    Stratifying spends the same compute where the errors actually are.
    """
    rows = [json.loads(l) for l in path.read_text().splitlines()]
    out = [{"text": r[field], "label": r["label"], "speed": r["speed"],
            "kind": r["kind"], "obstacle": obstacle_class(r[field]),
            "dist": r["distance"]}
           for r in rows]

    if not limit:
        return out
    if not balance:
        return out[:limit]

    # Round-robin across (obstacle class, label, distance band) cells.
    #
    # Balancing on (obstacle, label) alone was not enough and actively caused a
    # failure: negatives are dominated by the ~28 frames just before a jump
    # window, so equalising the cells thinned out long-range negatives until
    # the model had barely seen "dist=326 -> no". It then produced erratic,
    # confident nonsense out there and jumped at 326px. Banding by distance
    # forces coverage across the whole range the game actually presents.
    rng = random.Random(seed)
    cells: dict[tuple, list] = {}
    for r in out:
        cells.setdefault(
            (r["obstacle"], r["label"], _dist_band(r["dist"])), []).append(r)
    for v in cells.values():
        rng.shuffle(v)

    picked, keys, i = [], sorted(cells), 0
    while len(picked) < limit and any(cells[k] for k in keys):
        k = keys[i % len(keys)]
        if cells[k]:
            picked.append(cells[k].pop())
        i += 1
    rng.shuffle(picked)
    return picked


def encode_rows(rows, tok, max_len: int):
    """Tokenize once up front — the prompt is short and this keeps the training
    loop free of tokenizer overhead."""
    out = []
    for r in rows:
        ids, markers = build_sequence(tok, r["text"], QUESTION, max_len=max_len)
        if len(markers) != 2:
            continue  # malformed; noul must have exactly [false, true]
        out.append({**r, "ids": ids, "markers": markers})
    return out


def make_batch(items, pad_id: int, device):
    n = max(len(it["ids"]) for it in items)
    input_ids = torch.full((len(items), n), pad_id, dtype=torch.long)
    attn = torch.zeros((len(items), n), dtype=torch.long)
    marker_pos = torch.zeros((len(items), 2), dtype=torch.long)
    for i, it in enumerate(items):
        L = len(it["ids"])
        input_ids[i, :L] = torch.tensor(it["ids"], dtype=torch.long)
        attn[i, :L] = 1
        marker_pos[i] = torch.tensor(it["markers"], dtype=torch.long)
    labels = torch.tensor([it["label"] for it in items], dtype=torch.long)
    marker_mask = torch.ones((len(items), 2), dtype=torch.bool)
    qtype = torch.full((len(items),), QTYPES["noul"], dtype=torch.long)
    return (input_ids.to(device), attn.to(device), marker_pos.to(device),
            marker_mask.to(device), qtype.to(device), labels.to(device))


@torch.no_grad()
def evaluate(model, items, pad_id, device, batch_size, temperature: float = 1.0):
    model.eval()
    probs, labels = [], []
    for i in range(0, len(items), batch_size):
        b = items[i:i + batch_size]
        ids, attn, mpos, mmask, qt, y = make_batch(b, pad_id, device)
        logits, _ = model(ids, attn, mpos, mmask, qt, detach_encoder=True)
        p = torch.softmax(logits.float() / temperature, -1)[:, 1]
        probs.append(p.cpu()); labels.append(y.cpu())
    p = torch.cat(probs).numpy()
    y = torch.cat(labels).numpy()
    pred = (p >= 0.5).astype(int)
    acc = float((pred == y).mean())

    # Confidence in the *predicted* class, which is what ECE is defined over.
    conf = np.where(pred == 1, p, 1.0 - p)
    ece = float(ece_score(conf, (pred == y).astype(int)))

    # A jump missed is fatal; a jump too many is survivable. Track both.
    tp = int(((pred == 1) & (y == 1)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    recall = tp / max(1, tp + fn)
    precision = tp / max(1, tp + fp)
    return {"acc": acc, "ece": ece, "recall": recall, "precision": precision,
            "probs": p, "labels": y}


def fit_temperature(model, items, pad_id, device, batch_size) -> float:
    """One scalar, fitted by NLL on held-out data. laya's README reports this
    cutting ECE from 0.466 to 0.081; we measure our own."""
    model.eval()
    logits_all, labels_all = [], []
    with torch.no_grad():
        for i in range(0, len(items), batch_size):
            b = items[i:i + batch_size]
            ids, attn, mpos, mmask, qt, y = make_batch(b, pad_id, device)
            lg, _ = model(ids, attn, mpos, mmask, qt, detach_encoder=True)
            logits_all.append(lg.float().cpu()); labels_all.append(y.cpu())
    logits = torch.cat(logits_all); labels = torch.cat(labels_all)

    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), labels)
        loss.backward()
        return loss

    opt.step(closure)
    t = float(log_t.exp().item())

    # laya clamps its own shipped temperatures to [0.5, 5] and warns when a
    # checkpoint falls outside that range. An unclamped fit on a small or
    # poorly-separated calibration set runs away (we saw 137), which flattens
    # every probability to 0.5 and destroys the calibration we're measuring.
    lo, hi = 0.5, 5.0
    if not math.isfinite(t) or not (lo <= t <= hi):
        print(f"  [warn] fitted temperature {t:.3f} outside [{lo}, {hi}]; clamping")
        t = min(hi, max(lo, t)) if math.isfinite(t) else 1.0
    return t


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--encoding", default="numeric", choices=["numeric", "bucketed"])
    p.add_argument("--checkpoint", default="convaiinnovations/laya-multilingual")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-len", type=int, default=160)
    p.add_argument("--limit-train", type=int, default=24000)
    p.add_argument("--limit-eval", type=int, default=4000)
    p.add_argument("--out", type=Path, default=Path("checkpoints"))
    p.add_argument("--balance", action="store_true",
                   help="stratify training rows by obstacle class")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)

    print(f"loading {args.checkpoint} ...", flush=True)
    agent = laya.load(args.checkpoint)
    model, tok, device = agent.model, agent.tok, agent.device
    pad_id = tok.pad_token_id
    print(f"device={device}  encoding={args.encoding}", flush=True)

    train = encode_rows(load_rows(args.data / "train.jsonl", args.encoding,
                                  args.limit_train, args.balance, args.seed),
                        tok, args.max_len)
    # Eval stays at natural frequencies — balancing it would flatter the model
    # on exactly the rare cases we oversampled in training.
    held = encode_rows(load_rows(args.data / "eval.jsonl", args.encoding,
                                 args.limit_eval), tok, args.max_len)

    import collections
    mix = collections.Counter(r["obstacle"] for r in train)
    print("train obstacle mix: " + "  ".join(
        f"{k}={v/len(train):.1%}" for k, v in mix.most_common()), flush=True)
    # Split held-out into calibration + test so temperature is never fitted on
    # the numbers we report.
    cut = len(held) // 2
    calib, test = held[:cut], held[cut:]
    print(f"train={len(train)}  calib={len(calib)}  test={len(test)}", flush=True)
    print(f"tokens/row: {int(np.mean([len(t['ids']) for t in train]))} avg, "
          f"{max(len(t['ids']) for t in train)} max", flush=True)

    # Freeze the encoder; train the head, type embedding and scorer.
    for prm in model.encoder.parameters():
        prm.requires_grad_(False)
    trainable = [prm for prm in model.parameters() if prm.requires_grad]
    n_train = sum(p_.numel() for p_ in trainable)
    n_total = sum(p_.numel() for p_ in model.parameters())
    print(f"trainable: {n_train/1e6:.1f}M / {n_total/1e6:.1f}M params "
          f"({n_train/n_total:.1%})", flush=True)

    # Class weights: a missed jump ends the run, an extra jump usually doesn't.
    pos = sum(r["label"] for r in train)
    w = torch.tensor([1.0, max(1.0, (len(train) - pos) / max(1, pos))], device=device)
    print(f"positives={pos}/{len(train)} ({pos/len(train):.1%})  class_weight={w.tolist()}",
          flush=True)

    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(train) / args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.1)

    print(f"\nbaseline (before training):", flush=True)
    base = evaluate(model, test, pad_id, device, args.batch_size)
    print(f"  acc={base['acc']:.3f}  recall={base['recall']:.3f}  ece={base['ece']:.3f}",
          flush=True)

    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        model.encoder.eval()  # keep frozen encoder's dropout/norm deterministic
        random.shuffle(train)
        running = 0.0
        for i in range(0, len(train), args.batch_size):
            b = train[i:i + args.batch_size]
            ids, attn, mpos, mmask, qt, y = make_batch(b, pad_id, device)
            logits, _ = model(ids, attn, mpos, mmask, qt, detach_encoder=True)
            loss = F.cross_entropy(logits.float(), y, weight=w)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step(); sched.step()
            running += loss.item(); step += 1
            if step % 100 == 0:
                rate = step * args.batch_size / (time.time() - t0)
                print(f"  ep{ep} step{step}/{steps} loss={running/100:.4f} "
                      f"({rate:.0f} rows/s)", flush=True)
                running = 0.0
        m = evaluate(model, test, pad_id, device, args.batch_size)
        print(f"epoch {ep}: acc={m['acc']:.3f} recall={m['recall']:.3f} "
              f"precision={m['precision']:.3f} ece={m['ece']:.3f}", flush=True)

    temp = fit_temperature(model, calib, pad_id, device, args.batch_size)
    pre = evaluate(model, test, pad_id, device, args.batch_size)
    post = evaluate(model, test, pad_id, device, args.batch_size, temperature=temp)
    print(f"\nfitted temperature: {temp:.3f}")
    print(f"ECE  before={pre['ece']:.4f}  after={post['ece']:.4f}")
    print(f"acc  before={pre['acc']:.4f}  after={post['acc']:.4f}  "
          f"(temperature cannot change argmax; acc should be identical)")

    args.out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.encoding}"
    torch.save({
        "head": model.head.state_dict(),
        "type_emb": model.type_emb.state_dict(),
        "scorer": model.scorer.state_dict(),
        "act_head": model.act_head.state_dict(),
        "temperature": temp,
        "encoding": args.encoding,
        "checkpoint": args.checkpoint,
        "max_len": args.max_len,
    }, args.out / f"dino_head_{tag}.pt")

    (args.out / f"metrics_{tag}.json").write_text(json.dumps({
        "encoding": args.encoding, "checkpoint": args.checkpoint,
        "baseline": {k: base[k] for k in ("acc", "ece", "recall", "precision")},
        "final": {k: post[k] for k in ("acc", "ece", "recall", "precision")},
        "ece_before_temperature": pre["ece"], "temperature": temp,
        "n_train": len(train), "n_test": len(test),
    }, indent=2))
    print(f"saved -> {args.out / f'dino_head_{tag}.pt'}")


if __name__ == "__main__":
    main()
