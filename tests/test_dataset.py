"""Integrity checks for the generated training data.

If labels are wrong, nothing downstream reports an error — training "succeeds",
accuracy looks fine, and the dino just dies. These tests re-derive labels from
the simulation independently of the generator that wrote them.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from laya_dino import state
from laya_dino.game import Game
from laya_dino.oracle import best_action, must_jump

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("data")
    subprocess.run(
        [sys.executable, "scripts/gen_dataset.py", "--train-seeds", "3",
         "--eval-seeds", "2", "--max-frames", "2500", "--out", str(out)],
        cwd=ROOT, check=True, capture_output=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
    )
    rows = [json.loads(l) for l in (out / "train.jsonl").read_text().splitlines()]
    return rows


def test_dataset_is_balanced(dataset):
    pos = sum(r["label"] for r in dataset)
    frac = pos / len(dataset)
    # A degenerate split teaches "always say no", which scores well and dies.
    assert 0.3 < frac < 0.7, f"positives are {frac:.1%} of the data"


def test_on_trajectory_labels_match_the_oracle(dataset):
    """Replay each seed and confirm the stored label is what the oracle says.

    Only on-trajectory rows can be checked this way — window interiors are
    reached by deliberately *not* following the oracle, so seed+frame doesn't
    locate them.
    """
    on_traj = [r for r in dataset if not r["kind"].startswith("window_interior")]
    assert on_traj, "no on-trajectory rows to verify"

    by_seed: dict[int, dict[int, int]] = {}
    for r in on_traj:
        by_seed.setdefault(r["seed"], {})[r["frame"]] = r["label"]

    checked = 0
    for seed, frames in by_seed.items():
        g = Game(seed)
        target = max(frames)
        while g.frame <= target:
            if not g.jumping and g.frame in frames:
                assert must_jump(g) == bool(frames[g.frame]), (
                    f"seed {seed} frame {g.frame}: stored label "
                    f"{frames[g.frame]} disagrees with the oracle"
                )
                checked += 1
            if not g.tick(best_action(g)):
                break
    assert checked > 50, f"only verified {checked} rows"


def test_window_interiors_are_all_positive(dataset):
    interiors = [r for r in dataset if r["kind"].startswith("window_interior")]
    assert interiors, "no off-trajectory rows were generated"
    assert all(r["label"] == 1 for r in interiors)


def test_both_encodings_present_and_differ(dataset):
    r = dataset[0]
    assert r["numeric"] and r["bucketed"]
    assert r["numeric"] != r["bucketed"]


def test_no_label_leakage_in_the_prompt(dataset):
    """The state text must never contain the answer. A stray 'jump' or
    'window' field would make the benchmark meaningless."""
    for r in dataset[:2000]:
        for field in ("numeric", "bucketed"):
            text = r[field].lower()
            assert "jump" not in text, f"leak in {field}: {r[field]}"
            assert "window" not in text, f"leak in {field}: {r[field]}"
            assert "label" not in text


def test_train_and_eval_seeds_are_disjoint(tmp_path_factory):
    out = tmp_path_factory.mktemp("split")
    subprocess.run(
        [sys.executable, "scripts/gen_dataset.py", "--train-seeds", "2",
         "--eval-seeds", "2", "--max-frames", "1200", "--out", str(out)],
        cwd=ROOT, check=True, capture_output=True,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"},
    )
    tr = {json.loads(l)["seed"] for l in (out / "train.jsonl").read_text().splitlines()}
    ev = {json.loads(l)["seed"] for l in (out / "eval.jsonl").read_text().splitlines()}
    assert tr and ev
    assert not (tr & ev), f"seed overlap between train and eval: {tr & ev}"


def test_encoding_is_short_enough_to_stay_fast(dataset):
    """Latency scales with token count; the measured 29.6ms assumes a short
    state. Guard against the encoding quietly growing."""
    worst = max(len(r["numeric"]) for r in dataset)
    assert worst < 140, f"longest numeric state is {worst} chars"
