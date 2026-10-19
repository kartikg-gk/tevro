import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reasoning_data import generate, payout  # noqa: E402


def questions(rows):
    return [json.loads(r["questions_json"])["q"] for r in rows]


def test_deterministic():
    assert generate(50, seed=3) == generate(50, seed=3)


def test_labels_are_balanced():
    qs = questions(generate(3000, seed=1))
    yes = [q["label"] for q in qs if q["type"] == "noul"]
    assert 0.4 < sum(yes) / len(yes) < 0.6
    levels = Counter(q["label"] for q in qs if q["type"] == "score")
    assert set(levels) == {0, 1, 2, 3}
    positions = Counter(list(q["criteria"]).index(q["label"]) for q in qs if q["type"] == "choice")
    assert len(positions) == 4 and max(positions.values()) / sum(positions.values()) < 0.35


def test_payout_matches_terms():
    rng = random.Random(7)
    for _ in range(200):
        state, _, q = payout(rng)
        nums = [int(x.replace(",", "")) for x in re.findall(r"\$([\d,]+)", state)]
        deductible, limit, amount = nums[0], nums[1], nums[2]
        if q["label"] == "pay_capped":
            assert q["criteria"]["pay_capped"] == f"Pay ${max(0, min(amount - deductible, limit)):,}"
        assert len(set(q["criteria"].values())) == len(q["criteria"])  # no duplicate option texts
