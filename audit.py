"""Shortcut audit, CPU only: checks whether labels can be guessed from surface patterns instead of the state.

Each check compares a rate to what chance would give. A large gap means the model can score without reading.
"""
import argparse
import json
import re
from collections import Counter, defaultdict

from dataset import TASK_NAMES, load_questions
from synthetic import generate_questions

NEGATION = re.compile(r"\b(not|no|never|none|neither|nor|without|isn't|doesn't|don't|cannot|can't)\b", re.I)


def position_bias(rows):
    """Gold position vs uniform, per number of choices (choice questions)."""
    by_k = defaultdict(Counter)
    for r in rows:
        by_k[len(r["choices"])][r["label"]] += 1
    report = {}
    for k, counts in sorted(by_k.items()):
        n = sum(counts.values())
        report[k] = {"n": n, "max_position_rate": round(max(counts.values()) / n, 3), "chance": round(1 / k, 3)}
    return report


def length_bias(rows):
    """How often the gold option is the longest / shortest, vs chance."""
    longest = shortest = n = 0
    chance = 0.0
    for r in rows:
        lengths = [len(c) for c in r["choices"]]
        if len(set(lengths)) == 1:
            continue
        n += 1
        # Only a unique longest / shortest option counts, so fixed rubrics with tied lengths don't inflate it
        longest += lengths[r["label"]] == max(lengths) and lengths.count(max(lengths)) == 1
        shortest += lengths[r["label"]] == min(lengths) and lengths.count(min(lengths)) == 1
        chance += 1 / len(lengths)
    if not n:
        return {}
    return {"n": n, "gold_longest_rate": round(longest / n, 3), "gold_shortest_rate": round(shortest / n, 3),
            "chance": round(chance / n, 3)}


def negation_bias(rows):
    """Yes-rate for questions with and without a negation word (yes/no questions)."""
    groups = {True: [], False: []}
    for r in rows:
        groups[bool(NEGATION.search(r["instructions"]))].append(r["label"])
    return {("negated" if neg else "plain"): {"n": len(labels), "yes_rate": round(sum(labels) / len(labels), 3)}
            for neg, labels in groups.items() if labels}


def majority_baseline(rows):
    counts = Counter(r["label"] for r in rows)
    return {"n": len(rows), "majority_rate": round(max(counts.values()) / len(rows), 3)}


def state_leakage(train_rows, test_rows):
    """Share of test questions whose exact state also appears in train."""
    train_states = {r["state"] for r in train_rows}
    leaked = sum(r["state"] in train_states for r in test_rows)
    return {"n": len(test_rows), "leaked_rate": round(leaked / len(test_rows), 3)}


def audit(rows, test_rows=None):
    by_type = defaultdict(list)
    for r in rows:
        by_type[TASK_NAMES[r["task_type"]]].append(r)
    report = {name: {"majority": majority_baseline(rs)} for name, rs in by_type.items()}
    if "choice" in by_type:
        report["choice"]["position"] = position_bias(by_type["choice"])
        report["choice"]["length"] = length_bias(by_type["choice"])
    if "score" in by_type:
        report["score"]["length"] = length_bias(by_type["score"])
    if "noul" in by_type:
        report["noul"]["negation"] = negation_bias(by_type["noul"])
    if test_rows is not None:
        report["leakage"] = state_leakage(rows, test_rows)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_questions", type=int, default=None, help="train questions to audit (default: all)")
    parser.add_argument("--synthetic", type=int, default=0, help="audit this many synthetic questions instead")
    parser.add_argument("--out", default=None, help="optional path for the JSON report")
    args = parser.parse_args()

    if args.synthetic:
        report = audit(list(generate_questions(args.synthetic)))
    else:
        report = audit(list(load_questions("train", args.max_questions)), list(load_questions("test")))
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out:
        with open(args.out, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
