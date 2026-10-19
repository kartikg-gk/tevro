"""Evaluate a checkpoint on the public JevBench set (231 items: easy, original/standard, hard).

Data: https://github.com/fstandhartinger/jevbench (MIT), datasets/public/{easy,original,hard}.jsonl
Scoring follows the public rule: the highest-probability label is the answer; for yes/no that is P(yes) vs P(no).

    python jevbench_eval.py <checkpoint_dir> <jevbench_repo>/datasets/public
"""
import argparse
import json
from pathlib import Path

import torch

from dataset import TASK_TYPES, collate_fn, question_to_choices
from inference import autocast, get_device, to_device
from logger import load_checkpoint
from tokenization import encode_example

TIERS = ["easy", "original", "hard"]


def to_question(item):
    """JevBench item -> the dataset's question dict, with the gold label in the dataset's label format."""
    q = dict(item["question"])
    if q["type"] == "choice":
        q["label"] = item["expected"]
    elif q["type"] == "score":
        q["label"] = int(item["expected"])
    else:
        q["label"] = item["expected"] == "yes"
    return q


@torch.no_grad()
def predict(network, meta, item, device, noul_criteria=False):
    q = to_question(item)
    choices, label = question_to_choices(q)
    criteria = q.get("criteria")
    if noul_criteria and q["type"] == "noul" and isinstance(criteria, dict) and {"true", "false"} <= set(criteria):
        # Spell out what yes and no mean, the way choice options carry their descriptions
        choices = [f"No: {criteria['false']}", f"Yes: {criteria['true']}"]
    state = item["state"] if isinstance(item["state"], str) else json.dumps(item["state"], ensure_ascii=False)
    example = encode_example(state, q["instructions"], choices, q["type"], meta["max_state_tokens"],
                             meta["max_choice_tokens"], meta.get("use_answer_prompt", True))
    example.update(task_type=TASK_TYPES[q["type"]], label=label)
    batch = to_device(collate_fn([example]), device)
    with autocast(device):
        logits = network(**batch)[0].float()
    return int(logits.argmax()) == label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("data_dir", help="JevBench datasets/public folder")
    parser.add_argument("--out", help="optional JSON file for per-item results")
    parser.add_argument("--noul_criteria", action="store_true",
                        help="yes/no options carry the item's true/false criteria text instead of plain No/Yes")
    args = parser.parse_args()

    device = get_device()
    network, meta = load_checkpoint(args.checkpoint, device)
    results, per_item = {}, []
    for tier in TIERS:
        items = [json.loads(line) for line in open(Path(args.data_dir) / f"{tier}.jsonl", encoding="utf-8")]
        correct = [predict(network, meta, item, device, args.noul_criteria) for item in items]
        per_item += [{"id": it["id"], "tier": tier, "type": it["question"]["type"], "correct": c}
                     for it, c in zip(items, correct)]
        results[tier] = (sum(correct), len(correct))
        print(f"{tier:9s} {sum(correct)}/{len(correct)} = {sum(correct) / len(correct):.1%}")

    total = sum(c for c, _ in results.values())
    n = sum(n for _, n in results.values())
    print(f"{'public':9s} {total}/{n} = {total / n:.1%}")
    by_type = {}
    for r in per_item:
        c, k = by_type.get(r["type"], (0, 0))
        by_type[r["type"]] = (c + r["correct"], k + 1)
    for t, (c, k) in by_type.items():
        print(f"  {t:7s} {c}/{k} = {c / k:.1%}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"tiers": results, "total": [total, n], "items": per_item}, f, indent=2)


if __name__ == "__main__":
    main()
