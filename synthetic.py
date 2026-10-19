"""Content-free format questions: made-up item codes, shelves and attributes, so the model learns to read
the question format and the state instead of recalling facts.

Negation is decorrelated from the answer: negated and plain wordings appear equally often with every label,
and the gold option sits at a uniformly random position.
"""
import random

from datasets import Dataset

COLORS = ["red", "blue", "green", "black", "white", "orange"]
SIZES = ["small", "medium", "large"]
WEIGHT_LEVELS = ["light (under 5 kg)", "medium (5 to 20 kg)", "heavy (over 20 kg)"]
WEIGHT_RANGES = [(1, 4), (5, 20), (21, 40)]  # one level drawn uniformly, then a weight inside it
TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}


def make_code(rng):
    return f"{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}-{rng.randint(100, 999)}"


def make_shelf(rng):
    return f"{rng.choice('ABCDEFGH')}{rng.randint(1, 9)}"


def weight_level(kg):
    return 0 if kg < 5 else 1 if kg <= 20 else 2


def make_items(rng, n):
    codes, shelves, items = set(), set(), []
    while len(items) < n:
        code, shelf = make_code(rng), make_shelf(rng)
        if code in codes or shelf in shelves:
            continue
        codes.add(code)
        shelves.add(shelf)
        items.append({"code": code, "shelf": shelf, "color": rng.choice(COLORS), "size": rng.choice(SIZES),
                      "fragile": rng.random() < 0.5, "kg": rng.randint(*rng.choice(WEIGHT_RANGES))})
    return items


def describe(rng, item):
    # Fragility is stated both ways, so "not" in the state carries no answer by itself
    fragile = "is fragile" if item["fragile"] else rng.choice(["is not fragile", "is sturdy"])
    return (f"Item {item['code']} is on shelf {item['shelf']}. It is {item['color']} and {item['size']}, "
            f"weighs {item['kg']} kg and {fragile}.")


def place_gold(rng, gold, distractors):
    """Shuffle so the gold option lands at a uniformly random position; returns (options, label)."""
    options = distractors[:]
    label = rng.randint(0, len(options))
    options.insert(label, gold)
    return options, label


def shelf_question(rng, items, target):
    others = [it for it in items if it is not target]
    distractors = [it["shelf"] for it in rng.sample(others, rng.randint(1, min(5, len(others))))]
    options, label = place_gold(rng, target["shelf"], distractors)
    return f"Which shelf holds item {target['code']}?", options, label


def choice_question(rng, items, target):
    if rng.random() < 0.5:
        return shelf_question(rng, items, target)
    # "Which item is fragile?" and "Which item is not fragile?" equally often; exactly one listed option fits
    negated = rng.random() < 0.5
    want_fragile = not negated
    golds = [it for it in items if it["fragile"] == want_fragile]
    wrongs = [it for it in items if it["fragile"] != want_fragile]
    if not golds or not wrongs:
        return shelf_question(rng, items, target)
    gold = rng.choice(golds)["code"]
    distractors = [it["code"] for it in rng.sample(wrongs, rng.randint(1, min(5, len(wrongs))))]
    options, label = place_gold(rng, gold, distractors)
    return f"Which of these items is {'not ' if negated else ''}fragile?", options, label


def noul_question(rng, target):
    negated = rng.random() < 0.5
    answer_yes = rng.random() < 0.5  # label drawn independently of the wording
    claim_true = answer_yes != negated  # the un-negated claim holds exactly when yes XOR negated
    attribute = rng.choice(["fragile", "color", "shelf"])
    if attribute == "fragile":
        claim = "fragile" if target["fragile"] == claim_true else "sturdy"
    elif attribute == "color":
        claim = target["color"] if claim_true else rng.choice([c for c in COLORS if c != target["color"]])
    else:
        shelf = target["shelf"]
        while not claim_true and shelf == target["shelf"]:
            shelf = make_shelf(rng)
        claim = f"on shelf {shelf}"
    instructions = f"Is item {target['code']} {'not ' if negated else ''}{claim}?"
    return instructions, ["No", "Yes"], int(answer_yes)


def score_question(target):
    instructions = f"How heavy is item {target['code']}?"
    options = [f"{i}: {level}" for i, level in enumerate(WEIGHT_LEVELS)]
    return instructions, options, weight_level(target["kg"])


def generate_questions(num_questions, seed=0):
    """Rows in the same exploded format as dataset.load_questions: state, task_type, instructions, choices, label."""
    rng = random.Random(seed)
    out = {"state": [], "task_type": [], "instructions": [], "choices": [], "label": []}
    while len(out["state"]) < num_questions:
        items = make_items(rng, rng.randint(3, 8))
        state = " ".join(describe(rng, it) for it in items)
        target = rng.choice(items)
        task = rng.choice(list(TASK_TYPES))
        if task == "choice":
            instructions, options, label = choice_question(rng, items, target)
        elif task == "noul":
            instructions, options, label = noul_question(rng, target)
        else:
            instructions, options, label = score_question(target)
        out["state"].append(state)
        out["task_type"].append(TASK_TYPES[task])
        out["instructions"].append(instructions)
        out["choices"].append(options)
        out["label"].append(label)
    return Dataset.from_dict(out)
