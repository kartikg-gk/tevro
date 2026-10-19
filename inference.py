import argparse
from collections import defaultdict

import torch
import torch.nn.functional as F

from calibrate import apply_temperature, brier, ece, murphy, selective
from dataset import TASK_NAMES, TASK_TYPES, TevroDataset, collate_fn, load_questions, question_to_choices
from logger import load_checkpoint
from tokenization import encode_example


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast(device):
    # bf16 keeps fp32's exponent range, so no loss scaling is needed (unlike fp16)
    return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type in ("cuda", "mps"))


def to_device(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def binary_auc(scores, labels):
    """Probability that a random positive gets a higher score than a random negative (ties count half)."""
    scores, labels = torch.tensor(scores), torch.tensor(labels, dtype=torch.bool)
    positives, negatives = scores[labels], scores[~labels].sort().values
    below = torch.searchsorted(negatives, positives, right=False)
    below_or_tied = torch.searchsorted(negatives, positives, right=True)
    return (below + below_or_tied).sum().item() / 2 / (len(positives) * len(negatives))


@torch.no_grad()
def collect(network, dataloader, device):
    """Runs the network over a loader -> list of (task_type, logits over that question's own choices, label)."""
    network.eval()
    records = []
    for batch in dataloader:
        batch = to_device(batch, device)
        with autocast(device):
            logits = network(**batch).float()
        for row, task_type, label, mask in zip(logits, batch["task_type"].tolist(), batch["labels"].tolist(),
                                               batch["choice_mask"]):
            records.append((task_type, row[mask].cpu(), label))
    network.train()
    return records


def metrics_from_records(records, temperatures=None, thresholds=(0.9, 0.95)):
    temperatures = temperatures or {}
    total_loss, correct, count = 0.0, defaultdict(int), defaultdict(int)
    probs, labels, confidences, hits = [], [], [], []
    noul_p_yes, noul_labels = [], []

    for task_type, logits, label in records:
        name = TASK_NAMES[task_type]
        logits = apply_temperature(logits, task_type, temperatures)
        p = torch.softmax(logits, -1)
        total_loss += F.cross_entropy(logits.unsqueeze(0), torch.tensor([label])).item()
        hit = int(p.argmax().item() == label)
        count[name] += 1
        correct[name] += hit
        probs.append(p)
        labels.append(label)
        confidences.append(p.max().item())
        hits.append(hit)
        if task_type == TASK_TYPES["noul"]:
            noul_p_yes.append(p[1].item())
            noul_labels.append(label)

    n = len(records)
    metrics = {"loss": total_loss / n, "accuracy": sum(correct.values()) / n}
    for name in count:
        metrics[f"accuracy_{name}"] = correct[name] / count[name]
    if 0 < sum(noul_labels) < len(noul_labels):
        metrics["auc_noul"] = binary_auc(noul_p_yes, noul_labels)
    metrics["ece"] = ece(confidences, hits)
    metrics["brier"] = brier(probs, labels)
    for key, value in murphy(confidences, hits).items():
        metrics[f"murphy_{key}"] = value
    for t in thresholds:
        coverage, accuracy = selective(confidences, hits, t)
        metrics[f"coverage@{t}"] = coverage
        metrics[f"selective_accuracy@{t}"] = accuracy
    return metrics


def evaluate(network, dataloader, device, temperatures=None):
    return metrics_from_records(collect(network, dataloader, device), temperatures)


@torch.no_grad()
def answer(network, meta, state, question, device):
    """Answers one question dict (dataset format) with a typed answer, calibrated, plus an abstain flag."""
    network.eval()
    choices, _ = question_to_choices(question)
    example = encode_example(state, question["instructions"], choices, question["type"],
                             meta["max_state_tokens"], meta["max_choice_tokens"], meta.get("use_answer_prompt", True))
    task_type = TASK_TYPES[question["type"]]
    example.update(task_type=task_type, label=None)
    batch = to_device(collate_fn([example]), device)

    with autocast(device):
        logits = network(**batch)[0].float().cpu()
    calibration = meta.get("calibration", {})
    probs = torch.softmax(apply_temperature(logits, task_type, calibration.get("temperatures", {})), -1)
    confidence = probs.max().item()
    out = {"confidence": confidence,
           "abstain": confidence < calibration.get("thresholds", {}).get(question["type"], 0.0)}
    probs = probs.tolist()

    if question["type"] == "choice":
        keys = list(question["criteria"])
        out.update(choice=keys[max(range(len(keys)), key=probs.__getitem__)], probabilities=dict(zip(keys, probs)))
    elif question["type"] == "score":
        out.update(score=sum(i * p for i, p in enumerate(probs)), probabilities=probs)
    else:
        out.update(noul=probs[1])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="runs/<exp_id>/checkpoints/<name>")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_questions", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = get_device()
    network, meta = load_checkpoint(args.checkpoint, device)
    questions = load_questions(args.split, args.max_questions)
    # Same truncation as during training, read from the checkpoint
    dataset = TevroDataset(questions, meta["max_state_tokens"], meta["max_choice_tokens"],
                            meta.get("use_answer_prompt", True))
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn,
                                         num_workers=args.num_workers)
    records = collect(network, loader, device)
    print("raw:       ", metrics_from_records(records))
    if meta.get("calibration"):
        print("calibrated:", metrics_from_records(records, meta["calibration"]["temperatures"]))


if __name__ == "__main__":
    main()
