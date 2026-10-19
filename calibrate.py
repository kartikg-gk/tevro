"""Post-training calibration and confidence metrics.

Temperatures: one per task type, fitted on a held-out calibration split by minimising NLL.
Abstention: per task type, the lowest confidence threshold whose answered items stay within a target error.
"""
import math

import torch
import torch.nn.functional as F

from dataset import TASK_NAMES


def fit_temperature(logits, labels, steps=100):
    """logits: list of 1-D tensors (one per question, any number of choices), labels: list of ints."""
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=0.1, max_iter=steps)

    def closure():
        optimizer.zero_grad()
        t = log_t.exp()
        loss = sum(F.cross_entropy(l.unsqueeze(0) / t, torch.tensor([y])) for l, y in zip(logits, labels))
        loss = loss / len(logits)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.detach().exp().clamp(0.05, 20.0))


def fit_temperatures(records):
    """records: list of (task_type, logits 1-D tensor, label). Returns {task_name: temperature}."""
    temperatures = {}
    for task_type, name in TASK_NAMES.items():
        rows = [(l, y) for t, l, y in records if t == task_type]
        if len(rows) >= 20:  # too few items gives a noisy fit; keep T = 1
            temperatures[name] = fit_temperature([l for l, _ in rows], [y for _, y in rows])
    return temperatures


def apply_temperature(logits, task_type, temperatures):
    return logits / temperatures.get(TASK_NAMES[task_type], 1.0)


def fit_thresholds(confidences, correct, task_types, target_error=0.10, min_answered=10):
    """Per task type: smallest threshold whose answered items (confidence >= threshold) keep error <= target."""
    thresholds = {}
    for task_type, name in TASK_NAMES.items():
        rows = sorted(((c, k) for c, k, t in zip(confidences, correct, task_types) if t == task_type), reverse=True)
        best = 1.0  # answer nothing until some prefix qualifies
        hits = 0
        for i, (conf, hit) in enumerate(rows, 1):
            hits += hit
            end_of_ties = i == len(rows) or rows[i][0] != conf  # a threshold answers every item tied at it
            if end_of_ties and i >= min_answered and 1 - hits / i <= target_error:
                best = conf
        if rows:
            thresholds[name] = best
    return thresholds


def ece(confidences, correct, bins=15):
    n, total = len(confidences), 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confidences[i] for i in idx) / len(idx)
            total += len(idx) / n * abs(acc - conf)
    return total


def murphy(confidences, correct, bins=15):
    """Brier score of the top answer's confidence, split into reliability - resolution + uncertainty."""
    n = len(confidences)
    base = sum(correct) / n
    reliability = resolution = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(confidences) if lo < c <= hi or (b == 0 and c == 0)]
        if idx:
            acc = sum(correct[i] for i in idx) / len(idx)
            conf = sum(confidences[i] for i in idx) / len(idx)
            reliability += len(idx) / n * (conf - acc) ** 2
            resolution += len(idx) / n * (acc - base) ** 2
    return {"reliability": reliability, "resolution": resolution, "uncertainty": base * (1 - base)}


def brier(probs, labels):
    """Multi-class Brier score, averaged over questions."""
    total = 0.0
    for p, y in zip(probs, labels):
        target = torch.zeros_like(p)
        target[y] = 1.0
        total += float(((p - target) ** 2).sum())
    return total / len(probs)


def selective(confidences, correct, threshold):
    answered = [k for c, k in zip(confidences, correct) if c >= threshold]
    coverage = len(answered) / len(confidences)
    accuracy = sum(answered) / len(answered) if answered else math.nan
    return coverage, accuracy
