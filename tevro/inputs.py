"""Turns question dicts into model inputs. Same logic the training code uses, without the dataset dependencies."""
import torch

from tevro.tokenization import tokenizer

TASK_TYPES = {"choice": 0, "noul": 1, "score": 2}
TASK_NAMES = {v: k for k, v in TASK_TYPES.items()}


def question_to_choices(question):
    """Turns any of the 3 question types into a list of option strings (+ label index if present)."""
    if question["type"] == "choice":
        keys = list(question["criteria"])
        choices = [f"{key}: {question['criteria'][key]}" if question["criteria"][key] else key for key in keys]
        label = keys.index(question["label"]) if "label" in question else None
    elif question["type"] == "score":
        # Choices share position ids, so the level index is written into the text to keep the ordering
        choices = [f"{i}: {level}" for i, level in enumerate(question["criteria"])]
        label = question.get("label")
    else:  # noul -> P(Yes)
        choices = ["No", "Yes"]
        label = int(question["label"]) if "label" in question else None
    return choices, label


def collate_fn(batch):
    B = len(batch)
    L = max(len(ex["input_ids"]) for ex in batch)
    C = max(len(ex["choice_read_idx"]) for ex in batch)

    input_ids = torch.full((B, L), tokenizer.pad_token_id, dtype=torch.long)
    position_ids = torch.zeros((B, L), dtype=torch.long)
    # Padded query rows attend only to themselves, otherwise their softmax row is all -inf -> NaN
    attention_mask = torch.eye(L).repeat(B, 1, 1)
    token_mask = torch.zeros((B, L), dtype=torch.bool)
    choice_read_idx = torch.zeros((B, C), dtype=torch.long)
    choice_mask = torch.zeros((B, C), dtype=torch.bool)
    answer_end_idx = torch.zeros(B, dtype=torch.long)

    for b, ex in enumerate(batch):
        n, c = len(ex["input_ids"]), len(ex["choice_read_idx"])
        token_mask[b, :n] = True
        input_ids[b, :n] = torch.tensor(ex["input_ids"])
        position_ids[b, :n] = torch.tensor(ex["position_ids"])
        attention_mask[b, :n, :n] = ex["attention_mask"]
        choice_read_idx[b, :c] = torch.tensor(ex["choice_read_idx"])
        choice_mask[b, :c] = True
        answer_end_idx[b] = ex["answer_end_idx"]

    attention_mask = torch.where(attention_mask == 0, -float("inf"), 0.0).unsqueeze(1)  # [B, 1, L, L]

    out = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "token_mask": token_mask,
        "choice_read_idx": choice_read_idx,
        "choice_mask": choice_mask,
        "answer_end_idx": answer_end_idx,
        "task_type": torch.tensor([ex["task_type"] for ex in batch], dtype=torch.long),
    }
    if all(ex["label"] is not None for ex in batch):
        out["labels"] = torch.tensor([ex["label"] for ex in batch], dtype=torch.long)
    return out
