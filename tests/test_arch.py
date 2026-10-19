import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calibrate import fit_temperature, fit_thresholds  # noqa: E402
from dataset import TASK_TYPES, collate_fn  # noqa: E402
from inference import metrics_from_records  # noqa: E402
from network import build_network  # noqa: E402
from synthetic import generate_questions  # noqa: E402
from tokenization import OPTION_CLOSE, encode_example, tokenizer  # noqa: E402
from train import rps_loss  # noqa: E402

LORA = {"r": 4, "alpha": 8, "dropout": 0.0, "last_k_layers": 2}
HEADS = {
    "choicehead": {"type": "choicehead", "new_dim": 32, "num_layers": 2, "num_heads": 4},
    "posthead": {"type": "posthead", "num_layers": 2, "dropout": 0.1},
}

EXAMPLES = [
    ("The customer was charged twice for the March invoice.", "Which team handles this?",
     ["billing: payments and refunds", "technical: bugs and outages", "sales: new plans"], "choice", 0),
    ("The server returned 500 errors all night.", "Is this an outage?", ["No", "Yes"], "noul", 1),
    ("Reply was polite but did not solve the issue.", "How helpful was the reply?",
     ["0: not helpful", "1: somewhat helpful", "2: very helpful"], "score", 1),
]


def make_network(head="choicehead"):
    torch.manual_seed(0)
    config = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    network = build_network(None, None, LORA, HEADS[head], len(TASK_TYPES), backbone=Qwen3Model(config))
    for name, param in network.named_parameters():
        if "lora_B" in name:  # LoRA B starts at zero; give it values so the adapters change the output
            torch.nn.init.normal_(param, std=0.02)
    return network


def encode(examples, use_answer_prompt=True):
    batch = []
    for state, question, choices, task, label in examples:
        ex = encode_example(state, question, choices, task, max_state_tokens=128, max_choice_tokens=16,
                            use_answer_prompt=use_answer_prompt)
        ex["task_type"] = TASK_TYPES[task]
        ex["label"] = label
        batch.append(ex)
    return collate_fn(batch)


def test_layout_choices_share_positions_and_cannot_see_each_other():
    state, question, choices, task, _ = EXAMPLES[0]
    ex = encode_example(state, question, choices, task, 128, 16)
    reads = ex["choice_read_idx"]
    mask, pos = ex["attention_mask"], ex["position_ids"]
    starts = [r for r in range(len(pos)) if r > 0 and pos[r] <= pos[r - 1]]  # position resets = choice 2, 3
    assert len(starts) == len(choices) - 1
    assert mask[reads[1], reads[0]] == 0 and mask[reads[0], reads[1]] == 0
    assert mask[ex["answer_end_idx"], reads].all()
    for r, choice in zip(reads, choices):
        assert ex["input_ids"][r + 1:r + 1 + len(OPTION_CLOSE)] == OPTION_CLOSE
        assert tokenizer.decode(ex["input_ids"][r]) in choice


@pytest.mark.parametrize("head", ["choicehead", "posthead"])
def test_shuffled_choices_give_permuted_logits(head):
    network = make_network(head).eval()
    state, question, choices, task, _ = EXAMPLES[0]
    order = [2, 0, 1]
    with torch.no_grad():
        logits = network(**encode([(state, question, choices, task, 0)]))
        shuffled = network(**encode([(state, question, [choices[i] for i in order], task, 0)]))
    assert torch.allclose(logits[0, order], shuffled[0], atol=1e-4)


@pytest.mark.parametrize("head", ["choicehead", "posthead"])
def test_mixed_batch_has_finite_logits_and_masked_padding(head):
    network = make_network(head).eval()
    with torch.no_grad():
        logits = network(**encode(EXAMPLES))
    assert torch.isfinite(logits[0]).all()
    assert torch.isinf(logits[1, 2]) and logits[1, 2] < 0  # yes/no row has only 2 choices


@pytest.mark.parametrize("head", ["choicehead", "posthead"])
def test_overfits_three_questions(head):
    network = make_network(head).train()
    batch = encode(EXAMPLES)
    optimizer = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad], lr=3e-3)
    for _ in range(60):
        logits = network(**batch)
        loss = torch.nn.functional.cross_entropy(logits, batch["labels"])
        loss = loss + rps_loss(logits, batch["labels"], batch["choice_mask"], batch["task_type"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert (network(**batch).argmax(-1) == batch["labels"]).all()


def test_rps_penalises_distant_levels_more():
    labels, mask, types = torch.tensor([0]), torch.ones(1, 3, dtype=torch.bool), torch.tensor([TASK_TYPES["score"]])
    near = rps_loss(torch.tensor([[0.0, 5.0, -5.0]]), labels, mask, types)
    far = rps_loss(torch.tensor([[0.0, -5.0, 5.0]]), labels, mask, types)
    right = rps_loss(torch.tensor([[5.0, -5.0, -5.0]]), labels, mask, types)
    assert right < near < far


def test_temperature_cools_overconfident_logits():
    torch.manual_seed(0)
    logits, labels = [], []
    for i in range(300):  # always 10x too confident, right 70% of the time
        label = 0 if i % 10 < 7 else 1
        logits.append(torch.tensor([10.0, 0.0]))
        labels.append(label)
    assert fit_temperature(logits, labels) > 2.0


def test_thresholds_keep_answered_error_within_target():
    confidences = [0.99] * 50 + [0.6] * 50
    correct = [1] * 50 + [0] * 25 + [1] * 25
    thresholds = fit_thresholds(confidences, correct, [0] * 100, target_error=0.10)
    assert thresholds["choice"] == 0.99


def test_synthetic_negation_does_not_predict_the_answer():
    rows = [r for r in generate_questions(3000, seed=3) if r["task_type"] == TASK_TYPES["noul"]]
    negated = [r["label"] for r in rows if " not " in r["instructions"]]
    plain = [r["label"] for r in rows if " not " not in r["instructions"]]
    assert abs(sum(negated) / len(negated) - sum(plain) / len(plain)) < 0.08


def test_metrics_from_records():
    records = [(0, torch.tensor([2.0, 0.0, -1.0]), 0), (1, torch.tensor([0.0, 1.0]), 0),
               (1, torch.tensor([0.0, 3.0]), 1), (2, torch.tensor([0.0, 2.0, 0.0]), 1)]
    metrics = metrics_from_records(records)
    assert metrics["accuracy"] == 0.75
    assert 0 <= metrics["ece"] <= 1 and "auc_noul" in metrics and "murphy_reliability" in metrics


def test_layout_without_answer_prompt_ends_after_choices():
    state, question, choices, task, _ = EXAMPLES[0]
    ex = encode_example(state, question, choices, task, 128, 16, use_answer_prompt=False)
    assert ex["input_ids"][-len(OPTION_CLOSE):] == OPTION_CLOSE
    assert len(ex["input_ids"]) == len(ex["position_ids"]) == ex["attention_mask"].shape[0]
