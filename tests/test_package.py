import json
import sys
from pathlib import Path

import pytest
import torch
from transformers import Qwen3Config, Qwen3Model

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import dataset  # noqa: E402
import tevro  # noqa: E402
from tevro import inputs  # noqa: E402
from tevro.network import build_network  # noqa: E402
from tevro.predict import Decider  # noqa: E402
from tevro.tokenization import tokenizer, encode_example  # noqa: E402


@pytest.mark.parametrize("name", ["network.py", "tokenization.py"])
def test_package_copies_match_the_training_modules(name):
    """The pip package ships verbatim copies of these two modules; after editing one, copy it to tevro/."""
    assert (ROOT / name).read_text(encoding="utf-8") == (ROOT / "tevro" / name).read_text(encoding="utf-8")


def test_inputs_match_the_dataset_module():
    questions = [
        {"type": "choice", "instructions": "q", "criteria": {"a": "x", "b": "", "c": "z"}, "label": "b"},
        {"type": "noul", "instructions": "q", "label": True},
        {"type": "score", "instructions": "q", "criteria": ["low", "mid", "high"], "label": 2},
    ]
    batch, batch_ref = [], []
    for q in questions:
        assert inputs.question_to_choices(q) == dataset.question_to_choices(q)
        choices, label = inputs.question_to_choices(q)
        ex = encode_example("A state.", q["instructions"], choices, q["type"], 64, 16)
        ex.update(task_type=inputs.TASK_TYPES[q["type"]], label=label)
        batch.append(ex)
    out, ref = inputs.collate_fn(batch), dataset.collate_fn(batch)
    assert out.keys() == ref.keys()
    for k in out:
        assert torch.equal(out[k], ref[k]), k


def make_decider(head="posthead"):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16)
    head_cfg = {"type": "posthead", "num_layers": 2, "dropout": 0.1}
    network = build_network(None, None, {"r": 4, "alpha": 8, "dropout": 0.0, "last_k_layers": 2}, head_cfg, 3,
                            backbone=Qwen3Model(cfg))
    meta = {"max_state_tokens": 128, "max_choice_tokens": 16, "use_answer_prompt": True, "calibration": None}
    return Decider(network, meta, "cpu")


def test_answer_shapes_and_option_order_invariance():
    model = make_decider()
    criteria = {"a": "billing", "b": "technical", "c": "sales"}
    q = {"type": "choice", "instructions": "Which team?", "criteria": criteria}
    out = model.answer("Customer was charged twice.", q)
    assert set(out["probabilities"]) == {"a", "b", "c"} and abs(sum(out["probabilities"].values()) - 1) < 1e-5
    assert out["choice"] in criteria and 0 < out["confidence"] <= 1 and out["abstain"] is False
    shuffled = model.answer("Customer was charged twice.", {**q, "criteria": {k: criteria[k] for k in ["c", "a", "b"]}})
    for k, p in out["probabilities"].items():
        assert abs(p - shuffled["probabilities"][k]) < 1e-4


def test_yes_no_score_and_structured_state():
    model = make_decider()
    yn = model.answer({"case": 12, "note": "no receipt"}, {"type": "noul", "instructions": "Refund?"})
    assert 0 <= yn["noul"] <= 1
    sc = model.answer("It crashed.", {"type": "score", "instructions": "Severity?", "criteria": ["a", "b", "c", "d"]})
    assert 0 <= sc["score"] <= 3 and len(sc["probabilities"]) == 4


def test_lazy_exports_and_cli_help(capsys):
    assert tevro.__version__ == "0.1.0" and callable(tevro.load) and tevro.DEFAULT_MODEL == "kartikg1/tevro-decider"
    with pytest.raises(AttributeError):
        tevro.nope
    from tevro.cli import main
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0 and "predict" in capsys.readouterr().out
