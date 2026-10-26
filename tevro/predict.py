"""Load a trained Tevro model and answer one question at a time."""
import json
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from safetensors.torch import load_file

from tevro.inputs import TASK_TYPES, collate_fn, question_to_choices
from tevro.network import TevroNetwork, build_head, load_backbone
from tevro.tokenization import encode_example

DEFAULT_MODEL = "kartikg1/tevro-decider"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve(model):
    """A local folder is used as is; anything else is treated as a Hugging Face repo id."""
    if Path(model).is_dir():
        return Path(model)
    return Path(snapshot_download(model, allow_patterns=["meta.json", "head.safetensors", "lora/*"]))


class Decider:
    """A loaded model. `answer(state, question)` returns the chosen option and a probability for every option."""

    def __init__(self, network, meta, device):
        self.network = network.to(device).eval()
        self.meta = meta
        self.device = torch.device(device)

    @torch.no_grad()
    def answer(self, state, question):
        """state: text or any JSON-able object. question: {"type": "choice"|"noul"|"score", "instructions": ...,
        "criteria": {key: description} for choice, a list of ordered levels for score, optional for noul}."""
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        choices, _ = question_to_choices(question)
        example = encode_example(state, question["instructions"], choices, question["type"],
                                 self.meta["max_state_tokens"], self.meta["max_choice_tokens"],
                                 self.meta.get("use_answer_prompt", True))
        example.update(task_type=TASK_TYPES[question["type"]], label=None)
        batch = {k: v.to(self.device) for k, v in collate_fn([example]).items()}

        autocast = torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.device.type in ("cuda", "mps"))
        with autocast:
            logits = self.network(**batch)[0].float().cpu()
        calibration = self.meta.get("calibration") or {}
        logits = logits / calibration.get("temperatures", {}).get(question["type"], 1.0)
        probs = torch.softmax(logits, -1)
        confidence = probs.max().item()
        threshold = calibration.get("thresholds", {}).get(question["type"], 0.0)
        out = {"confidence": confidence, "abstain": confidence < threshold}
        probs = probs.tolist()

        if question["type"] == "choice":
            keys = list(question["criteria"])
            out.update(choice=keys[max(range(len(keys)), key=probs.__getitem__)], probabilities=dict(zip(keys, probs)))
        elif question["type"] == "score":
            out.update(score=sum(i * p for i, p in enumerate(probs)), probabilities=probs)
        else:
            out.update(noul=probs[1])
        return out

    def decide(self, state, questions):
        """Answers several questions about one state in JEV's format: {id: question} in, {id: answer} out."""
        answers = {}
        for qid, question in questions.items():
            if question.get("type") not in TASK_TYPES:
                raise ValueError(f"{qid}: unknown type {question.get('type')!r}")
            out = self.answer(state, question)
            if question["type"] == "score":
                out["probabilities"] = {str(i): p for i, p in enumerate(out["probabilities"])}
            answers[qid] = {"type": question["type"], **out}
        return answers


def load(model=DEFAULT_MODEL, device=None, precision="bf16"):
    """Loads the base model, the LoRA adapter and the decision head. `model` is a Hugging Face repo id or a folder.
    precision "bf16" (default) halves the memory of the frozen weights; "fp32" is exact."""
    folder = resolve(model)
    meta = json.loads((folder / "meta.json").read_text())
    device = torch.device(device) if device else get_device()

    backbone = load_backbone(meta["model_name"], meta.get("num_layers"), {"fp32": "float32", "bf16": "bfloat16"}[precision])
    if (folder / "lora").exists():
        backbone = PeftModel.from_pretrained(backbone, folder / "lora")
    head = build_head(meta["head_config"], backbone.config.hidden_size, meta["head_config"]["num_task_types"])
    head.load_state_dict(load_file(str(folder / "head.safetensors"), device="cpu"))
    return Decider(TevroNetwork(backbone, head), meta, device)
