"""Load a trained Tevro model and answer one question at a time."""
import json
from pathlib import Path

import torch
import transformers
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
    """A loaded model. `decide(state, questions)` answers several questions; `answer(state, question)` answers one
    and also returns a confidence."""

    def __init__(self, network, meta, device, precision="fp32"):
        self.network = network.to(device).eval()
        self.meta = meta
        self.device = torch.device(device)
        self.precision = precision

    @torch.no_grad()
    def probabilities(self, state, questions):
        """One probability list per question, all questions in one batch."""
        if not isinstance(state, str):
            state = json.dumps(state, ensure_ascii=False)
        examples = []
        for question in questions:
            if question.get("type") not in TASK_TYPES:
                raise ValueError(f"unknown type {question.get('type')!r}")
            choices, _ = question_to_choices(question)
            example = encode_example(state, question["instructions"], choices, question["type"],
                                     self.meta["max_state_tokens"], self.meta["max_choice_tokens"],
                                     self.meta.get("use_answer_prompt", True))
            example.update(task_type=TASK_TYPES[question["type"]], label=None)
            examples.append(example)
        batch = {k: v.to(self.device) for k, v in collate_fn(examples).items()}

        # bf16 autocast on GPU / Apple Silicon (as in training); the CPU always runs fp32
        use_bf16 = self.precision == "bf16" and self.device.type in ("cuda", "mps")
        autocast = torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_bf16)
        with autocast:
            logits = self.network(**batch).float().cpu()
        calibration = self.meta.get("calibration") or {}
        out = []
        for row, question, example in zip(logits, questions, examples):
            row = row[:len(example["choice_read_idx"])] / calibration.get("temperatures", {}).get(question["type"], 1.0)
            out.append(torch.softmax(row, -1).tolist())
        return out

    def answer(self, state, question):
        """state: text or any JSON-able object. question: {"type": "choice"|"noul"|"score", "instructions": ...,
        "criteria": {key: description} for choice, a list of ordered levels for score, optional for noul}."""
        probs = self.probabilities(state, [question])[0]
        confidence = max(probs)
        threshold = ((self.meta.get("calibration") or {}).get("thresholds") or {}).get(question["type"], 0.0)
        out = {"confidence": confidence, "abstain": confidence < threshold}
        if question["type"] == "choice":
            keys = list(question["criteria"])
            out.update(choice=keys[max(range(len(keys)), key=probs.__getitem__)], probabilities=dict(zip(keys, probs)))
        elif question["type"] == "score":
            out.update(score=sum(i * p for i, p in enumerate(probs)), probabilities=probs)
        else:
            out.update(noul=probs[1])
        return out

    def decide(self, state, questions, batch_size=16):
        """Answers several questions about one state in JEV's format: {id: question} in, {id: answer} out."""
        ids = list(questions)
        answers = {}
        for start in range(0, len(ids), batch_size):
            chunk = ids[start:start + batch_size]
            for qid, p in zip(chunk, self.probabilities(state, [questions[q] for q in chunk])):
                kind = questions[qid]["type"]
                if kind == "noul":
                    answers[qid] = {"type": "noul", "noul": p[1]}
                elif kind == "choice":
                    keys = list(questions[qid]["criteria"])
                    best = max(range(len(p)), key=p.__getitem__)
                    answers[qid] = {"type": "choice", "choice": keys[best], "probabilities": dict(zip(keys, p))}
                else:
                    answers[qid] = {"type": "score", "score": sum(i * x for i, x in enumerate(p)),
                                    "probabilities": {str(i): x for i, x in enumerate(p)}}
        return answers


def load(model=DEFAULT_MODEL, device=None, precision="bf16"):
    """Loads the base model, the LoRA adapter and the decision head. `model` is a Hugging Face repo id or a folder.
    precision "bf16" (default) runs under bf16 autocast on CUDA / Apple Silicon, which is faster but adds rounding noise
    of about 0.001; "fp32" is exact. Weights stay fp32 and the CPU always runs fp32."""
    if precision not in ("bf16", "fp32"):
        raise ValueError(f"precision must be 'bf16' or 'fp32', not {precision!r}")
    transformers.logging.set_verbosity_error()  # hides the load report for the unused upper layers
    folder = resolve(model)
    meta = json.loads((folder / "meta.json").read_text())
    device = torch.device(device) if device else get_device()

    backbone = load_backbone(meta["model_name"], meta.get("num_layers"), "float32")
    if (folder / "lora").exists():
        # Merged into the base weights: same result, no adapter overhead per forward pass
        backbone = PeftModel.from_pretrained(backbone, folder / "lora").merge_and_unload()
    head = build_head(meta["head_config"], backbone.config.hidden_size, meta["head_config"]["num_task_types"])
    head.load_state_dict(load_file(str(folder / "head.safetensors"), device="cpu"))
    return Decider(TevroNetwork(backbone, head), meta, device, precision)
