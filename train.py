import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from peft import PeftModel, set_peft_model_state_dict
from safetensors.torch import load_file
from transformers import get_cosine_schedule_with_warmup

from calibrate import apply_temperature, fit_temperatures, fit_thresholds
from dataset import (TASK_TYPES, TevroDataset, collate_fn, load_questions,
                     mix_synthetic)
from inference import autocast, collect, get_device, metrics_from_records, to_device
from logger import RunLogger
from network import build_network


def rps_loss(logits, labels, choice_mask, task_type):
    """Ranked probability score on score questions: predicting 1 when the answer is 4 costs more than 3."""
    is_score = task_type == TASK_TYPES["score"]
    if not is_score.any():
        return logits.new_zeros(())
    probs = torch.softmax(logits[is_score], dim=-1)  # padded slots are -inf -> probability 0
    mask = choice_mask[is_score].float()
    target = F.one_hot(labels[is_score], logits.shape[-1]).float()
    k = mask.sum(-1).clamp(min=2)
    rps = (((probs.cumsum(-1) - target.cumsum(-1)) ** 2) * mask).sum(-1) / (k - 1)
    return rps.mean()


def split_calibration(questions, fraction, max_items, seed):
    """Hold out a slice of the (real) training questions for fitting temperatures and abstention thresholds."""
    n = min(max_items, int(fraction * len(questions)))
    if n == 0:
        return questions, None
    parts = questions.train_test_split(test_size=n, seed=seed)
    return parts["train"], parts["test"]


def make_loader(questions, config, device, shuffle):
    data = config["data"]
    dataset = TevroDataset(questions, data["max_state_tokens"], data["max_choice_tokens"],
                            data.get("use_answer_prompt", True))
    pin_memory = data.get("pin_memory", True) and device.type == "cuda"
    common = dict(collate_fn=collate_fn, num_workers=data["num_workers"],
                  persistent_workers=data["num_workers"] > 0, pin_memory=pin_memory)
    # Eval loaders (shuffle=False) can use a smaller batch, which only lowers peak memory
    batch_size = data["batch_size"] if shuffle else data.get("eval_batch_size", data["batch_size"])
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **common)


def resolve_checkpoint(resume):
    """--resume takes a checkpoint folder, or an experiment id (runs/<id>/checkpoints/final)."""
    path = Path(resume)
    if not (path / "meta.json").exists():
        path = Path("runs") / resume / "checkpoints" / "final"
    if not (path / "meta.json").exists():
        raise FileNotFoundError(f"no checkpoint at {resume} or {path}")
    return path


def load_weights(network, ckpt_dir):
    """Load LoRA + head weights into a trainable network with the same architecture."""
    has_lora = (ckpt_dir / "lora").exists()
    if has_lora != isinstance(network.backbone, PeftModel):
        raise ValueError(f"{ckpt_dir} {'has' if has_lora else 'has no'} LoRA weights, but the config's lora.last_k_layers disagrees")
    if has_lora:
        weights = load_file(ckpt_dir / "lora" / "adapter_model.safetensors")
        result = set_peft_model_state_dict(network.backbone, weights)
        missing = [k for k in result.missing_keys if "lora_" in k]
        if result.unexpected_keys or missing:
            raise ValueError(f"LoRA weights don't match the config: unexpected={result.unexpected_keys[:3]} missing={missing[:3]}")
    network.head.load_state_dict(torch.load(ckpt_dir / "head.pt", map_location="cpu"))


def validate(network, test_loader, device, logger, step, best_accuracy):
    metrics = metrics_from_records(collect(network, test_loader, device))
    logger.log(step, split="test", **metrics)
    if metrics["accuracy"] > best_accuracy:
        best_accuracy = metrics["accuracy"]
        logger.save_checkpoint(network, name="best", step=step)
    return metrics, best_accuracy


def calibrate(network, calib_loader, device, target_error):
    records = collect(network, calib_loader, device)
    temperatures = fit_temperatures(records)
    confidences, correct, task_types = [], [], []
    for task_type, logits, label in records:
        p = torch.softmax(apply_temperature(logits, task_type, temperatures), -1)
        confidences.append(p.max().item())
        correct.append(int(p.argmax().item() == label))
        task_types.append(task_type)
    thresholds = fit_thresholds(confidences, correct, task_types, target_error)
    return {"temperatures": temperatures, "thresholds": thresholds, "target_error": target_error,
            "num_items": len(records)}


def train(config, run_name, resume=None):
    torch.manual_seed(config["seed"])
    device = get_device()
    resume_dir, resume_meta = None, {}
    if resume:
        resume_dir = resolve_checkpoint(resume)
        resume_meta = json.loads((resume_dir / "meta.json").read_text())
        config = {**config, "resumed_from": str(resume_dir)}
    logger = RunLogger(config, run_name, device)
    model, lora, optim, log, data = config["model"], config["lora"], config["optim"], config["logging"], config["data"]
    synthetic, calib = config.get("synthetic", {}), config.get("calibration", {})

    network = build_network(model["name"], model["num_layers"], lora, config["head"], len(TASK_TYPES),
                            dtype=model.get("dtype", "float32"))
    if resume_dir:
        load_weights(network, resume_dir)
        print(f"Loaded LoRA + head weights from {resume_dir} (step {resume_meta['step']})")
    network = network.to(device)
    count = lambda params: sum(p.numel() for p in params if p.requires_grad)
    print(f"trainable params: backbone={count(network.backbone.parameters()):,} head={count(network.head.parameters()):,}")

    # Calibration items come from real questions only; synthetic items are mixed into the rest
    train_questions = load_questions("train", data["max_train_questions"], config["seed"], data.get("question_files"))
    train_questions, calib_questions = split_calibration(
        train_questions, calib.get("fraction", 0.0), calib.get("max_items", 0), config["seed"])
    train_questions = mix_synthetic(train_questions, synthetic.get("num_questions", 0),
                                    synthetic.get("drop_real_fraction", 0.0), config["seed"])
    train_loader = make_loader(train_questions, config, device, shuffle=True)
    test_loader = None
    if data["max_test_questions"] != 0:
        test_loader = make_loader(load_questions("test", data["max_test_questions"], config["seed"], data.get("question_files")),
                                  config, device, shuffle=False)

    lora_params = [p for p in network.backbone.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([
        {"params": lora_params, "lr": float(optim["lr_lora"])},
        {"params": network.head.parameters(), "lr": float(optim["lr_head"])},
    ], weight_decay=float(optim["weight_decay"]))
    # grad_accum_steps > 1 splits each optimizer step over that many loader batches (batch_size is per micro-batch)
    accum = optim.get("grad_accum_steps", 1)
    steps_per_epoch = len(train_loader) // accum
    if steps_per_epoch < 1:
        raise ValueError("Training data cannot fill one gradient accumulation group")
    max_steps = optim.get("max_steps")
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError("optim.max_steps must be a positive integer")
    total_steps = max_steps if max_steps is not None else optim["epochs"] * steps_per_epoch
    epochs_to_run = (total_steps + steps_per_epoch - 1) // steps_per_epoch
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(optim["warmup_ratio"] * total_steps), total_steps)
    rps_weight = float(optim.get("rps_weight", 0.0))

    step = 0
    best_accuracy = -1.0
    if resume_dir and test_loader is not None:
        _, best_accuracy = validate(network, test_loader, device, logger, step, best_accuracy)
    network.train()
    train_start = time.time()
    for epoch in range(epochs_to_run):
        epoch_loss, epoch_correct, epoch_count = 0.0, 0, 0
        optimizer.zero_grad()
        for micro, batch in enumerate(train_loader, 1):
            if micro > (len(train_loader) // accum) * accum:
                break  # drop the incomplete final accumulation group
            batch = to_device(batch, device)
            with autocast(device):
                logits = network(**batch)
            loss = F.cross_entropy(logits, batch["labels"])
            if rps_weight:
                loss = loss + rps_weight * rps_loss(logits, batch["labels"], batch["choice_mask"], batch["task_type"])
            (loss / accum).backward()

            correct = (logits.argmax(-1) == batch["labels"]).sum().item()
            epoch_loss += loss.item() * len(logits)
            epoch_correct += correct
            epoch_count += len(logits)
            if micro % accum:
                continue

            torch.nn.utils.clip_grad_norm_(network.parameters(), optim["grad_clip"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % log["log_every"] == 0:
                lrs = {"lr_head": scheduler.get_last_lr()[1]}
                if lora_params:
                    lrs["lr_lora"] = scheduler.get_last_lr()[0]
                elapsed = time.time() - train_start
                eta_min = elapsed / step * (total_steps - step) / 60  # projected minutes left, to check the budget early
                if device.type == "cuda":
                    lrs["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                logger.log(step, split="train", epoch=epoch, loss=loss.item(), accuracy=correct / len(logits),
                           total_steps=total_steps, eta_min=round(eta_min, 1), **lrs)
            if step % log["save_every"] == 0:
                logger.save_checkpoint(network, name="latest", step=step)
            if test_loader is not None and step % log["eval_every"] == 0:
                test_metrics, best_accuracy = validate(network, test_loader, device, logger, step, best_accuracy)
            if step >= total_steps:
                break

        logger.log(step, split="train_epoch", epoch=epoch, loss=epoch_loss / epoch_count,
                   accuracy=epoch_correct / epoch_count)
        if step >= total_steps:
            break
    assert step == total_steps, f"Expected {total_steps} updates, completed {step}"

    summary = {"final_train_loss": epoch_loss / epoch_count, "final_train_accuracy": epoch_correct / epoch_count}
    calibration = None
    if calib_questions is not None:
        calibration = calibrate(network, make_loader(calib_questions, config, device, shuffle=False), device,
                                float(calib.get("target_error", 0.10)))
        logger.log(step, split="calibration", **{f"T_{k}": v for k, v in calibration["temperatures"].items()},
                   **{f"threshold_{k}": v for k, v in calibration["thresholds"].items()})
        summary["calibration"] = calibration
    if test_loader is not None:
        records = collect(network, test_loader, device)
        if data.get("save_test_records", False):
            with open(logger.run_dir / "test_records.json", "w") as f:
                json.dump([{"task_type": int(t), "logits": scores.tolist(), "label": int(label)}
                           for t, scores, label in records], f)
        summary["test"] = metrics_from_records(records)
        logger.log(step, split="test_final", **summary["test"])
        if calibration:
            summary["test_calibrated"] = metrics_from_records(records, calibration["temperatures"])
            logger.log(step, split="test_calibrated", **summary["test_calibrated"])
        if summary["test"]["accuracy"] > best_accuracy:
            logger.save_checkpoint(network, name="best", step=step, calibration=calibration)
        summary["best_test_accuracy"] = max(best_accuracy, summary["test"]["accuracy"])

    ckpt_dir = logger.save_checkpoint(network, step=step, calibration=calibration)
    logger.finish(checkpoint=str(ckpt_dir), **summary)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="YAML config, e.g. configs/smoke.yaml")
    parser.add_argument("--name", help="run name, used as runs/<name>-<timestamp> (default: config file name)")
    parser.add_argument("--resume", help="start from a checkpoint's LoRA + head weights: an experiment id "
                                         "(uses its final checkpoint) or a checkpoint folder. Optimizer and LR schedule start fresh.")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    train(config, args.name or Path(args.config).stem, args.resume)


if __name__ == "__main__":
    main()
