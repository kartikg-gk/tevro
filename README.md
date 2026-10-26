# Tevro

Tevro is a small decision model, and this repository holds its training code and a pip package for running it. It reads a state, a question and a fixed set of options, and returns one probability per option in a single forward pass. It handles three question types: multiple choice, yes/no, and ordinal scores.

![Tevro architecture](https://raw.githubusercontent.com/kartikg-gk/tevro/main/arch.png)

## Use the model

The trained model is on Hugging Face: **[kartikg1/tevro-decider](https://huggingface.co/kartikg1/tevro-decider)**. It needs Python 3.12 and about 3 GB of free memory; a GPU is optional.

```bash
pip install tevro
tevro predict request.json --device cpu
```

`tevro predict` reads one request from a JSON file (examples below) and prints the answer. The weights and the base model are downloaded on the first run. Use `--device cuda` on an NVIDIA GPU. From Python:

```python
import tevro

model = tevro.load()                      # downloads kartikg1/tevro-decider on first use
model.answer("Parcel A is on shelf B2. Parcel B is on shelf C3.",
             {"type": "choice", "instructions": "Which shelf holds Parcel A?",
              "criteria": {"a": "B2", "b": "C3"}})
```

The pip package only runs the model. Training code, data tools and evaluation scripts live in this repository.

**Multiple choice**

```json
{"state": "Parcel A is on shelf B2. Parcel B is on shelf C3.",
 "question": {"type": "choice", "instructions": "Which shelf holds Parcel A?",
              "criteria": {"a": "B2", "b": "C3"}}}
```
```json
{"confidence": 0.99996, "abstain": false, "choice": "a",
 "probabilities": {"a": 0.99996, "b": 0.00004}}
```

**Yes/no**

```json
{"state": "Refunds need a receipt and a purchase within 30 days. The customer bought it 12 days ago and has no receipt.",
 "question": {"type": "noul", "instructions": "Should the refund be issued?"}}
```
```json
{"confidence": 0.933, "abstain": false, "noul": 0.067}
```

`noul` is the probability of yes.

**Score**

```json
{"state": "The app crashes for every user when they open the payments page. No data is lost.",
 "question": {"type": "score", "instructions": "How severe is this bug?",
              "criteria": ["Cosmetic only", "Minor, with a workaround", "A core feature is blocked", "Data loss or harm"]}}
```
```json
{"confidence": 0.431, "abstain": false, "score": 1.65,
 "probabilities": [0.019, 0.431, 0.428, 0.122]}
```

`score` is the expected level (0 to 3 here). A low `confidence` like this one means the model is unsure; send those cases to a person.

### Server

`tevro serve` loads the model once and answers HTTP requests in JEV's `/v1/systemone` format, so apps written for that API can use Tevro locally.

```bash
pip install "tevro[serve]"
tevro serve --port 8008 --device cpu
curl -s localhost:8008/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Order 1182 arrived with a cracked screen.",
  "questions": {
    "damaged": {"type": "noul", "instructions": "Was the item damaged on arrival?"},
    "next": {"type": "choice", "instructions": "What should support do?",
             "criteria": {"replace": "Send a replacement", "refund": "Refund the order"}}
  }
}'
```

`POST /v1/systemone` takes `{"state", "questions", "model"?}`, where `questions` maps your own ids to questions, and returns `{"model", "answers", "latency_ms"}`. Each answer carries its `type`; score probabilities are keyed `"0"`, `"1"`, and so on. `GET /v1/models` describes the loaded model. On a 4-core CPU one question takes about 0.25 s once the model is loaded. The server has no authentication and listens on `127.0.0.1` by default.

From Python, `model.decide(state, questions)` returns the same `answers`.

## Results

| Evaluation | Accuracy |
| --- | ---: |
| Separate held-out test split (5,000) | 80.2% |
| — multiple choice / yes-no / score | 80.8% / 88.0% / 65.8% |
| Training dataset's test split (5,000) | 80.9% |
| JevBench public (231) | 57.6% |
| — easy / standard / hard | 97.9% / 66.7% / 34.2% |

On the held-out split, yes/no AUC is 0.953 and expected calibration error is 0.037. At confidence 0.95 or higher the model answers 48.9% of questions and is right on 98.0% of them.

## Set up for training

```bash
git clone https://github.com/kartikg-gk/tevro
cd tevro
uv sync
uv run pytest -q tests
```

Python 3.12, managed with [uv](https://docs.astral.sh/uv/). On Windows, `uv sync` installs the CUDA 13.0 build of PyTorch; on Linux it uses the default PyPI build. The tests use a tiny randomly initialised model and run on CPU.

## Train

```bash
uv run python train.py configs/smoke.yaml
uv run python train.py configs/check_20k.yaml
uv run python train.py configs/release_12k.yaml
```

| Config | What it does | Hardware |
| --- | --- | --- |
| `smoke.yaml` | Overfits 10 questions; train accuracy should reach 1.0 | any GPU |
| `check_20k.yaml` | 20,000 questions, 1,000 test, bf16 weights, micro-batch 2 × 32 | 6 GB laptop GPU, ~45 min |
| `release_12k.yaml` | The released model: 12,000 updates of 64 questions, fp32 | A100 80 GB, ~6.5 h |
| `finetune_rules.yaml`, `finetune_v2.yaml` | Short fine-tunes of the released model on a 20K rule-reasoning mix (`--resume` the release checkpoint); ~10 min | A100 80 GB |

Each run writes `runs/<name>-<timestamp>/` with the config, `metrics.jsonl`, `summary.json` and checkpoints (`best`, `latest`, `final`; LoRA adapter and head only). Every training log line shows `eta_min`, the projected minutes left. `--resume <run or checkpoint>` reloads the weights and starts a fresh optimizer.

### On Modal

```bash
modal run --detach modal_app.py --config configs/release_12k.yaml
modal run modal_app.py::evaluate --checkpoint runs/<run>/checkpoints/final --out eval.txt
```

`modal_app.py` uploads this folder, keeps `runs/` and the Hugging Face cache on Modal Volumes, runs on an A100 80 GB (set `TEVRO_GPU` to change it), and caps a training job at 9 hours.

## Evaluate

```bash
uv run python inference.py runs/<run>/checkpoints/final --max_questions 5000
uv run python jevbench_eval.py runs/<run>/checkpoints/final <jevbench>/datasets/public
```

The first scores a checkpoint on the training dataset's test split: accuracy per question type, yes/no AUC, ECE, Brier score, reliability and resolution, and accuracy at confidence 0.9 and 0.95. The second scores the 231 public [JevBench](https://github.com/fstandhartinger/jevbench) items by tier.

## Data

Each row is one state with one or more questions:

```json
{
  "state": "Parcel A is on shelf B2. Parcel B is on shelf C3.",
  "questions_json": "{\"q1\": {\"type\": \"choice\", \"instructions\": \"Which shelf holds Parcel A?\", \"criteria\": {\"a\": \"B2\", \"b\": \"C3\"}, \"label\": \"a\"}}"
}
```

- `choice`: `criteria` maps option keys to descriptions; `label` is a key.
- `noul` (yes/no): `label` is `true` or `false`.
- `score`: `criteria` is an ordered list of levels; `label` is the level index.

By default the code loads [kartikg1/tevro-decision-v1](https://huggingface.co/datasets/kartikg1/tevro-decision-v1) (250,000 training and 5,000 test questions; private). Point `data.question_files` at your own parquet files (`train` and `test`) to train on your data. `audit.py` reports label-position, length and negation shortcuts before you spend GPU time.

## Model

| Part | Setting |
| --- | --- |
| Base | Qwen3-0.6B, first 20 of 28 layers |
| Adapter | LoRA rank 8, alpha 16, on q/k/v/o of layers 6–19 |
| Input | task hint + question + state, then each option as its own span starting at the same position, then `The answer is:` |
| Head | question-type embedding on every token, 2 bidirectional transformer layers (1024-d, 16 heads, dropout 0.1), scalar scorer at each option's last token |
| Loss | cross-entropy over the options |
| Optimizer | AdamW, LoRA lr 1e-4, head lr 1e-3, weight decay 0.01, cosine schedule, 5% warmup, gradient clip 1.0 |

Options cannot attend to each other, so reordering them only reorders the probabilities. The test suite checks this.

## Repository layout

```
tevro/              the pip package: copies of network.py and tokenization.py, plus predict.py, serve.py and cli.py
tokenization.py     input layout: positions, attention mask, read indices
network.py          base model, LoRA, decision head
dataset.py          loading, question expansion, batching
train.py            training loop
inference.py        evaluation metrics and single answers
calibrate.py        temperature fitting, abstention thresholds, calibration metrics
jevbench_eval.py    JevBench public evaluation
audit.py            data shortcut checks
synthetic.py        procedural decision questions
data_variants.py    paired question variations
reasoning_data.py   procedural rule-reasoning questions (policies, payouts, counting)
build_mix.py        builds a 20K fine-tuning mix from generated, public and replayed questions
llm_items.py        writes hard items with an LLM through OpenRouter and keeps those a blind second pass agrees with (needs `pydantic` and `OPENROUTER_API_KEY`)
modal_app.py        Modal training and evaluation
configs/            run settings
tests/              layout, invariance, loss and data tests
```

## License

MIT. The base model, Qwen3-0.6B, is released under its own license.
