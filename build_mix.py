"""Build the 20K fine-tuning mix: rule-reasoning questions, MNLI and BoolQ in the question format, and replayed
questions from the main training set. Writes train.parquet and test.parquet (the main test split, unchanged).

    python build_mix.py out_dir
"""
import json
import random
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from huggingface_hub import hf_hub_download

from reasoning_data import generate

MAIN = "kartikg1/tevro-decision-v1"
SIZES = {"rules": 8000, "mnli": 1000, "boolq": 1000, "replay": 10000}
NLI_OPTIONS = {
    "entailment": "The text shows the statement is true",
    "neutral": "The text neither confirms nor rules out the statement",
    "contradiction": "The text shows the statement is false",
}


def row(state, q, domain):
    return {"state": state, "questions_json": json.dumps({"q": q}), "domain": domain,
            "question_types": [q["type"]], "question_count": 1}


def mnli_rows(n, seed):
    ds = load_dataset("nyu-mll/glue", "mnli", split="train").shuffle(seed=seed).select(range(n))
    names = ["entailment", "neutral", "contradiction"]
    out = []
    for r in ds:
        q = {"type": "choice", "instructions": f'Statement: "{r["hypothesis"]}" What does the text say about this statement?',
             "criteria": NLI_OPTIONS, "label": names[r["label"]]}
        out.append(row(r["premise"], q, "nli"))
    return out


def boolq_rows(n, seed):
    ds = load_dataset("google/boolq", split="train").shuffle(seed=seed).select(range(n))
    out = []
    for r in ds:
        question = r["question"].strip()
        question = question[0].upper() + question[1:] + ("" if question.endswith("?") else "?")
        q = {"type": "noul", "instructions": question, "label": bool(r["answer"])}
        out.append(row(r["passage"], q, "reading"))
    return out


def replay_rows(n, seed):
    table = pq.read_table(hf_hub_download(MAIN, "data/train.parquet", repo_type="dataset"))
    idx = random.Random(seed).sample(range(table.num_rows), n)
    return table.take(idx).to_pylist()


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    seed = 0
    rows = (generate(SIZES["rules"], seed) + mnli_rows(SIZES["mnli"], seed) + boolq_rows(SIZES["boolq"], seed)
            + replay_rows(SIZES["replay"], seed))
    random.Random(seed).shuffle(rows)
    schema = pa.schema([("state", pa.large_string()), ("questions_json", pa.large_string()), ("domain", pa.string()),
                        ("question_types", pa.list_(pa.string())), ("question_count", pa.int16())])
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out / "train.parquet")
    test = pq.read_table(hf_hub_download(MAIN, "data/test.parquet", repo_type="dataset"))
    pq.write_table(test, out / "test.parquet")
    print(f"train rows {len(rows)} ({SIZES}), test rows {test.num_rows} -> {out}")


if __name__ == "__main__":
    main()
