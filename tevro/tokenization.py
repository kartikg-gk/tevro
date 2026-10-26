import os

import torch
from transformers import AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # the tokenizer is shared with DataLoader worker processes

model_name = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

OPTION_OPEN = tokenizer.encode("<option>")
OPTION_CLOSE = tokenizer.encode("</option>")
ANSWER_PROMPT = tokenizer.encode("The answer is:")

# Short natural-language task hint
TASK_PROMPTS = {
    "choice": "Choose the best option.",
    "score": "Rate it on the given scale.",
    "noul": "Answer yes or no.",
}


def truncate(text, max_tokens):
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])


def build_question(state, instructions, task_type, max_state_tokens):
    # Task hint and question come before the state, so every state token is read knowing what is asked
    return f"{TASK_PROMPTS[task_type]}\nQuestion: {instructions}\n\n{truncate(state, max_state_tokens)}\n"


def build_layout(question_ids, choice_ids, use_answer_prompt=True):
    """One sequence: question, then every choice wrapped in <option>...</option>, then "The answer is:".

    Returns token ids, position ids, a [L, L] attention mask (1 = may attend) and the read indices.
    Every choice starts at the same position and cannot see the other choices, so the choices are
    read in parallel: reordering them only reorders the outputs. The answer tokens come after the
    longest choice and see everything, which is where the choices are compared.
    """
    q = len(question_ids)
    input_ids = list(question_ids)
    position_ids = list(range(q))
    spans = []
    for ids in choice_ids:
        wrapped = OPTION_OPEN + ids + OPTION_CLOSE
        start = len(input_ids)
        input_ids += wrapped
        position_ids += range(q, q + len(wrapped))
        spans.append((start, len(input_ids)))  # [start, end)

    answer_start = len(input_ids)
    answer_pos = max(position_ids) + 1
    answer_ids = ANSWER_PROMPT if use_answer_prompt else []
    input_ids += answer_ids
    position_ids += range(answer_pos, answer_pos + len(answer_ids))
    L = len(input_ids)

    mask = torch.zeros((L, L))
    mask[:q, :q] = torch.tril(torch.ones(q, q))           # question: causal
    for start, end in spans:
        n = end - start
        mask[start:end, :q] = 1                            # each choice reads the question and state
        mask[start:end, start:end] = torch.tril(torch.ones(n, n))  # and itself, causally
    n = L - answer_start
    mask[answer_start:, :answer_start] = 1                 # answer reads everything before it
    mask[answer_start:, answer_start:] = torch.tril(torch.ones(n, n))

    # Choices are read at their last content token, not at the closing tag
    choice_read_idx = [end - len(OPTION_CLOSE) - 1 for _, end in spans]
    return {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "attention_mask": mask,
        "choice_read_idx": choice_read_idx,
        "answer_end_idx": L - 1,
    }


def encode_example(state, instructions, choices, task_type, max_state_tokens, max_choice_tokens,
                   use_answer_prompt=True):
    question = build_question(state, instructions, task_type, max_state_tokens)
    choices = [truncate(choice, max_choice_tokens) for choice in choices]
    return build_layout(tokenizer.encode(question), [tokenizer.encode(choice) for choice in choices],
                        use_answer_prompt)
