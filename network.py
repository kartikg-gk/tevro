import math

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Model


class ChoiceHead(nn.Module):
    """Self-attention over [task token, choice vectors..., answer vector], then a bilinear score per choice.

    No positional encoding: permuting the choices permutes the logits, so any number of choices works.
    """

    def __init__(self, hidden_dim, num_task_types, new_dim=512, num_layers=2, num_heads=8):
        super().__init__()
        self.new_dim = new_dim
        self.task_embedding = nn.Embedding(num_task_types, new_dim)
        self.input_proj = nn.Linear(hidden_dim, new_dim)
        layer = nn.TransformerEncoderLayer(new_dim, num_heads, 4 * new_dim, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.final_norm = nn.LayerNorm(new_dim)
        self.answer_proj = nn.Linear(new_dim, new_dim)
        self.choice_proj = nn.Linear(new_dim, new_dim)

    def forward(self, choice_embeddings, answer_embedding, choice_mask, task_type):
        # choice_embeddings: [B, C, H], answer_embedding: [B, H], choice_mask: [B, C], task_type: [B]
        B = choice_embeddings.shape[0]
        x = torch.cat([
            self.task_embedding(task_type).unsqueeze(1),
            self.input_proj(choice_embeddings),
            self.input_proj(answer_embedding).unsqueeze(1),
        ], dim=1)  # [B, C+2, D]

        always_valid = torch.ones(B, 1, dtype=torch.bool, device=choice_mask.device)
        ignore = ~torch.cat([always_valid, choice_mask, always_valid], dim=1)
        x = self.final_norm(self.layers(x, src_key_padding_mask=ignore))

        answer = self.answer_proj(x[:, -1])    # [B, D]
        choices = self.choice_proj(x[:, 1:-1])  # [B, C, D]
        logits = torch.einsum("bd,bcd->bc", answer, choices) / math.sqrt(self.new_dim)
        return logits.masked_fill(~choice_mask, -float("inf"))

    def score(self, hidden, token_mask, choice_read_idx, choice_mask, answer_end_idx, task_type):
        rows = torch.arange(hidden.shape[0], device=hidden.device)
        return self(hidden[rows[:, None], choice_read_idx], hidden[rows, answer_end_idx], choice_mask, task_type)


class PostHead(nn.Module):
    """Decision head over every token: type embedding added to all hidden states, a few bidirectional
    transformer layers over the whole sequence, then a scalar scorer at each choice's read token.

    No answer token is needed: the bidirectional layers let every choice see the state and the other
    choices before scoring. No positional encoding, and the trunk gives choices shared positions,
    so permuting the choices permutes the logits.
    """

    def __init__(self, hidden_dim, num_task_types, num_layers=2, dropout=0.1):
        super().__init__()
        num_heads = max(1, hidden_dim // 64)
        self.task_embedding = nn.Embedding(num_task_types, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, 4 * hidden_dim, dropout,
                                           batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.scorer = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
                                    nn.Linear(hidden_dim, 1))

    def score(self, hidden, token_mask, choice_read_idx, choice_mask, answer_end_idx, task_type):
        h = hidden + self.task_embedding(task_type)[:, None, :]
        h = self.layers(h, src_key_padding_mask=~token_mask)
        rows = torch.arange(h.shape[0], device=h.device)
        logits = self.scorer(h[rows[:, None], choice_read_idx]).squeeze(-1)  # [B, C]
        return logits.masked_fill(~choice_mask, -float("inf"))


class TevroNetwork(nn.Module):
    def __init__(self, backbone, head):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, input_ids, position_ids, attention_mask, token_mask, choice_read_idx, choice_mask,
                answer_end_idx, task_type, **kwargs):
        output = self.backbone(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask,
                               use_cache=False)
        hidden = output.last_hidden_state.float()
        # Head runs in fp32 even under bf16 autocast: bf16 logits are too coarse for a stable loss
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            return self.head.score(hidden, token_mask, choice_read_idx, choice_mask, answer_end_idx, task_type)


def load_backbone(model_name, num_layers=None, dtype="float32"):
    # Only the first num_layers decoder layers are built and loaded (null = all of them).
    # bfloat16 halves the frozen weights for small GPUs; compute already runs in bf16 under autocast
    overrides = {} if num_layers is None else {"num_hidden_layers": num_layers}
    return Qwen3Model.from_pretrained(model_name, dtype=getattr(torch, dtype), **overrides)


def lora_target_regex(num_layers, last_k_layers):
    layer_ids = "|".join(str(i) for i in range(num_layers - last_k_layers, num_layers))
    return rf".*layers\.({layer_ids})\.self_attn\.(q_proj|k_proj|v_proj|o_proj)"


def add_lora(backbone, r, alpha, dropout, last_k_layers):
    if last_k_layers == 0:  # frozen Qwen, only the head trains
        backbone.requires_grad_(False)
        return backbone
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=lora_target_regex(backbone.config.num_hidden_layers, last_k_layers),
    )
    return get_peft_model(backbone, config)


def build_network(model_name, num_layers, lora_config, head_config, num_task_types, backbone=None,
                  dtype="float32"):
    if backbone is None:
        backbone = load_backbone(model_name, num_layers, dtype)
    backbone = add_lora(backbone, lora_config["r"], lora_config["alpha"], lora_config["dropout"],
                        lora_config["last_k_layers"])
    head = build_head(head_config, backbone.config.hidden_size, num_task_types)
    return TevroNetwork(backbone, head)


HEADS = {"choicehead": ChoiceHead, "posthead": PostHead}


def build_head(head_config, hidden_dim, num_task_types):
    """head.type: choicehead = attention over [task, choices, answer token] + bilinear score |
    posthead = bidirectional layers over every token + scalar scorer per choice."""
    head_config = dict(head_config)
    head_type = head_config.pop("type", "choicehead")
    head_config.pop("num_task_types", None)
    return HEADS[head_type](hidden_dim, num_task_types, **head_config)
