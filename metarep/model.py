"""
transformer and transformer parts for meta-learning
"""
__all__ = ["TransformerConfig", "Transformer", "TransformerBlock", "SelfAttention", "MLP"]

from dataclasses import dataclass
from typing import Optional

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F


@dataclass
class TransformerConfig:
    input_size: int = 2304
    embedding: bool = True
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    num_layers: int = 6
    hidden_act: str = "gelu"
    bias: bool = True
    logit_bias: bool = True
    attention_dropout: float = 0.1
    sequence_length: int = 100
    name: Optional[str] = None
    positional_embedding_type: str = "learned" # Only "learned" is supported

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")

        if self.positional_embedding_type != "learned":
            raise ValueError(f"positional_embedding_type must be 'learned', but got '{self.positional_embedding_type}'")


class MLP(nn.Module):
    def __init__(self, hidden_size:int, intermediate_size:int, hidden_act: str, bias: bool = False):
        super().__init__()
        self.act_fn = getattr(F, hidden_act, None)
        if self.act_fn is None: raise ValueError(f"Activation function '{hidden_act}' is not supported.")

        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, bias: bool = False, attention_dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.qkv_split = Rearrange('batch sequence (three head head_dim) -> three batch head sequence head_dim', three=3, head=num_attention_heads)
        self.o_proj_transpose = Rearrange('batch head sequence head_dim -> batch sequence (head head_dim)')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = self.qkv_split(qkv)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attention_dropout if self.training else 0.0)

        attn_output = self.o_proj_transpose(attn_output)

        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int, hidden_act: str, bias: bool = False, attention_dropout: float = 0.0):
        super().__init__()
        self.self_attn = SelfAttention(hidden_size, num_attention_heads, bias=bias, attention_dropout=attention_dropout)
        self.mlp = MLP(hidden_size, intermediate_size, hidden_act, bias=bias)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(self.ln1(x))
        x = x + attn_output
        x = x + self.mlp(self.ln2(x))
        return x

class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, seq_len: int, hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(seq_len, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[1]
        position_ids = torch.arange(seq_len, device=x.device)
        pos_embeds = self.embedding(position_ids)
        return x + pos_embeds


class Transformer(torch.nn.Module):
    def __init__(self, config: TransformerConfig=TransformerConfig()):
        super().__init__()
        self.config = config

        # actual input size includes features + 2 for target encoding
        actual_input_size = config.input_size + 2

        if not config.embedding and actual_input_size != config.hidden_size: raise ValueError(f"Input size {actual_input_size} must match hidden size {config.hidden_size} when embedding is False.")

        if config.embedding: self.embedding = nn.Linear(actual_input_size, config.hidden_size, bias=config.bias)
        else: self.embedding = nn.Identity()

        self.pos_encoder = LearnedPositionalEmbedding(seq_len=config.sequence_length, hidden_size=config.hidden_size)

        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                bias=config.bias,
                attention_dropout=config.attention_dropout
            ) for _ in range(config.num_layers)
        ])
        self.final_ln = nn.LayerNorm(config.hidden_size)
        self.linear_head = nn.Linear(config.hidden_size,1, bias=config.logit_bias)


    def _prep_inputs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Prepares the input tensor by concatenating the one-hot encoded previous targets with the input features.
        Inputs are the same as the forward method.
        """
        batch_size = x.shape[0]

        # create previous targets: shift targets right and prepend
        prev_targets = torch.cat([torch.zeros(batch_size, 1, device=x.device), y[:, :-1]], dim=1)

        # one-hot encode previous targets
        target_onehot = F.one_hot(prev_targets.long(), num_classes=2).float()

        # replace the first position with 0s, our BOS token is always 0
        target_onehot[:, 0] = 0.

        # concatenate target encoding with input features
        return torch.cat([target_onehot, x], dim=-1)

    def forward(self,
                x: torch.Tensor, # (batch_size, seq_len, feature_dim) - input features
                y: torch.Tensor # (batch_size, seq_len) - binary targets for each position
                ) -> torch.Tensor:

        x = self._prep_inputs(x, y)  # (batch_size, seq_len, input_size)

        x = self.embedding(x)
        x = self.pos_encoder(x)

        for layer in self.layers:
            x = layer(x)
        return self.linear_head(self.final_ln(x))
