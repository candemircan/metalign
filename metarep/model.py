"""
transformer and transformer parts for meta-learning
"""
__all__ = ["TransformerConfig", "Transformer", "TransformerBlock", "SelfAttention", "MLP"]

from dataclasses import dataclass
from typing import Optional

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F

from .rope import apply_rotary_emb, precompute_freqs_cis


@dataclass
class TransformerConfig:
    input_size: int = 2306
    embedding: bool = True
    hidden_size: int = 768
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    num_layers: int = 6
    hidden_act: str = "gelu"
    bias: bool = False
    logit_bias: bool = True
    attention_dropout: float = 0.1
    sequence_length: int = 100
    name: Optional[str] = None


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

        if hidden_size % num_attention_heads != 0: raise ValueError(f"hidden_size {hidden_size} must be divisible by num_heads {num_attention_heads}")

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = rearrange(qkv, 'batch sequence (three head head_dim) -> three batch head sequence head_dim', three=3, head=self.num_heads)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attention_dropout)

        # From (batch, num_heads, sequence, head_dim) to (batch, sequence, hidden_size)
        attn_output = rearrange(attn_output, 'batch head sequence head_dim -> batch sequence (head head_dim)')
        
        return self.o_proj(attn_output)
    

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int, hidden_act: str, bias: bool = False, attention_dropout: float = 0.0):
        super().__init__()
        self.self_attn = SelfAttention(hidden_size, num_attention_heads, bias=bias, attention_dropout=attention_dropout)
        self.mlp = MLP(hidden_size, intermediate_size, hidden_act, bias=bias)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(self.ln1(x), freqs_cis=freqs_cis)
        x = x + attn_output
        x = x + self.mlp(self.ln2(x))
        return x
    

class Transformer(torch.nn.Module):
    def __init__(self, config: TransformerConfig=TransformerConfig()):
        super().__init__()
        self.config = config

        # rope stuff
        freqs_cis = precompute_freqs_cis(config.hidden_size // config.num_attention_heads, config.sequence_length)        
        self.register_buffer("freqs_cis", freqs_cis)
        
        if not config.embedding and config.input_size != config.hidden_size: raise ValueError(f"Input size {config.input_size} must match hidden size {config.hidden_size} when embedding is False.")

        if config.embedding: self.embedding = nn.Linear(config.input_size, config.hidden_size, bias=config.bias)
        else: self.embedding = nn.Identity()
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        seq_len = x.shape[1]
        freqs_cis = self.freqs_cis[:seq_len]
        for layer in self.layers:
            x = layer(x,freqs_cis)
        return self.linear_head(self.final_ln(x))