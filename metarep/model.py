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

from .rope import RotaryPositionalEmbeddings


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
    positional_embedding_type: str = "learned" # Can be "learned" or "rope"

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")

        if self.positional_embedding_type == "rope":
            head_dim = self.hidden_size // self.num_attention_heads
            if head_dim % 2 != 0:
                raise ValueError(f"head_dim ({head_dim}), calculated as hidden_size / num_attention_heads, must be even for RoPE")
        elif self.positional_embedding_type not in ["learned", "rope"]:
            raise ValueError(f"positional_embedding_type must be 'learned' or 'rope', but got '{self.positional_embedding_type}'")


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
    def __init__(self, hidden_size: int, num_attention_heads: int, rope: Optional[RotaryPositionalEmbeddings], bias: bool = False, attention_dropout: float = 0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout
        self.rope = rope

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.qkv_split = Rearrange('batch sequence (three head head_dim) -> three batch head sequence head_dim', three=3, head=num_attention_heads)
        self.rope_transpose = Rearrange('batch head sequence head_dim -> batch sequence head head_dim')
        self.sdpa_transpose = Rearrange('batch head sequence head_dim -> batch sequence head head_dim')
        self.o_proj_transpose = Rearrange('batch head sequence head_dim -> batch sequence (head head_dim)')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = self.qkv_split(qkv)

        if self.rope:
            q = self.rope_transpose(q)
            k = self.rope_transpose(k)
            q = self.rope(q)
            k = self.rope(k)
            q = self.sdpa_transpose(q)
            k = self.sdpa_transpose(k)
        
        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attention_dropout)

        attn_output = self.o_proj_transpose(attn_output)
        
        return self.o_proj(attn_output)
    

class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int, hidden_act: str, rope: Optional[RotaryPositionalEmbeddings], bias: bool = False, attention_dropout: float = 0.0):
        super().__init__()
        self.self_attn = SelfAttention(hidden_size, num_attention_heads, rope=rope, bias=bias, attention_dropout=attention_dropout)
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
        
        if not config.embedding and config.input_size != config.hidden_size: raise ValueError(f"Input size {config.input_size} must match hidden size {config.hidden_size} when embedding is False.")

        if config.embedding: self.embedding = nn.Linear(config.input_size, config.hidden_size, bias=config.bias)
        else: self.embedding = nn.Identity()


        if config.positional_embedding_type == "learned":
            self.pos_encoder = LearnedPositionalEmbedding(seq_len=config.sequence_length,hidden_size=config.hidden_size)
            self.rope = None
        elif config.positional_embedding_type == "rope":
            self.pos_encoder = None
            self.rope = RotaryPositionalEmbeddings(dim=config.hidden_size // config.num_attention_heads,max_seq_len=config.sequence_length)
        
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                rope=self.rope, # Pass the selected rope object (or None) to the blocks
                bias=config.bias,
                attention_dropout=config.attention_dropout
            ) for _ in range(config.num_layers)
        ])
        self.final_ln = nn.LayerNorm(config.hidden_size)
        self.linear_head = nn.Linear(config.hidden_size,1, bias=config.logit_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)

        if self.pos_encoder: x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x)
        return self.linear_head(self.final_ln(x))