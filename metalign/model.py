"""
transformer and transformer parts for meta-learning
"""
__all__ = ["TransformerConfig", "Transformer", "TransformerBlock", "SelfAttention", "MLP", "RotaryPositionalEmbeddings"]

from dataclasses import dataclass
from typing import Optional

import torch
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn import functional as F


@dataclass
class TransformerConfig:
    """
    Configuration used to initialize the transformer model

    The defaults here are kinda random. They exist just to make testing easy. 
    """
    input_size: int = 512
    embedding: bool = True
    hidden_size: int = 128
    num_attention_heads: int = 4
    intermediate_size: int = 512
    num_layers: int = 2
    hidden_act: str = "gelu"
    bias: bool = True
    logit_bias: bool = True
    attention_dropout: float = 0.1
    sequence_length: int = 120
    normalize: bool = True

    def __post_init__(self):
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})")

class MLP(nn.Module):
    def __init__(self, hidden_size:int, intermediate_size:int, hidden_act: str, bias: bool):
        super().__init__()
        self.act_fn = getattr(F, hidden_act, None)
        if self.act_fn is None: raise ValueError(f"Activation function '{hidden_act}' is not supported.")

        self.fc1 = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.fc2 = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act_fn(self.fc1(x)))


class SelfAttention(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, bias: bool, attention_dropout: float, rope: nn.Module):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.attention_dropout = attention_dropout
        self.rope = rope

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=bias)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.qkv_split = Rearrange('batch sequence (qkv head head_dim) -> qkv batch head sequence head_dim', qkv=3, head=num_attention_heads)
        self.o_proj_transpose = Rearrange('batch head sequence head_dim -> batch sequence (head head_dim)')


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = self.qkv_split(qkv)
        
        q, k = self.rope(q, k)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attention_dropout if self.training else 0.0)

        attn_output = self.o_proj_transpose(attn_output)

        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_size: int, num_attention_heads: int, intermediate_size: int, hidden_act: str, bias: bool, attention_dropout: float, rope: nn.Module):
        super().__init__()
        self.self_attn = SelfAttention(hidden_size, num_attention_heads, bias=bias, attention_dropout=attention_dropout, rope=rope)
        self.mlp = MLP(hidden_size, intermediate_size, hidden_act, bias=bias)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(self.ln1(x))
        x = x + attn_output
        x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # actual input size includes features + 2 for target encoding
        actual_input_size = config.input_size + 2

        if not config.embedding and actual_input_size != config.hidden_size: raise ValueError(f"Input size {actual_input_size} must match hidden size {config.hidden_size} when embedding is False.")

        if config.normalize: self.norm = nn.LayerNorm(actual_input_size)
        else: self.norm = nn.Identity()

        if config.embedding: self.embedding = nn.Linear(actual_input_size, config.hidden_size, bias=config.bias)
        else: self.embedding = nn.Identity()

        self.rope = RotaryPositionalEmbeddings(dim=config.hidden_size // config.num_attention_heads, max_seq_len=config.sequence_length)


        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size=config.hidden_size,
                num_attention_heads=config.num_attention_heads,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                bias=config.bias,
                attention_dropout=config.attention_dropout,
                rope=self.rope
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

        # replace the first position with 0s, our BOS token is always [0., 0.]
        target_onehot[:, 0] = 0.

        # concatenate target encoding with input features
        return torch.cat([target_onehot, x], dim=-1)

    def forward(self,
                x: torch.Tensor, # (batch_size, seq_len, feature_dim) - input features
                y: torch.Tensor | None = None, # (batch_size, seq_len) - binary targets for each position - or None, used to just get the representations
                ) -> torch.Tensor:
        

        # prepend BOS tokens to all tokens if y is None, else use y as is
        y = y if y is not None else torch.zeros(x.shape[0], x.shape[1], device=x.device)
        x = self._prep_inputs(x, y)  # (batch_size, seq_len, input_size)

        x = self.norm(x)
        x = self.embedding(x) 

        for layer in self.layers:
            x = layer(x)
        return self.linear_head(self.final_ln(x))

# The below code is taken from torchtune and modified
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the torchtune repository [here](https://github.com/pytorch/torchtune)
class RotaryPositionalEmbeddings(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    # TODO: delete this once all our recipes are moved off of FSDP1 since we
    # no longer need to explicitly name our param init method reset_parameters
    def reset_parameters(self):
        self.rope_init()

    def rope_init(self):
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        # Create position indexes `[0, 1, ..., max_seq_len - 1]`
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )

        # Outer product of theta and position index; output tensor has
        # a shape of [max_seq_len, dim // 2]
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()

        # cache includes both the cos and sin components and so the output shape is
        # [max_seq_len, dim // 2, 2]
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def _apply_rope(self, x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(1, 1, x.size(2), x.shape[-1] // 2, 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0] - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0] + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, *, input_pos: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q (torch.Tensor): query tensor with shape
                ``[b, n_h, s, h_d]``
            k (torch.Tensor): key tensor with shape
                ``[b, n_h, s, h_d]``
            input_pos (Optional[torch.Tensor]): Optional tensor which contains the position ids
                of each token. During training, this is used to indicate the positions
                of each token relative to its sample when packed, shape [b, s].
                During inference, this indicates the position of the current token.
                If none, assume the index of the token is its position id. Default is None.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: output tensors with shape ``[b, n_h, s, h_d]``

        Notation used for tensor shapes:
            - b: batch size
            - s: sequence length
            - n_h: num heads
            - h_d: head dim
        """
        seq_len = q.size(2)
        rope_cache = self.cache[:seq_len] if input_pos is None else self.cache[input_pos]

        q_out = self._apply_rope(q, rope_cache)
        k_out = self._apply_rope(k, rope_cache)
        return q_out, k_out