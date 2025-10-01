"""
transformer and transformer parts for meta-learning
"""
__all__ = ["TransformerConfig", "Transformer", "TransformerBlock", "SelfAttention", "MLP", "RotaryPositionalEmbeddings"]

from dataclasses import dataclass
from typing import Optional

import torch
from einops.layers.torch import Rearrange
from fastcore.script import call_parse
from torch import nn
from torch.nn import functional as F

@dataclass
class TransformerConfig:
    """
    Configuration for the transformer model.
    
    The defaults are kinda random and set for easy testing.
    """
    x_sz:int = 512
    n_heads:int = 4
    int_sz:int = 512
    n_layers:int = 2
    act:str = "gelu"
    bias:bool = True
    logit_bias:bool = True
    attn_drop:float = 0.1
    sl:int = 12
    use_mlp:bool = True

    def __post_init__(self):
        if self.x_sz % self.n_heads != 0:
            raise ValueError(f"input size ({self.x_sz}) must be divisible by the number of attention heads ({self.n_heads})")

class MLP(nn.Module):
    def __init__(self, x_sz:int, int_sz:int, act:str, bias:bool):
        super().__init__()
        self.act_fn = getattr(F, act, None)
        if self.act_fn is None: raise ValueError(f"Activation function '{act}' is not supported.")

        self.fc1 = nn.Linear(x_sz, int_sz, bias=bias)
        self.fc2 = nn.Linear(int_sz, x_sz, bias=bias)

    def forward(self, x:torch.Tensor) -> torch.Tensor: return self.fc2(self.act_fn(self.fc1(x)))


class SelfAttention(nn.Module):
    def __init__(self, x_sz:int, n_heads:int, bias:bool, attn_drop:float, rope:nn.Module):
        super().__init__()
        self.x_sz,self.n_heads,self.attn_drop,self.rope = x_sz,n_heads,attn_drop,rope
        self.head_sz = x_sz // n_heads

        self.qkv = nn.Linear(x_sz, x_sz * 3, bias=bias)
        self.o_proj = nn.Linear(x_sz, x_sz, bias=bias)
        self.qkv_split = Rearrange('b s (qkv h hd) -> qkv b h s hd', qkv=3, h=n_heads)
        self.o_proj_t = Rearrange('b h s hd -> b s (h hd)')


    def forward(self, x:torch.Tensor) -> torch.Tensor:
        qkv = self.qkv(x)
        q, k, v = self.qkv_split(qkv)
    
        q, k = self.rope(q, k)

        attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.attn_drop if self.training else 0.0)
        attn_output = self.o_proj_t(attn_output)
        return self.o_proj(attn_output)


class TransformerBlock(nn.Module):
    def __init__(self, x_sz:int, n_heads:int, int_sz:int, act:str, bias:bool, attn_drop:float, rope:nn.Module, use_mlp:bool):
        super().__init__()
        self.attn = SelfAttention(x_sz, n_heads, bias=bias, attn_drop=attn_drop, rope=rope)
        self.ln1 = nn.LayerNorm(x_sz)
        self.mlp = MLP(x_sz, int_sz, act, bias=bias) if use_mlp else None
        if self.mlp: self.ln2 = nn.LayerNorm(x_sz)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        if self.mlp: x = x + self.mlp(self.ln2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, c:TransformerConfig):
        super().__init__()
        self.c = c
        onehot_x_sz = c.x_sz + 2 # binary one-hot encoded previous target

        self.embed = nn.Linear(onehot_x_sz, c.x_sz, bias=c.bias)
        self.rope = RotaryPositionalEmbeddings(dim=c.x_sz // c.n_heads, max_seq_len=c.sl)
        self.layers = nn.ModuleList([
            TransformerBlock(x_sz=c.x_sz, n_heads=c.n_heads, int_sz=c.int_sz, act=c.act,
                             bias=c.bias, attn_drop=c.attn_drop, rope=self.rope, use_mlp=c.use_mlp) for _ in range(c.n_layers)
                             ])
        self.ln_f = nn.LayerNorm(c.x_sz)
        self.head = nn.Linear(c.x_sz, 1, bias=c.logit_bias)


    def _prep_inputs(self, x:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        "Concatenate one-hot previous targets to input features `x`."
        bs = x.shape[0]
        # get the previous targets and prepend with zeros
        prev_targets = torch.cat([torch.zeros(bs, 1, device=x.device), y[:, :-1]], dim=1)
        target_onehot = F.one_hot(prev_targets.long(), num_classes=2).float()
        target_onehot[:, 0] = 0. # our BOS token is always [0., 0.]
        return torch.cat([target_onehot, x], dim=-1)

    def forward(self,
                x:torch.Tensor, # (batch_size, seq_len, feature_dim) - input features
                y:torch.Tensor | None = None, # (batch_size, seq_len) - binary targets for each position - or None
                ) -> torch.Tensor:
        y = y if y is not None else torch.zeros(x.shape[0], x.shape[1], device=x.device)
        x = self._prep_inputs(x, y)

        x = self.embed(x)
        for layer in self.layers: x = layer(x)
        return self.head(self.ln_f(x))

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
    

    

@call_parse
def main():
    # config and mock input
    bs = 2
    c = TransformerConfig(sl=10)
    x = torch.randn(bs, c.sl, c.x_sz)

    # mlp shape
    mlp = MLP(c.x_sz, c.int_sz, c.act, c.bias)
    assert mlp(x).shape == (bs, c.sl, c.x_sz)

    # attn shape
    rope = RotaryPositionalEmbeddings(dim=c.x_sz // c.n_heads, max_seq_len=c.sl)
    attn = SelfAttention(c.x_sz, c.n_heads, c.bias, c.attn_drop, rope)
    assert attn(x).shape == (bs, c.sl, c.x_sz)

    # block shape
    block = TransformerBlock(c.x_sz, c.n_heads, c.int_sz, c.act, c.bias, c.attn_drop, rope, c.use_mlp)
    assert block(x).shape == (bs, c.sl, c.x_sz)

    # transformer shape
    model = Transformer(c)
    y = torch.randint(0, 2, (bs, c.sl)).float()
    assert model(x, y).shape == (bs, c.sl, 1)

    # prep input check
    x_prep = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]], dtype=torch.float32).unsqueeze(0)
    y_prep = torch.tensor([1, 0, 0, 1], dtype=torch.long).unsqueeze(0)
    x_oh_truth = torch.tensor([[0., 0., 10, 20, 30], [0., 1., 40, 50, 60], [1., 0., 70, 80, 90], [1., 0., 100, 110, 120]], dtype=torch.float32).unsqueeze(0)

    x_oh = model._prep_inputs(x_prep, y_prep)
    assert torch.equal(x_oh, x_oh_truth), "Prepared inputs do not match expected values."
    assert x_oh.shape == (1, 4, 5), "Prepared inputs shape mismatch."
    
    # rope test
    rope = RotaryPositionalEmbeddings(dim=8, max_seq_len=10)
    q = torch.randn(2, 4, 10, 8)
    k = torch.randn(2, 4, 10, 8)
    q_out, k_out = rope(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert not torch.equal(q, q_out)
    assert not torch.equal(k, k_out)