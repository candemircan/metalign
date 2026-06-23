"""
transformer and transformer parts for meta-learning as well as some baselines
"""
__all__ = ["MetaLearnerConfig", "get_model", "Transformer", "LSTM", "StaticLinear",
           "TransformerBlock", "SelfAttention", "MLP", "RotaryPositionalEmbeddings"]

from dataclasses import dataclass
from typing import Optional

import torch
from einops.layers.torch import Rearrange
from fastcore.script import call_parse
from torch import nn
from torch.nn import functional as F


@dataclass
class MetaLearnerConfig:
    """
    Configuration for meta-learning models (Transformer).

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
    mag_std:float = 1.0  # std of magnitudes from training data
    no_embed:bool = False  # if True, backbone features pass through unchanged; only prev magnitude is projected (1 -> x_sz) and added
    num_dims:int = 1       # output dims for StaticLinear heads; 1 for sequence models
    model_type:str = "transformer"  # stored in checkpoint for correct class restoration


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


class BaseMetaLearner(nn.Module):
    """Base class for meta-learning models with shared input prep, embed, and head."""
    def __init__(self, c:MetaLearnerConfig):
        super().__init__()
        self.c = c

        self.head_binary = nn.Linear(c.x_sz, 1, bias=c.logit_bias)
        self.head_magnitude = nn.Linear(c.x_sz, 1, bias=c.logit_bias)

        if getattr(c, 'no_embed', False):
            # backbone features pass through unchanged; only magnitude is projected and added
            self.embed    = None
            self.mag_proj = nn.Linear(1, c.x_sz, bias=c.bias)
        else:
            embed_x_sz = c.x_sz + 1  # magnitude only (encodes both label and strength)
            self.embed    = nn.Linear(embed_x_sz, c.x_sz, bias=c.bias)
            self.mag_proj = None
        self.ln_f = nn.LayerNorm(c.x_sz)

        self.register_buffer('mag_std', torch.tensor(c.mag_std))

        # Kendall et al. uncertainty weighting: learnable log(σ²) for each task
        # L = exp(-s₁)*L_ce + exp(-s₂)*L_mse + s₁/2 + s₂/2  where s = log(σ²)
        self.log_var_ce = nn.Parameter(torch.zeros(1))
        self.log_var_mse = nn.Parameter(torch.zeros(1))

    def compute_loss(self, loss_ce:torch.Tensor, loss_mse:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute uncertainty-weighted combined loss (Kendall et al.).
        """
        precision_ce = torch.exp(-self.log_var_ce)
        precision_mse = 0.5 * torch.exp(-self.log_var_mse)
        reg = 0.5 * (self.log_var_ce + self.log_var_mse)  # = log(σ₁) + log(σ₂)
        loss = (precision_ce * loss_ce + precision_mse * loss_mse + reg).squeeze()
        # return sigmas for logging: σ = exp(log_var / 2)
        sigma_ce = torch.exp(0.5 * self.log_var_ce)
        sigma_mse = torch.exp(0.5 * self.log_var_mse)
        return loss, sigma_ce, sigma_mse

    def _prep_inputs(self, x:torch.Tensor, y_magnitude:torch.Tensor|None = None) -> torch.Tensor:
        """concatenate scaled previous magnitude to input features `x`.
           magnitude encodes both label (sign/zero) and strength.
           if `y_magnitude` is `None`, zeros are used (BOS/no info).
        """
        bs = x.shape[0]
        if y_magnitude is not None:
            prev_mag = torch.cat([torch.zeros(bs, 1, device=x.device), y_magnitude[:, :-1]], dim=1)
            prev_mag_norm = (prev_mag / self.mag_std).unsqueeze(-1)
        else:
            prev_mag_norm = torch.zeros(bs, x.shape[1], 1, device=x.device)

        return torch.cat([prev_mag_norm, x], dim=-1)

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """Extract the representation used for downstream evaluation (no sequence context).

        Args:
            x (torch.Tensor): Backbone features (n_images, feature_dim).

        Returns:
            torch.Tensor: Representations (n_images, x_sz).
        """
        x = self._prep_inputs(x.unsqueeze(1))  # (n, 1, feature_dim+1) with zeros for prev mag
        return self.embed(x).squeeze(1)         # (n, x_sz)

    def _process_sequence(self, x:torch.Tensor) -> torch.Tensor:
        """Override in subclasses to process the embedded sequence."""
        raise NotImplementedError

    def forward(self,
                x:torch.Tensor,
                y_magnitude:torch.Tensor | None = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (batch_size, seq_len, feature_dim) - input features.
            y_magnitude: (batch_size, seq_len) - magnitude values (encodes label via sign/zero).
        """
        x = self._prep_inputs(x, y_magnitude)
        if self.embed is not None:
            x = self.embed(x)
        else:
            mag, feats = x[..., :1], x[..., 1:]
            x = feats + self.mag_proj(mag)
        x = self._process_sequence(x)
        x = self.ln_f(x)
        return self.head_binary(x), self.head_magnitude(x)


class Transformer(BaseMetaLearner):
    def __init__(self, c:MetaLearnerConfig):
        if c.x_sz % c.n_heads != 0:
            raise ValueError(f"input size ({c.x_sz}) must be divisible by the number of attention heads ({c.n_heads})")
        super().__init__(c)
        self.rope = RotaryPositionalEmbeddings(dim=c.x_sz // c.n_heads, max_seq_len=c.sl)
        self.layers = nn.ModuleList([
            TransformerBlock(x_sz=c.x_sz, n_heads=c.n_heads, int_sz=c.int_sz, act=c.act,
                             bias=c.bias, attn_drop=c.attn_drop, rope=self.rope, use_mlp=c.use_mlp)
            for _ in range(c.n_layers)
        ])

    def _process_sequence(self, x:torch.Tensor) -> torch.Tensor:
        for layer in self.layers: x = layer(x)
        return x


class LSTM(BaseMetaLearner):
    def __init__(self, c:MetaLearnerConfig):
        super().__init__(c)
        self.lstm = nn.LSTM(c.x_sz, c.x_sz, num_layers=c.n_layers, batch_first=True)

    def _process_sequence(self, x:torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return out


class StaticLinear(nn.Module):
    """No meta-learning baseline: per-image linear projection + per-function heads.

    Two linear layers with no activation. Representations are taken from the first layer output.
    Per-function binary and magnitude heads. Uses hurdle loss (Kendall uncertainty weighting).
    """
    def __init__(self, c:MetaLearnerConfig):
        super().__init__()
        self.c = c
        self.proj           = nn.Linear(c.x_sz, c.x_sz, bias=c.bias)
        self.ln_f           = nn.LayerNorm(c.x_sz)
        self.head_binary    = nn.Linear(c.x_sz, c.num_dims, bias=c.logit_bias)
        self.head_magnitude = nn.Linear(c.x_sz, c.num_dims, bias=c.logit_bias)

        self.log_var_ce  = nn.Parameter(torch.zeros(1))
        self.log_var_mse = nn.Parameter(torch.zeros(1))

    def compute_loss(self, loss_ce:torch.Tensor, loss_mse:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        precision_ce  = torch.exp(-self.log_var_ce)
        precision_mse = 0.5 * torch.exp(-self.log_var_mse)
        reg = 0.5 * (self.log_var_ce + self.log_var_mse)
        loss = (precision_ce * loss_ce + precision_mse * loss_mse + reg).squeeze()
        return loss, torch.exp(0.5 * self.log_var_ce), torch.exp(0.5 * self.log_var_mse)

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """Extract representations (n_images, x_sz)."""
        return self.ln_f(self.proj(x))

    def forward(self, x:torch.Tensor, y_magnitude:torch.Tensor|None = None) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.ln_f(self.proj(x))
        return self.head_binary(x), self.head_magnitude(x)


def get_model(c:MetaLearnerConfig) -> nn.Module:
    """Instantiate the right model class from a MetaLearnerConfig."""
    model_type = getattr(c, 'model_type', 'transformer')
    return {"transformer": Transformer, "lstm": LSTM, "static_linear": StaticLinear}[model_type](c)


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
    c = MetaLearnerConfig(sl=10)
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

    # transformer shape (no magnitude - inference mode)
    model = Transformer(c)
    binary_out, mag_out = model(x)
    assert binary_out.shape == (bs, c.sl, 1)
    assert mag_out.shape == (bs, c.sl, 1)

    # transformer shape with magnitudes (training mode)
    # magnitude encodes both label (sign/zero) and strength
    y_magnitude = torch.rand(bs, c.sl) * 8.0 - 2.0  # some positive, some negative
    binary_out, mag_out = model(x, y_magnitude)
    assert binary_out.shape == (bs, c.sl, 1)
    assert mag_out.shape == (bs, c.sl, 1)

    # prep input check - magnitude only, no one-hot
    x_prep = torch.tensor([[10, 20, 30], [40, 50, 60], [70, 80, 90], [100, 110, 120]], dtype=torch.float32).unsqueeze(0)
    y_mag_prep = torch.tensor([2.0, -1.0, 0.0, 3.0], dtype=torch.float32).unsqueeze(0)

    # with mag_std=1.0 (default), magnitude is just divided by 1
    # prev_mag shifted: [0, 2.0, -1.0, 0.0] -> normalized: [0, 2.0, -1.0, 0.0]
    x_prepped = model._prep_inputs(x_prep, y_mag_prep)
    # Trial 0: [0, 10, 20, 30]
    # Trial 1: [2.0, 40, 50, 60]
    # Trial 2: [-1.0, 70, 80, 90]
    # Trial 3: [0, 100, 110, 120]
    x_truth = torch.tensor([[0., 10, 20, 30], [2., 40, 50, 60], [-1., 70, 80, 90], [0., 100, 110, 120]], dtype=torch.float32).unsqueeze(0)
    assert torch.equal(x_prepped, x_truth), f"prepared inputs do not match expected values.\nGot: {x_prepped}\nExpected: {x_truth}"
    assert x_prepped.shape == (1, 4, 4), f"prepared inputs shape mismatch: {x_prepped.shape}"

    # prep input check - no magnitude (inference)
    x_prepped_no_mag = model._prep_inputs(x_prep)
    x_truth_no_mag = torch.tensor([[0., 10, 20, 30], [0., 40, 50, 60], [0., 70, 80, 90], [0., 100, 110, 120]], dtype=torch.float32).unsqueeze(0)
    assert torch.equal(x_prepped_no_mag, x_truth_no_mag), "prepared inputs without magnitude do not match."

    # rope test
    rope = RotaryPositionalEmbeddings(dim=8, max_seq_len=10)
    q = torch.randn(2, 4, 10, 8)
    k = torch.randn(2, 4, 10, 8)
    q_out, k_out = rope(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert not torch.equal(q, q_out)
    assert not torch.equal(k, k_out)

    # no_embed transformer test (backbone features pass through, only magnitude projected)
    c_noembed = MetaLearnerConfig(sl=10, no_embed=True)
    model_noembed = Transformer(c_noembed)
    binary_out_ne, mag_out_ne = model_noembed(x)
    assert binary_out_ne.shape == (bs, c.sl, 1)
    assert mag_out_ne.shape == (bs, c.sl, 1)
    assert model_noembed.embed is None
    assert model_noembed.mag_proj is not None
    binary_out_ne, mag_out_ne = model_noembed(x, y_magnitude)
    assert binary_out_ne.shape == (bs, c.sl, 1)

    # LSTM test
    c_lstm = MetaLearnerConfig(sl=10, model_type="lstm")
    model_lstm = LSTM(c_lstm)
    b_out, m_out = model_lstm(x, y_magnitude)
    assert b_out.shape == (bs, c.sl, 1)
    assert m_out.shape == (bs, c.sl, 1)

    # LSTM no_embed test
    c_lstm_ne = MetaLearnerConfig(sl=10, model_type="lstm", no_embed=True)
    model_lstm_ne = LSTM(c_lstm_ne)
    b_out, m_out = model_lstm_ne(x, y_magnitude)
    assert b_out.shape == (bs, c.sl, 1)
    assert model_lstm_ne.embed is None

    # StaticLinear test
    num_dims = 5
    c_sl = MetaLearnerConfig(x_sz=512, num_dims=num_dims, model_type="static_linear")
    model_sl = StaticLinear(c_sl)
    x_static = torch.randn(bs, c_sl.x_sz)
    b_out, m_out = model_sl(x_static)
    assert b_out.shape == (bs, num_dims)
    assert m_out.shape == (bs, num_dims)
    assert model_sl.encode(x_static).shape == (bs, c_sl.x_sz)

    # get_model test
    assert isinstance(get_model(c), Transformer)
    assert isinstance(get_model(c_lstm), LSTM)
    assert isinstance(get_model(c_sl), StaticLinear)

    # uncertainty weighting test (Kendall et al.)
    loss_ce = torch.tensor(0.5)
    loss_mse = torch.tensor(1.0)
    for m in [model, model_sl]:
        loss, sigma_ce, sigma_mse = m.compute_loss(loss_ce, loss_mse)
        assert loss.shape == (), f"loss should be scalar, got {loss.shape}"
        assert sigma_ce.shape == (1,), f"sigma_ce should have shape (1,), got {sigma_ce.shape}"
        assert torch.allclose(sigma_ce, torch.ones(1)), f"initial sigma_ce should be 1, got {sigma_ce}"
        assert torch.allclose(loss, torch.tensor(1.0)), f"initial loss should be 1.0, got {loss}"

