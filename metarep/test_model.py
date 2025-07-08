import pytest
import torch

from metarep.model import MLP, SelfAttention, Transformer, TransformerBlock, TransformerConfig
from metarep.rope import precompute_freqs_cis


@pytest.fixture
def config():
    """Default TransformerConfig for testing."""
    return TransformerConfig(
        input_size=512,
        hidden_size=128,
        num_attention_heads=4,
        intermediate_size=256,
        num_layers=2,
        sequence_length=64,
        embedding=True,
    )

def test_mlp(config: TransformerConfig):
    mlp = MLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
    )
    x = torch.randn(2, 10, config.hidden_size)  # (batch, seq_len, hidden_size)
    output = mlp(x)
    assert output.shape == x.shape

def test_self_attention(config: TransformerConfig):
    attention = SelfAttention(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
    )
    x = torch.randn(2, 10, config.hidden_size)
    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, config.sequence_length)
    seq_len = x.shape[1]
    output = attention(x, freqs_cis=freqs_cis[:seq_len])
    assert output.shape == x.shape

def test_transformer_block(config: TransformerConfig):
    block = TransformerBlock(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
    )
    x = torch.randn(2, 10, config.hidden_size)
    head_dim = config.hidden_size // config.num_attention_heads
    freqs_cis = precompute_freqs_cis(head_dim, config.sequence_length)
    seq_len = x.shape[1]
    output = block(x, freqs_cis=freqs_cis[:seq_len])
    assert output.shape == x.shape

def test_transformer(config: TransformerConfig):
    model = Transformer(config)
    x = torch.randn(2, 10, config.input_size)
    output = model(x)
    assert output.shape == (2, 10, 1)

def test_transformer_no_embedding(config: TransformerConfig):
    config.embedding = False
    config.input_size = config.hidden_size
    model = Transformer(config)
    x = torch.randn(2, 10, config.input_size)
    output = model(x)
    assert output.shape == (2, 10, 1)

def test_transformer_raises_on_size_mismatch():
    with pytest.raises(ValueError):
        Transformer(TransformerConfig(embedding=False, input_size=128, hidden_size=256))
