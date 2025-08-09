import pytest
import torch

from metalign.model import MLP, SelfAttention, Transformer, TransformerBlock, TransformerConfig
from metalign.rope import RotaryPositionalEmbeddings


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
        bias=True,
        attention_dropout=0.1,
    )

def test_mlp(config: TransformerConfig):
    mlp = MLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        bias=config.bias,
    )
    x = torch.randn(2, 10, config.hidden_size)  # (batch, seq_len, hidden_size)
    output = mlp(x)
    assert output.shape == x.shape

def test_self_attention(config: TransformerConfig):
    rope = RotaryPositionalEmbeddings(dim=config.hidden_size // config.num_attention_heads, max_seq_len=config.sequence_length)
    attention = SelfAttention(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        bias=config.bias,
        attention_dropout=config.attention_dropout,
        rope=rope,
    )
    x = torch.randn(2, 10, config.hidden_size)
    output = attention(x)
    assert output.shape == x.shape

def test_transformer_block(config: TransformerConfig):
    rope = RotaryPositionalEmbeddings(dim=config.hidden_size // config.num_attention_heads, max_seq_len=config.sequence_length)
    block = TransformerBlock(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        bias=config.bias,
        attention_dropout=config.attention_dropout,
        rope=rope,
    )
    x = torch.randn(2, 10, config.hidden_size)

    output = block(x)
    assert output.shape == x.shape

def test_transformer(config: TransformerConfig):
    model = Transformer(config)
    x = torch.randn(2, 10, config.input_size)
    y = torch.randint(0, 2, (2, 10))
    output = model(x, y)
    assert output.shape == (2, 10, 1)

def test_transformer_no_embedding(config: TransformerConfig):
    config.embedding = False
    config.input_size = config.hidden_size - 2  # Account for the +2 that happens internally
    model = Transformer(config)
    x = torch.randn(2, 10, config.input_size)
    y = torch.randint(0, 2, (2, 10))
    output = model(x, y)
    assert output.shape == (2, 10, 1)

def test_transformer_raises_on_size_mismatch():
    with pytest.raises(ValueError):
        Transformer(TransformerConfig(embedding=False, input_size=128, hidden_size=256, sequence_length=64))

def test_transformer_prep_inputs(config: TransformerConfig):
    model = Transformer(config)
    x = torch.tensor([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
        [100, 110, 120],
    ], dtype=torch.float32).unsqueeze(0)  # (batch_size, seq_len, input_size)

    y = torch.tensor([1, 0, 0, 1], dtype=torch.long).unsqueeze(0)  # (batch_size, seq_len)

    expected_prepped_inputs = torch.tensor([
        [0., 0., 10, 20, 30],
        [0., 1., 40, 50, 60],
        [1., 0., 70, 80, 90],
        [1., 0., 100, 110, 120],
    ], dtype=torch.float32).unsqueeze(0)  # (batch_size, seq_len, input_size + 2)

    prepped_inputs = model._prep_inputs(x, y)
    assert torch.equal(prepped_inputs, expected_prepped_inputs), "Prepared inputs do not match expected values."
    assert prepped_inputs.shape == (1, 4, 5), "Prepared inputs shape mismatch."

def test_rope():
    rope = RotaryPositionalEmbeddings(dim=32, max_seq_len=100)
    q = torch.randn(2, 4, 10, 32) # B, H, S, D
    k = torch.randn(2, 4, 10, 32)
    q_out, k_out = rope(q, k)
    assert q_out.shape == q.shape
    assert k_out.shape == k.shape
    assert not torch.equal(q, q_out)
    assert not torch.equal(k, k_out)