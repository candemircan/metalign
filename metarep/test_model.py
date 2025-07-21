import pytest
import torch

from metarep.model import MLP, SelfAttention, Transformer, TransformerBlock, TransformerConfig
from metarep.rope import RotaryPositionalEmbeddings


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
        rope=RotaryPositionalEmbeddings(dim=config.hidden_size // config.num_attention_heads, max_seq_len=config.sequence_length),
    )
    x = torch.randn(2, 10, config.hidden_size)
    output = attention(x)
    assert output.shape == x.shape

def test_transformer_block(config: TransformerConfig):
    block = TransformerBlock(
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
        rope=RotaryPositionalEmbeddings(dim=config.hidden_size // config.num_attention_heads, max_seq_len=config.sequence_length),
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
        Transformer(TransformerConfig(embedding=False, input_size=128, hidden_size=256))

def test_bos_token():
    config = TransformerConfig(input_size=10, hidden_size=48, num_attention_heads=4, sequence_length=5)
    model = Transformer(config)
    
    # Check that BOS token is learnable parameter
    assert hasattr(model, 'bos_token')
    assert model.bos_token.shape == (2,)
    assert model.bos_token.requires_grad
    
    # Test forward pass works
    x = torch.randn(1, 3, config.input_size)
    y = torch.randint(0, 2, (1, 3))
    output = model(x, y)
    assert output.shape == (1, 3, 1)
