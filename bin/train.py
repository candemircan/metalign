import tomllib
from fastcore.script import bool_arg, call_parse

from metarep.model import Transformer, TransformerConfig


@call_parse
def main(
    backbone: str = "dinov2_vitb14_reg", # backbone from which the representations are extracted. the name must match data/backbone_reps/{backbone}.npz
    input_type: str = "all", # can be one of "all", "cls", "register", "patch". If "all", all representations are concatenated. "registers" is possible only if the model is trained with registers.
    embedding: bool_arg = True, # whether to use an embedding layer at the beginning of the model
    hidden_size: int = 768, # hidden size of the transformer model. if this is different from the input size, an embedding layer must be used
    num_attention_heads: int = 8, # number of attention heads in the transformer model
    intermediate_size: int = 1536, # size of the intermediate layer in the MLP of the transformer model
    num_layers: int = 6, # number of transformer layers
    hidden_act: str = "gelu", # activation function for the MLP in the transformer model
    bias: bool = False, # whether to use bias in the linear layers of the transformer model
    attention_dropout: float = 0.0, # dropout rate for the attention layers in the transformer model
    max_position_embeddings: int = 1024, # maximum number of position embeddings in the transformer model
    config_file: str = None, # path to a config file. If provided, it will override the command line arguments. Note that the input_size will always be overriden by the input size of the backbone. See "data/example_transformer_config.toml" for an example config file.
    name: str = None, # name of the model. If provided, it will be used to log the model.
):
    """
    train a meta-learning transformer model over a bunch of function learning tasks
    """

    input_size = 2304

    if config_file:
        config = tomllib.load(open(config_file, "rb"))
        config['input_size'] = input_size  # always overriden
        config = TransformerConfig(**config)
    else:
        config = TransformerConfig(
            input_size=input_size,
            embedding=embedding,
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            hidden_act=hidden_act,
            bias=bias,
            attention_dropout=attention_dropout,
            max_position_embeddings=max_position_embeddings,
            name=name
        )

    model = Transformer(config)
    print(f"Model created: {model}")
    exit(0)