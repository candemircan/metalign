#TODO: no need for epochs, just steps
#TODO: checkpointing
import math

import numpy as np
import tomllib
import torch
from fastcore.script import bool_arg, call_parse
from sklearn.decomposition import PCA
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm, trange

import wandb
from metarep.data import ThingsFunctionLearning
from metarep.metrics import r2_score
from metarep.model import Transformer, TransformerConfig


@call_parse
def main(
    backbone: str = "dinov2_vitb14_reg", # backbone from which the representations are extracted. the name must match data/backbone_reps/{backbone}.npz
    input_type: str = "all", # can be one of "all", "cls", "register", "patch". If "all", all representations are concatenated. "registers" is possible only if the model is trained with registers.
    wandb_name: str = "metarep", # name of the wandb project. Used to log the model.
    embedding: bool = False, # whether to use an embedding layer at the beginning of the model
    hidden_size: int = 768, # hidden size of the transformer model. if this is different from the input size, an embedding layer must be used
    num_attention_heads: int = 12, # number of attention heads in the transformer model
    intermediate_size: int = 3072, # size of the intermediate layer in the MLP of the transformer model
    num_layers: int = 6, # number of transformer layers
    hidden_act: str = "gelu", # activation function for the MLP in the transformer model
    bias: bool = False, # whether to use bias in the linear layers of the transformer model
    logit_bias: bool_arg = True, # whether to use bias in the final linear layer of the transformer model. If False, the final layer will not have a bias term.
    attention_dropout: float = 0.1, # dropout rate for the attention layers in the transformer model
    sequence_length: int = 100, # maximum number of position embeddings in the transformer model
    config_file: str = None, # path to a config file. If provided, it will override the command line arguments. Note that the input_size will always be overriden by the input size of the backbone. See "data/example_transformer_config.toml" for an example config file.
    batch_size: int = 64, # batch size for training the model
    epochs: int = 100, # number of epochs to train the model
    steps_per_epoch: int = 100, # number of training steps per epoch
    lr: float = 1e-4, # learning rate for the optimizer
    weight_decay: float = 1e-4, # weight decay for the optimizer
    warmup_steps: int = 1000, # number of warmup steps for the learning rate scheduler
    max_latent: int = 41, # the maximum dimension ID + 1 that will be sampled from SPoSE during training. Dimensions 0, 1, and 2 are never sampled for training.
    name: str = None, # name of the model. If provided, it will be used to log the model.
    num_components: int = None, # number of components to use for dimensionality reduction. If None, the original data is used.
    scale: bool = False, # If True, do min-max scaling. If False, do not.
    constant_lr: bool = False, # If True, do not schedule the LR, also no warmup then
    easy_mode: bool = False, # If True, only train and eval on one dimension (0)
    easy_mode_dim: int = 0, # The dimension to use for easy mode. Only used if easy_mode is True.
):
    """
    train a meta-learning transformer model over a bunch of function learning tasks
    """

    eval_dims = [0, 1, 2] if not easy_mode else [easy_mode_dim]

    representations = np.load(f"data/backbone_reps/{backbone}.npz")
    if input_type != "all": representations = {input_type: representations[input_type]}
    data = ThingsFunctionLearning(representations=representations)
    
    if easy_mode:
        total_images = len(data)
        train_size = int(0.8 * total_images)
        indices = torch.randperm(total_images)
        train_indices = indices[:train_size]
        eval_indices = indices[train_size:]
    else:
        train_indices = eval_indices = None
    if num_components is not None:
        pca = PCA(n_components=num_components)
        data.X = torch.from_numpy(pca.fit_transform(data.X)).to(torch.float32)
        data.feature_dim = num_components
    input_size = data.feature_dim + 1 # +1 because we always prepend the target from the previous observation to the input


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
            logit_bias=logit_bias,
            attention_dropout=attention_dropout,
            sequence_length=sequence_length,
            name=name,
        )

    model = Transformer(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    no_decay = ["ln1", "ln2", "bias", "final_ln"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    
    scheduler = None
    if not constant_lr:
        num_training_steps = epochs * steps_per_epoch

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

    config_dict = vars(config)
    config_dict["num_components"] = num_components
    config_dict["easy_mode"] = easy_mode
    config_dict["easy_mode_dim"] = easy_mode_dim
    config_dict["constant_lr"] = constant_lr
    config_dict["scale"] = scale

    wandb.init(project=wandb_name, name=config.name, config=config_dict)

    
    criterion = torch.nn.MSELoss()

    
    pbar = trange(epochs, desc="Epochs")
    for epoch in pbar:
        model.train()
        for step in tqdm(range(steps_per_epoch), desc="Steps", leave=False):
            
            if easy_mode:
                batch_indices = torch.randint(0, len(train_indices), (batch_size, config.sequence_length))
                indices = train_indices[batch_indices]
            else:
                indices = torch.randint(0, len(data), (batch_size, config.sequence_length))
            dims = torch.randint(3, max_latent, (batch_size,)) if not easy_mode else torch.full((batch_size,), easy_mode_dim)

            batch_x = data.X[indices].to(device)
            
            # get the unscaled target values for each sequence in the batch
            dims_broadcast = dims.view(-1, 1).expand(-1, config.sequence_length)
            y_unscaled_batch = data.Y[indices, dims_broadcast]

            if scale:
                # calculate min and max for each sequence
                min_vals = torch.min(y_unscaled_batch, dim=1, keepdim=True).values
                max_vals = torch.max(y_unscaled_batch, dim=1, keepdim=True).values
                
                # scale each sequence
                range_vals = max_vals - min_vals
                batch_y = torch.zeros_like(y_unscaled_batch)
                valid_range_mask = range_vals.squeeze(1) > 1e-6
                
                if valid_range_mask.any(): batch_y[valid_range_mask] = (y_unscaled_batch[valid_range_mask] - min_vals[valid_range_mask]) / range_vals[valid_range_mask]
            else:
                batch_y = y_unscaled_batch
            
            batch_y = batch_y.to(device)
            
            prev_y = torch.zeros_like(batch_y)
            prev_y[:, 1:] = batch_y[:, :-1]
            
            transformer_input = torch.cat([batch_x, prev_y.unsqueeze(-1)], dim=-1)
            transformer_target = batch_y.unsqueeze(-1)
            
            optimizer.zero_grad()
            predictions = model(transformer_input)
            loss = criterion(predictions, transformer_target)
            r2 = r2_score(transformer_target, predictions)
            
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            
            log_data = {"train_loss": loss.item(), "train_r2": r2.item()}
            if scheduler: log_data["lr"] = scheduler.get_last_lr()[0]
            else: log_data["lr"] = lr
            wandb.log(log_data)

        # Evaluation
        model.eval()
        total_loss = 0
        total_r2 = 0
        num_eval_steps = 100 

        with torch.no_grad():
            for _ in range(num_eval_steps):
                seq_len = np.random.randint(3, config.sequence_length)
                if easy_mode:
                    batch_indices = torch.randint(0, len(eval_indices), (batch_size, seq_len))
                    indices = eval_indices[batch_indices]
                else:
                    indices = torch.randint(0, len(data), (batch_size, seq_len))
                
                dim = np.random.choice(eval_dims)

                batch_x = data.X[indices].to(device)
                y_unscaled_batch = data.Y[indices, dim]

                if scale:
                    min_vals = torch.min(y_unscaled_batch, dim=1, keepdim=True).values
                    max_vals = torch.max(y_unscaled_batch, dim=1, keepdim=True).values
                    
                    range_vals = max_vals - min_vals
                    
                    batch_y = torch.zeros_like(y_unscaled_batch)
                    valid_range_mask = range_vals.squeeze() > 1e-6
                    
                    if valid_range_mask.any():
                        batch_y[valid_range_mask] = (y_unscaled_batch[valid_range_mask] - min_vals[valid_range_mask]) / range_vals[valid_range_mask]
                else:
                    batch_y = y_unscaled_batch

                batch_y = batch_y.to(device)
                
                prev_y = torch.zeros_like(batch_y)
                prev_y[:, 1:] = batch_y[:, :-1]
                
                transformer_input = torch.cat([batch_x, prev_y.unsqueeze(-1)], dim=-1)
                transformer_target = batch_y.unsqueeze(-1)
                
                predictions = model(transformer_input)
                loss = criterion(predictions, transformer_target)
                r2 = r2_score(transformer_target, predictions)
                
                total_loss += loss.item()
                total_r2 += r2.item()

        avg_loss = total_loss / num_eval_steps
        avg_r2 = total_r2 / num_eval_steps
        eval_metrics = {"eval_loss": avg_loss, "eval_r2": avg_r2}
        wandb.log(eval_metrics)
        pbar.set_postfix(eval_metrics)


    if config.name:
        torch.save(model.state_dict(), f"{config.name}.pt")
        print(f"Model saved to {config.name}.pt")
    
    wandb.finish()