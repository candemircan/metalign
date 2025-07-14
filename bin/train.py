import math
import os

import numpy as np
import tomllib
import torch
from fastcore.script import Param, bool_arg, call_parse
from sklearn.decomposition import PCA
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange

import wandb
from metarep.data import ThingsFunctionLearning
from metarep.model import Transformer, TransformerConfig


@call_parse
def main(
    backbone: str = "dinov2_vitb14_reg",  # backbone from which the representations are extracted. the name must match data/backbone_reps/{backbone}.npz
    input_type: str = "all",  # can be one of "all", "cls", "register", "patch". If "all", all representations are concatenated. "registers" is possible only if the model is trained with registers.
    wandb_name: str = "metarep",  # name of the wandb project. Used to log the model.
    embedding: bool = False,  # whether to use an embedding layer at the beginning of the model
    hidden_size: int = 768,  # hidden size of the transformer model. if this is different from the input size, an embedding layer must be used
    num_attention_heads: int = 12,  # number of attention heads in the transformer model
    intermediate_size: int = 3072,  # size of the intermediate layer in the MLP of the transformer model
    num_layers: int = 6,  # number of transformer layers
    hidden_act: str = "gelu",  # activation function for the MLP in the transformer model
    bias: bool = False,  # whether to use bias in the linear layers of the transformer model
    logit_bias: bool_arg = True,  # whether to use bias in the final linear layer of the transformer model. If False, the final layer will not have a bias term.
    attention_dropout: float = 0.1,  # dropout rate for the attention layers in the transformer model
    sequence_length: int = 100,  # maximum number of position embeddings in the transformer model
    config_file: str = None,  # path to a config file. If provided, it will override the command line arguments. Note that the input_size will always be overriden by the input size of the backbone. See "data/example_transformer_config.toml" for an example config file.
    batch_size: int = 64,  # batch size for training the model
    training_steps: int = 20000,  # number of training steps per epoch
    lr: float = 1e-4,  # learning rate for the optimizer
    weight_decay: float = 1e-4,  # weight decay for the optimizer
    warmup_steps: int = 1000,  # number of warmup steps for the learning rate scheduler
    name: str = None,  # name of the model. If provided, it will be used to log the model.
    num_components: int = None,  # number of components to use for dimensionality reduction. If None, the original data is used.
    constant_lr: bool = False,  # If True, do not schedule the LR, also no warmup then
    log_interval_steps: int = 10,  # log training loss every N steps
    eval_interval_steps: int = 100,  # evaluate the model every eval_interval_steps steps
    num_eval_episodes: int = 128,  # number of episodes to sample for evaluation
    eval_dims: Param(help="the dimensions to evaluate the model on. These dimensions are not sampled during training. It cannot be empty.", type=int, nargs="*") = [0, 1, 2],
    checkpoint_dir: str = "checkpoints", # directory to save checkpoints. this will be placed under data/checkpoints/{name} if name is provided. If name is None, it will be saved under data/checkpoints
    checkpoint_interval_steps: int = 1000, # save checkpoint every N steps
    resume_from_checkpoint: str = None, # path to a checkpoint to resume from
):
    """
    train a meta-learning transformer model over a bunch of function learning tasks
    """
    checkpoint_dir = f"data/{checkpoint_dir}" if name is None else f"data/checkpoints/{name}"
    if not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir)

    representations = np.load(f"data/backbone_reps/{backbone}.npz")
    if input_type != "all": representations = {input_type: representations[input_type]}
    data = ThingsFunctionLearning(representations=representations)

    if num_components is not None:
        pca = PCA(n_components=num_components)
        data.X = torch.from_numpy(pca.fit_transform(data.X)).to(torch.float32)
        data.feature_dim = num_components
    
    # +2 because we always prepend the target from the previous observation to the input as a one hot vector
    # ie [0 1] if the target is 1, [1 0] if the target is 0
    input_size = data.feature_dim + 2
    if config_file:
        config = tomllib.load(open(config_file, "rb"))
        config["input_size"] = input_size  # always overriden
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

    # no weight decay on bias and layernorm parameters
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
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, training_steps - warmup_steps))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

    start_step = 0
    if resume_from_checkpoint:
        checkpoint = torch.load(resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"] + 1
        print(f"Resuming training from step {start_step}")

    best_eval_accuracy = -1.0
    best_checkpoint_path = f"{checkpoint_dir}/best.pt"
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        if 'eval_accuracy' in best_checkpoint:
            best_eval_accuracy = best_checkpoint['eval_accuracy']
            print(f"Found existing best checkpoint with accuracy: {best_eval_accuracy:.4f}")

    config_dict = vars(config)
    config_dict["num_components"] = num_components
    config_dict["constant_lr"] = constant_lr
    config_dict["log_interval_steps"] = log_interval_steps # Log this new param
    config_dict["eval_interval_steps"] = eval_interval_steps # Log this new param
    config_dict["num_eval_episodes"] = num_eval_episodes # Log this new param

    wandb.init(project=wandb_name, name=config.name, config=config_dict)   

    pbar = trange(start_step, training_steps, desc="Training Steps")
    num_dims = list(range(data.Y.shape[1]))
    train_dims = [d for d in num_dims if d not in eval_dims]
    
    accumulated_train_loss = 0.0
    accumulated_train_correct = 0
    accumulated_train_total = 0
    
    for training_step in pbar:
        model.train()
        
        sampled_dims = torch.randint(len(train_dims), (batch_size,))
        X_batch = []
        Y_batch = []
        
        for i in range(batch_size):
            dim = train_dims[sampled_dims[i]]
            X_episode, Y_episode = data.sample_episode(dim, sequence_length)
            
            prev_targets = torch.cat([torch.tensor([0]), Y_episode[:-1]]) 

            target_onehot = torch.nn.functional.one_hot(prev_targets.long(), num_classes=2).float()
            target_onehot[0] = 0.0 
            
            inputs = torch.cat([target_onehot, X_episode], dim=1)

            X_batch.append(inputs)
            Y_batch.append(Y_episode)
        
        X_batch = torch.stack(X_batch).to(device)
        Y_batch = torch.stack(Y_batch).to(device)
        
        logits = model(X_batch).squeeze(-1)
        loss =  F.binary_cross_entropy_with_logits(logits, Y_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler: scheduler.step()
        
        accumulated_train_loss += loss.item()
        predictions = (torch.sigmoid(logits) > 0.5).float()
        accumulated_train_correct += (predictions == Y_batch).sum().item()
        accumulated_train_total += Y_batch.numel()

        if (training_step + 1) % log_interval_steps == 0:
            avg_train_loss = accumulated_train_loss / log_interval_steps
            train_accuracy = accumulated_train_correct / accumulated_train_total
            wandb.log({"loss_train": avg_train_loss, "accuracy_train": train_accuracy, "lr": optimizer.param_groups[0]["lr"]}, step=training_step)
            accumulated_train_loss = 0.0
            accumulated_train_correct = 0
            accumulated_train_total = 0

        if (training_step + 1) % eval_interval_steps == 0:
            with torch.no_grad():
                model.eval()
                eval_losses = []
                correct_predictions = 0
                total_predictions = 0
                
                # collect episodes first, then process in batches
                X_eval_batch_list,Y_eval_batch_list = [], [] 

                for i in range(num_eval_episodes):
                    dim = eval_dims[i % len(eval_dims)] # cycle through eval_dims
                    X_episode, Y_episode = data.sample_episode(dim, sequence_length)
                    
                    prev_targets = torch.cat([torch.tensor([0]), Y_episode[:-1]])
                    target_onehot = torch.nn.functional.one_hot(prev_targets.long(), num_classes=2).float()
                    target_onehot[0] = 0.0
                    
                    inputs = torch.cat([target_onehot, X_episode], dim=1)
                    
                    X_eval_batch_list.append(inputs)
                    Y_eval_batch_list.append(Y_episode)
                
                for i in range(0, num_eval_episodes, batch_size):
                    batch_X = torch.stack(X_eval_batch_list[i:i+batch_size]).to(device)
                    batch_Y = torch.stack(Y_eval_batch_list[i:i+batch_size]).to(device)

                    logits_eval = model(batch_X).squeeze(-1)
                    loss_eval =  F.binary_cross_entropy_with_logits(logits_eval, batch_Y)
                    eval_losses.append(loss_eval.item())
                    
                    predictions = (torch.sigmoid(logits_eval) > 0.5).float()
                    correct_predictions += (predictions == batch_Y).sum().item()
                    total_predictions += batch_Y.numel()

                avg_eval_loss = np.mean(eval_losses)
                eval_accuracy = correct_predictions / total_predictions
                wandb.log({"loss_eval": avg_eval_loss, "accuracy_eval": eval_accuracy}, step=training_step)
                pbar.set_postfix(eval_loss=f"{avg_eval_loss:.4f}", eval_acc=f"{eval_accuracy:.4f}")

                if eval_accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_accuracy
                    torch.save(model.state_dict(), best_checkpoint_path)
        
        if (training_step + 1) % checkpoint_interval_steps == 0:
            checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
            torch.save({
                'step': training_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            }, checkpoint_path)

    wandb.finish()