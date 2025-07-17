import math
import os
import random
import tomllib
from pprint import pprint

import numpy as np
import torch
import wandb
from fastcore.script import Param, bool_arg, call_parse
from sklearn.decomposition import PCA
from torch.nn import functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange

from metarep.data import ThingsFunctionLearning
from metarep.model import Transformer, TransformerConfig

torch.set_float32_matmul_precision('high')

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
    config_file: str = None,  # path to a config file. If provided, any parameters in the file will override the corresponding command line arguments. See "data/example_transformer_config.toml" for an example config file.
    batch_size: int = 64,  # batch size for training the model
    training_steps: int = 500000,  # number of training steps per epoch
    seed: int = 1234, # random seed for reproducibility
    lr: float = 5e-5,  # learning rate for the optimizer
    weight_decay: float = 1e-4,  # weight decay for the optimizer
    warmup_steps: int = 10000,  # number of warmup steps for the learning rate scheduler
    name: str = None,  # name of the model. If provided, it will be used to log the model.
    num_components: int = None,  # number of components to use for dimensionality reduction. If None, the original data is used.
    constant_lr: bool = False,  # If True, do not schedule the LR, also no warmup then
    log_interval_steps: int = 10,  # log training loss every N steps
    eval_interval_steps: int = 100,  # evaluate the model every eval_interval_steps steps
    num_eval_episodes: int = 128,  # number of episodes to sample for evaluation
    eval_dims: Param(help="the dimensions to evaluate the model on. These dimensions are not sampled during training. It cannot be empty.", type=int, nargs="*") = [0, 1, 2], # type: ignore
    checkpoint_dir: str = "checkpoints", # directory to save checkpoints. this will be placed under data/checkpoints/{name} if name is provided. If name is None, it will be saved under data/checkpoints
    checkpoint_interval_steps: int = 1000, # save checkpoint every N steps
    resume_from_checkpoint: str = None, # path to a checkpoint to resume from
    scale: bool_arg = True,  # whether to scale the input data to have zero mean and unit variance
    spose_input: bool = False, # if True, use the SPoSE as input. Used for overfitting and debugging. The functions are also sampled from this. Therefore, this must be trivially easy. It will override `backbone` and `input_type`.
    fixed_label: bool = False,  # if True, the positives are always 1 and the negatives are always 0. If False, for a given sequence, they are reversed with 50% probability.
    weighted: bool = False, #  If True, sample positive and negative instances weighted by their magnitude. Otherwise, sample uniformly.
    compile: bool_arg = True,  # whether to compile the model with torch.compile
):
    """
    train a meta-learning transformer model over a bunch of function learning tasks
    """
    args = locals()
    if config_file:
        config_data = tomllib.load(open(config_file, "rb"))
        args.update(config_data)
    pprint(args)

    random.seed(args["seed"])
    np.random.seed(args["seed"])
    torch.manual_seed(args["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args["seed"])
    
    full_checkpoint_dir = f"data/checkpoints/{args["checkpoint_dir"]}" if args["name"] is None else f"data/checkpoints/{args["name"]}"
    if not os.path.exists(full_checkpoint_dir): os.makedirs(full_checkpoint_dir)

    representations = np.load(f"data/backbone_reps/{args["backbone"]}.npz")
    if args["input_type"] != "all": representations = {args["input_type"]: representations[args["input_type"]]}
    data = ThingsFunctionLearning(representations=representations, scale=args["scale"])

    if args["spose_input"]: 
        data.X = data.Y
        data.feature_dim = data.X.shape[1]
        args["backbone"] = "spose"

    if args["num_components"] is not None:
        pca = PCA(n_components=args["num_components"], random_state=args["seed"])
        data.X = torch.from_numpy(pca.fit_transform(data.X)).to(torch.float32)
        data.feature_dim = args["num_components"]
    
    # +2 because we always prepend the target from the previous observation to the input as a one hot vector
    # ie [0 1] if the target is 1, [1 0] if the target is 0
    input_size = data.feature_dim + 2
    
    config = TransformerConfig(
        input_size=input_size,
        embedding=args["embedding"],
        hidden_size=args["hidden_size"],
        num_attention_heads=args["num_attention_heads"],
        intermediate_size=args["intermediate_size"],
        num_layers=args["num_layers"],
        hidden_act=args["hidden_act"],
        bias=args["bias"],
        logit_bias=args["logit_bias"],
        attention_dropout=args["attention_dropout"],
        sequence_length=args["sequence_length"],
        name=args["name"],
    )

    model = Transformer(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    if args["compile"]: model.compile(fullgraph=True, mode="max-autotune")

    # no weight decay on bias and layernorm parameters
    no_decay = ["ln1", "ln2", "bias", "final_ln"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args["weight_decay"],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args["lr"])

    scheduler = None
    if not args["constant_lr"]:
        def lr_lambda(current_step):
            if current_step < args["warmup_steps"]:
                return float(current_step) / float(max(1, args["warmup_steps"]))
            progress = float(current_step - args["warmup_steps"]) / float(max(1, args["training_steps"] - args["warmup_steps"]))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        scheduler = LambdaLR(optimizer, lr_lambda)

    start_step = 0
    if args["resume_from_checkpoint"]:
        checkpoint = torch.load(args["resume_from_checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler and "scheduler_state_dict" in checkpoint and checkpoint["scheduler_state_dict"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_step = checkpoint["step"] + 1
        if args["compile"]: model.compile(fullgraph=True, mode="max-autotune")
        print(f"Resuming training from step {start_step}")

    best_eval_accuracy = -1.0
    best_checkpoint_path = f"{full_checkpoint_dir}/best.pt"
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        if 'eval_accuracy' in best_checkpoint:
            best_eval_accuracy = best_checkpoint['eval_accuracy']
            print(f"Found existing best checkpoint with accuracy: {best_eval_accuracy:.4f}")


    wandb.init(project=args["wandb_name"], name=config.name, config=args)   

    pbar = trange(start_step, args["training_steps"], desc="Training Steps")
    num_dims = list(range(data.Y.shape[1]))
    train_dims = torch.tensor([d for d in num_dims if d not in args["eval_dims"]], device=device)
    eval_dims_tensor = torch.tensor(args["eval_dims"], device=device)

    accumulated_train_loss = 0.0
    accumulated_train_correct = 0
    accumulated_train_total = 0
    
    for training_step in pbar:
        model.train()
        
        sampled_dims = torch.randint(len(train_dims), (args["batch_size"],))
        X_batch = []
        Y_batch = []
        
        for i in range(args["batch_size"]):
            dim = train_dims[sampled_dims[i]]
            X_episode, Y_episode = data.sample_episode(dim, args["sequence_length"], args["fixed_label"], args["weighted"])
            
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

        if (training_step + 1) % args["log_interval_steps"] == 0:
            avg_train_loss = accumulated_train_loss / args["log_interval_steps"]
            train_accuracy = accumulated_train_correct / accumulated_train_total
            wandb.log({"loss_train": avg_train_loss, "accuracy_train": train_accuracy, "lr": optimizer.param_groups[0]["lr"]}, step=training_step)
            accumulated_train_loss = 0.0
            accumulated_train_correct = 0
            accumulated_train_total = 0

        if (training_step + 1) % args["eval_interval_steps"] == 0:
            with torch.no_grad():
                model.eval()
                eval_losses = []
                correct_predictions = 0
                total_predictions = 0
                
                # collect episodes first, then process in batches
                X_eval_batch_list,Y_eval_batch_list = [], [] 

                for i in range(args["num_eval_episodes"]):
                    dim = eval_dims_tensor[i % len(eval_dims_tensor)] # cycle through eval_dims
                    X_episode, Y_episode =data.sample_episode(dim, args["sequence_length"], args["fixed_label"], args["weighted"])
                    
                    prev_targets = torch.cat([torch.tensor([0]), Y_episode[:-1]])
                    target_onehot = torch.nn.functional.one_hot(prev_targets.long(), num_classes=2).float()
                    target_onehot[0] = 0.0
                    
                    inputs = torch.cat([target_onehot, X_episode], dim=1)
                    
                    X_eval_batch_list.append(inputs)
                    Y_eval_batch_list.append(Y_episode)
                
                for i in range(0, args["num_eval_episodes"], args["batch_size"]):
                    batch_X = torch.stack(X_eval_batch_list[i:i+args["batch_size"]]).to(device)
                    batch_Y = torch.stack(Y_eval_batch_list[i:i+args["batch_size"]]).to(device)

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
                    torch.save({
                        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                        'eval_accuracy': eval_accuracy,
                        'step': training_step
                    }, best_checkpoint_path)
        
        if (training_step + 1) % args["checkpoint_interval_steps"] == 0:
            checkpoint_path = os.path.join(full_checkpoint_dir, "latest.pt")
            torch.save({
                'step': training_step,
                'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            }, checkpoint_path)

    wandb.finish()