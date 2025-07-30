import os
import random
import tomllib
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import wandb
from fastcore.script import Param, bool_arg, call_parse
from schedulefree import AdamWScheduleFree
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import trange

from metarep.data import SAEActivationsCache, SAEEpisodeDataset, SimpleEpisodeDataset, ThingsEpisodeDataset, prepare_things_spose
from metarep.model import Transformer, TransformerConfig

torch.set_float32_matmul_precision('high')

@call_parse
def main(
    backbone: str = "things_dinov2_vitb14_reg",  # backbone from which the representations are extracted. the name must match data/backbone_reps/{backbone}.npz
    input_type: str = "all",  # can be one of "all", "cls", "register", "patch". If "all", all representations are concatenated. "registers" is possible only if the model is trained with registers.
    wandb_log: bool_arg = True,  # whether to log the model to wandb. If False, no logging is done.
    wandb_name: str = "metarep",  # name of the wandb project. Used to log the model.
    embedding: bool_arg = True,  # whether to use an embedding layer at the beginning of the model
    hidden_size: int = 768,  # hidden size of the transformer model. if this is different from the input size, an embedding layer must be used
    num_attention_heads: int = 12,  # number of attention heads in the transformer model
    intermediate_size: int = 3072,  # size of the intermediate layer in the MLP of the transformer model
    num_layers: int = 6,  # number of transformer layers
    hidden_act: str = "gelu",  # activation function for the MLP in the transformer model
    bias: bool_arg = True,  # whether to use bias in the linear layers of the transformer model
    logit_bias: bool_arg = True,  # whether to use bias in the final linear layer of the transformer model. If False, the final layer will not have a bias term.
    attention_dropout: float = 0.0,  # dropout rate for the attention layers in the transformer model
    pe_dropout: float = 0.0,  # dropout rate for the positional encoding in the transformer model. Used only with the sinusoidal positional encoding.
    sequence_length: int = 120,  # maximum number of position embeddings in the transformer model
    config_file: str = None,  # path to a config file. If provided, any parameters in the file will override the corresponding command line arguments. See "data/example_transformer_config.toml" for an example config file.
    batch_size: int = 256,  # batch size for training the model
    training_steps: int = 1000000,  # number of training steps per epoch
    seed: int = 1234, # random seed for reproducibility
    lr: float = 0.0025,  # learning rate for the optimizer
    weight_decay: float = 0.,  # weight decay for the optimizer
    warmup_steps: int = 1000,  # number of warmup steps for the learning rate scheduler
    name: str = None,  # name of the model. If provided, it will be used to log the model.
    log_interval_steps: int = 10,  # log training loss every N steps
    eval_interval_steps: int = 100,  # evaluate the model every eval_interval_steps steps
    num_eval_episodes: int = 128,  # number of episodes to sample for evaluation
    eval_dims: Param(help="the dimensions to evaluate the model on. These dimensions are not sampled during training. For SAE mode, this is ignored.", type=int, nargs="*") = [0, 1, 2], # type: ignore
    tags: Param(help="tags to use for the wandb run. If empty, no tags are used.", type=str, nargs="*") = [],  # type: ignore
    checkpoint_dir: str = "checkpoints", # directory to save checkpoints. this will be placed under data/checkpoints/{name} if name is provided. If name is None, it will be saved under data/checkpoints
    checkpoint_interval_steps: int = 1000, # save checkpoint every N steps
    resume_from_checkpoint: str = None, # path to a checkpoint to resume from
    scale: bool_arg = True,  # whether to scale the input data to have zero mean and unit variance
    fixed_label: bool = False,  # if True, the positives are always 1 and the negatives are always 0. If False, for a given sequence, they are reversed with 50% probability.
    weighted: bool = False, #  If True, sample positive and negative instances weighted by their magnitude. Otherwise, sample uniformly.
    positional_embedding_type: str = "learned",  # needs to be one of "learned", "sinusoidal", or "rope"
    compile: bool = False,  # whether to compile the model with torch.compile
    sae_mode: bool = False,  # whether to use SAE training mode with separate train/test datasets
    sae_train_backbone: str = "coco_dinov2_vitb14_reg",  # for SAE mode: backbone for train data.
    sae_test_backbone: str = "things_dinov2_vitb14_reg",  # for SAE mode: backbone for test data.
    train_sae_features: str = "coco_sae-top_k-64-cls_only-layer_11-hook_resid_post",  # for SAE mode: SAE features for training data
    test_sae_features: str = "things_sae-top_k-64-cls_only-layer_11-hook_resid_post",  # for SAE mode: SAE features for test data
    min_nonzero: int = 120,  # for SAE mode: minimum number of non-zero activations per column to keep it in the final array
    simple_mode: bool = False,  # whether to use simple sklearn-generated classification functions instead of backbone representations
):
    """
    train a meta-learning transformer model over function learning tasks.
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

    if args["simple_mode"]:
        n_features = 5
        
        train_episode_dataset = SimpleEpisodeDataset(
            n_samples=10000,
            n_features=n_features,
            scale=args["scale"],
            random_state=args["seed"],
            seq_len=args["sequence_length"],
            fixed_label=args["fixed_label"],
            epoch_size=args["training_steps"]
        )
        
        eval_episode_dataset = SimpleEpisodeDataset(
            n_samples=10000,
            n_features=n_features,
            scale=args["scale"],
            random_state=args["seed"] + 9999,  # different seed for eval
            seq_len=args["sequence_length"],
            fixed_label=args["fixed_label"],
            epoch_size=args["num_eval_episodes"]
        )
        
    elif args["sae_mode"]:
        train_backbone = args["sae_train_backbone"]
        test_backbone = args["sae_test_backbone"] 
        
        train_representations = np.load(f"data/backbone_reps/{train_backbone}.npz")
        train_inputs = np.concatenate([train_representations[key] for key in train_representations.keys()], axis=1) if args["input_type"] == "all" else train_representations[args["input_type"]]
        
        test_representations = np.load(f"data/backbone_reps/{test_backbone}.npz")
        test_inputs = np.concatenate([test_representations[key] for key in test_representations.keys()], axis=1) if args["input_type"] == "all" else test_representations[args["input_type"]]
        
        # Get feature dimension from input data
        feature_dim = train_inputs.shape[1]
        
        train_dims = list(range(SAEActivationsCache(args["train_sae_features"], data_root=Path("data/sae"), min_nonzero=args["min_nonzero"]).activations.shape[1]))
        eval_dims = list(range(SAEActivationsCache(args["test_sae_features"], data_root=Path("data/sae"), min_nonzero=args["min_nonzero"]).activations.shape[1]))
        
        train_episode_dataset = SAEEpisodeDataset(
            inputs=train_inputs,
            sae_features=args["train_sae_features"],
            data_root=Path("data/sae"),
            seq_len=args["sequence_length"],
            scale=args["scale"],
            min_nonzero=args["min_nonzero"],
            fixed_label=args["fixed_label"],
            weighted=args["weighted"],
            train_dims=train_dims,
            epoch_size=args["training_steps"]
        )
        
        eval_episode_dataset = SAEEpisodeDataset(
            inputs=test_inputs,
            sae_features=args["test_sae_features"],
            data_root=Path("data/sae"),
            seq_len=args["sequence_length"],
            scale=args["scale"],
            min_nonzero=args["min_nonzero"],
            fixed_label=args["fixed_label"],
            weighted=args["weighted"],
            train_dims=eval_dims,
            epoch_size=args["num_eval_episodes"]
        )
        
    else:
        representations = np.load(f"data/backbone_reps/{args["backbone"]}.npz")
        if args["input_type"] != "all": representations = {args["input_type"]: representations[args["input_type"]]}
        
        # Get feature dimension from representations
        if args["input_type"] == "all":
            feature_dim = sum(representations[key].shape[1] for key in representations.keys())
        else:
            feature_dim = representations[args["input_type"]].shape[1]
        
        # Prepare SPoSE data for dimension info
        X, Y = prepare_things_spose(representations, data_root=Path("data/external"))
        num_dims = list(range(Y.shape[1]))
        train_dims = [d for d in num_dims if d not in args["eval_dims"]]
        eval_dims = args["eval_dims"]
        
        train_episode_dataset = ThingsEpisodeDataset(
            representations=representations,
            data_root=Path("data/external"),
            seq_len=args["sequence_length"],
            scale=args["scale"],
            fixed_label=args["fixed_label"],
            weighted=args["weighted"],
            train_dims=train_dims,
            epoch_size=args["training_steps"]
        )
        
        eval_episode_dataset = ThingsEpisodeDataset(
            representations=representations,
            data_root=Path("data/external"),
            seq_len=args["sequence_length"],
            scale=args["scale"],
            fixed_label=args["fixed_label"],
            weighted=args["weighted"],
            train_dims=eval_dims,
            epoch_size=args["num_eval_episodes"]
        )

    # Create DataLoaders
    train_loader = DataLoader(
        train_episode_dataset,
        batch_size=args["batch_size"],
        num_workers=min(4, os.cpu_count()),
        pin_memory=torch.cuda.is_available(),
        shuffle=True,
        drop_last=True
    )
    
    eval_loader = DataLoader(
        eval_episode_dataset,
        batch_size=args["batch_size"],
        num_workers=min(2, os.cpu_count()),
        pin_memory=torch.cuda.is_available(),
        shuffle=False,
        drop_last=False
    )

    config = TransformerConfig(
        input_size=feature_dim,
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
        positional_embedding_type=args["positional_embedding_type"],
    )

    model = Transformer(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)

    if args["compile"]: model.compile(fullgraph=True, mode="max-autotune")

    optimizer = AdamWScheduleFree(model.parameters(),lr=args["lr"],weight_decay=args["weight_decay"], warmup_steps=args["warmup_steps"])

    start_step = 0
    if args["resume_from_checkpoint"]:
        checkpoint = torch.load(args["resume_from_checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_step = checkpoint["step"] + 1
        print(f"Resuming training from step {start_step}")

    best_eval_accuracy = -1.0
    best_checkpoint_path = f"{full_checkpoint_dir}/best.pt"
    if os.path.exists(best_checkpoint_path):
        best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        if 'eval_accuracy' in best_checkpoint:
            best_eval_accuracy = best_checkpoint['eval_accuracy']
            print(f"Found existing best checkpoint with accuracy: {best_eval_accuracy:.4f}")


    if args["wandb_log"]:
        # hacky way to see if i'm training on juwels, which does not have internet in compute nodes
        # in this case, gotta also run bin/sync_wandb.sh from the login node
        device_name = os.uname()[1]
        wandb.init(project=args["wandb_name"], name=config.name, config=args, tags=args["tags"], mode="offline" if "juwels" in device_name else "online")
        # only watch if not compiled, as we get an error otherwise
        if not args["compile"]:
            wandb.watch(model, log='all', log_freq=args["log_interval_steps"] * 10)

    pbar = trange(start_step, args["training_steps"], desc="Training Steps")
    
    accumulated_train_loss = 0.0
    accumulated_train_correct = 0
    accumulated_train_total = 0

    train_iterator = iter(train_loader)

    for training_step in pbar:
        model.train()
        optimizer.train()

        # get next batch from DataLoader
        try:
            X_batch, Y_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            X_batch, Y_batch = next(train_iterator)

        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        logits = model(X_batch, Y_batch).squeeze(-1)
        loss =  F.binary_cross_entropy_with_logits(logits, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        accumulated_train_loss += loss.item()
        predictions = (torch.sigmoid(logits) > 0.5).float()
        accumulated_train_correct += (predictions == Y_batch).sum().item()
        accumulated_train_total += Y_batch.numel()

        if (training_step + 1) % args["log_interval_steps"] == 0:
            avg_train_loss = accumulated_train_loss / args["log_interval_steps"]
            train_accuracy = accumulated_train_correct / accumulated_train_total
            if args["wandb_log"]: wandb.log({"loss_train": avg_train_loss, "accuracy_train": train_accuracy}, step=training_step)
            accumulated_train_loss = 0.0
            accumulated_train_correct = 0
            accumulated_train_total = 0

        if (training_step + 1) % args["eval_interval_steps"] == 0:
            with torch.no_grad():
                model.eval()
                optimizer.eval()
                eval_losses = []
                correct_predictions = 0
                total_predictions = 0

                # Use DataLoader for evaluation
                for batch_X, batch_Y in eval_loader:
                    batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)

                    logits_eval = model(batch_X, batch_Y).squeeze(-1)
                    loss_eval =  F.binary_cross_entropy_with_logits(logits_eval, batch_Y)
                    eval_losses.append(loss_eval.item())

                    predictions = (torch.sigmoid(logits_eval) > 0.5).float()
                    correct_predictions += (predictions == batch_Y).sum().item()
                    total_predictions += batch_Y.numel()

                avg_eval_loss = np.mean(eval_losses)
                eval_accuracy = correct_predictions / total_predictions
                if args["wandb_log"]: wandb.log({"loss_eval": avg_eval_loss, "accuracy_eval": eval_accuracy}, step=training_step)
                pbar.set_postfix(eval_loss=f"{avg_eval_loss:.4f}", eval_acc=f"{eval_accuracy:.4f}")

                if eval_accuracy > best_eval_accuracy:
                    best_eval_accuracy = eval_accuracy
                    torch.save({
                        'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                        'eval_accuracy': eval_accuracy,
                        'step': training_step,
                        'config': config,
                    }, best_checkpoint_path)

        if (training_step + 1) % args["checkpoint_interval_steps"] == 0:
            model.eval()
            optimizer.eval()
            checkpoint_path = os.path.join(full_checkpoint_dir, "latest.pt")
            torch.save({
                'step': training_step,
                'model_state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
            }, checkpoint_path)

    if args["wandb_log"]: wandb.finish()
