import os
import tomllib
from pathlib import Path
from pprint import pprint

import torch
import wandb
from fastcore.script import Param, bool_arg, call_parse
from fastcore.utils import dict2obj
from schedulefree import AdamWScheduleFree
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader

from metalign.data import FunctionStaticDataset, load_backbone
from metalign.model import TwoLinear, TwoLinearConfig

torch.set_float32_matmul_precision('high')

@call_parse
def main(
    config_file: str = None,  # path to a config file. If provided, any parameters in the file will override the corresponding command line arguments.
    wandb_log: bool_arg = True,  # whether to log the model to wandb. If False, no logging is done.
    wandb_name: str = "metalign",  # name of the wandb project. Used to log the model.
    bias: bool_arg = True,  # whether to use bias in the linear layers of the transformer model
    batch_size: int = 256,  # batch size for training the model
    training_steps: int = 50000,  # number of training steps per epoch
    seed: int = 1234, # random seed for reproducibility
    lr: float = 0.0025,  # learning rate for the optimizer
    weight_decay: float = 0.,  # weight decay for the optimizer
    warmup_steps: int = 0,  # number of warmup steps for the learning rate scheduler
    name: str = None,  # name of the model. If provided, it will be used to log the model.
    log_interval_steps: int = 10,  # log training loss every N steps
    eval_interval_steps: int = 100,  # evaluate the model every eval_interval_steps steps
    num_eval_episodes: int = 128,  # number of episodes to sample for evaluation
    checkpoint_dir: str = "checkpoints", # directory to save checkpoints. this will be placed under data/checkpoints/{name} if name is provided. If name is None, it will be saved under data/checkpoints
    compile: bool_arg = True,  # whether to compile the model with torch.compile
    train_backbone: str = "coco_train_vit_base_patch16_dinov3.lvd1689m",  # backbone for train data. the name must match data/backbone_reps/{train_backbone}.h5
    eval_backbone: str = "coco_eval_vit_base_patch16_dinov3.lvd1689m",  # backbone for eval data. the name must match data/backbone_reps/{eval_backbone}.h5
    train_features: str = "coco_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",  # features for training data. the name must match data/sae/{train_features}.h5
    eval_features: str = "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",  # features for eval data. the name must match data/sae/{eval_features}.h5
    min_nonzero: int = 120,  # minimum number of non-zero activations per column to keep it in the final array
    tags: Param(help="tags to use for the wandb run. If empty, no tags are used.", type=str, nargs="*") = [],
    early_stopping_patience: int = 50, # number of evaluation intervals to wait for improvement before stopping
    early_stopping_min_delta: float = 0.001, # minimum change in evaluation loss to be considered an improvement
    early_stopping_max_steps: int = 1000, # after this, early stopping will be considered even if improvement is still happening
):
    """
    train a two-layer linear model over multi-label classification tasks.

    this is a non-meta learning baseline that learns to map inputs directly to multiple binary targets.
    each input can have multiple correct classes (multi-label classification).
    
    some of the defaults are always overriden by the config file.
    if the config doesn't have some of the fields, the defaults here are used.
    see `data/base_config.toml` for the defaults and `bin/generate_configs.py` for how the config files are generated.
    """

    args = locals()
    if config_file:
        with open(config_file, "rb") as f:
            config_data = tomllib.load(f)
        args.update(config_data)
    args = dict2obj(args)


    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    pprint(args)

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)

    full_checkpoint_dir = f"data/checkpoints/{args.checkpoint_dir}" if args.name is None else f"data/checkpoints/{args.name}"
    if not os.path.exists(full_checkpoint_dir): os.makedirs(full_checkpoint_dir)

    train_inputs = load_backbone(f"data/backbone_reps/{args.train_backbone}.h5")
    eval_inputs = load_backbone(f"data/backbone_reps/{args.eval_backbone}.h5")
    
    feature_dim = train_inputs.shape[1]
    
    train_features_path = Path(f"data/sae/{args.train_features}.h5") if "raw" not in args.train_features else Path(f"data/backbone_reps/{args.train_features}.h5")
    eval_features_path = Path(f"data/sae/{args.eval_features}.h5") if "raw" not in args.eval_features else Path(f"data/backbone_reps/{args.eval_features}.h5")

    train_episode_dataset = FunctionStaticDataset(inputs=train_inputs, features_path=train_features_path, min_nonzero=args.min_nonzero)
    
    eval_episode_dataset = FunctionStaticDataset(inputs=eval_inputs, features_path=eval_features_path, min_nonzero=args.min_nonzero)

    train_loader = DataLoader(
        train_episode_dataset, batch_size=args.batch_size, num_workers=min(4, os.cpu_count()),
        pin_memory=(device == "cuda"), drop_last=True, shuffle=True, persistent_workers=True
    )
    
    eval_loader = DataLoader(
        eval_episode_dataset, batch_size=args.batch_size, num_workers=min(2, os.cpu_count()),
        pin_memory=(device == "cuda"), drop_last=False, shuffle=False, persistent_workers=True
    )

    config = TwoLinearConfig(x_sz=feature_dim, y_sz=train_episode_dataset.Y.shape[1], bias=args.bias)

    model = TwoLinear(c=config)
    model.to(device)
    if args.compile: model = torch.compile(model, fullgraph=True, mode="max-autotune")

    optimizer = AdamWScheduleFree(model.parameters(),lr=args.lr,weight_decay=args.weight_decay, warmup_steps=args.warmup_steps)

    best_eval_loss = float('inf')
    early_stopping_counter = 0
    best_checkpoint_path = f"{full_checkpoint_dir}/model.pt"

    if args.wandb_log:
        device_name = os.uname()[1]
        # use offline mode if running on a juwels bc there compute nodes don't have internet
        wandb.init(project=args.wandb_name, name=args.name, config=args, tags=args.tags, mode="offline" if "juwels" in device_name else "online")
    
    accumulated_train_loss = 0.0
    accumulated_train_f1 = 0.0
    accumulated_train_precision = 0.0
    accumulated_train_recall = 0.0
    accumulated_batches = 0

    train_iterator = iter(train_loader)

    for training_step in range(args.training_steps):
        model.train()
        optimizer.train()

        try:
            X_batch, Y_batch = next(train_iterator)
        except StopIteration:
            # restart iterator when we've gone through all data
            train_iterator = iter(train_loader)
            X_batch, Y_batch = next(train_iterator)
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        with autocast(dtype=torch.bfloat16, device_type=device):
            logits = model(X_batch)
            loss = F.binary_cross_entropy_with_logits(logits, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        accumulated_train_loss += loss.item()
        
        predictions = (torch.sigmoid(logits) > 0.5).float()
        
        y_true_np = Y_batch.cpu().numpy()
        y_pred_np = predictions.cpu().numpy()
        
        batch_f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        batch_precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        batch_recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
        
        accumulated_train_f1 += batch_f1
        accumulated_train_precision += batch_precision
        accumulated_train_recall += batch_recall
        accumulated_batches += 1

        if (training_step + 1) % args.log_interval_steps == 0:
            if args.wandb_log:
                avg_loss = accumulated_train_loss / accumulated_batches
                avg_f1 = accumulated_train_f1 / accumulated_batches
                avg_precision = accumulated_train_precision / accumulated_batches
                avg_recall = accumulated_train_recall / accumulated_batches
                
                wandb.log({
                    "notmeta/loss_train": avg_loss,
                    "notmeta/f1_train": avg_f1,
                    "notmeta/precision_train": avg_precision,
                    "notmeta/recall_train": avg_recall
                }, step=training_step)

            accumulated_train_loss = 0.0
            accumulated_train_f1 = 0.0
            accumulated_train_precision = 0.0
            accumulated_train_recall = 0.0
            accumulated_batches = 0

        if (training_step + 1) % args.eval_interval_steps == 0:
            with torch.no_grad():
                model.eval()
                optimizer.eval()
                
                local_sum_eval_loss = 0.0
                local_eval_f1 = 0.0
                local_eval_precision = 0.0
                local_eval_recall = 0.0
                local_eval_batches = 0

                for batch_X, batch_Y in eval_loader:
                    batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)

                    with autocast(dtype=torch.bfloat16, device_type=device if device != "mps" else "cpu"):
                        logits_eval = model(batch_X)
                        loss_eval = F.binary_cross_entropy_with_logits(logits_eval, batch_Y)
                    
                    local_sum_eval_loss += loss_eval.item()
                    
                    predictions = (torch.sigmoid(logits_eval) > 0.5).float()
                    
                    y_true_np = batch_Y.cpu().numpy()
                    y_pred_np = predictions.cpu().numpy()
                    
                    batch_f1 = f1_score(y_true_np, y_pred_np, average='macro', zero_division=0)
                    batch_precision = precision_score(y_true_np, y_pred_np, average='macro', zero_division=0)
                    batch_recall = recall_score(y_true_np, y_pred_np, average='macro', zero_division=0)
                    
                    local_eval_f1 += batch_f1
                    local_eval_precision += batch_precision
                    local_eval_recall += batch_recall
                    local_eval_batches += 1

                avg_eval_loss = local_sum_eval_loss / local_eval_batches
                avg_eval_f1 = local_eval_f1 / local_eval_batches
                avg_eval_precision = local_eval_precision / local_eval_batches
                avg_eval_recall = local_eval_recall / local_eval_batches

                if args.wandb_log: 
                    wandb.log({
                        "notmeta/loss_eval": avg_eval_loss,
                        "notmeta/f1_eval": avg_eval_f1,
                        "notmeta/precision_eval": avg_eval_precision,
                        "notmeta/recall_eval": avg_eval_recall
                    }, step=training_step)

                if training_step >= args.early_stopping_max_steps:
                    if best_eval_loss - avg_eval_loss > args.early_stopping_min_delta:
                        best_eval_loss = avg_eval_loss
                        early_stopping_counter = 0
                        torch.save({
                            'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                            'eval_loss': avg_eval_loss, 
                            'eval_f1': avg_eval_f1,
                            'eval_precision': avg_eval_precision,
                            'eval_recall': avg_eval_recall,
                            'step': training_step, 'config': config,
                        }, best_checkpoint_path)
                    else: early_stopping_counter += 1
                
                elif avg_eval_loss < best_eval_loss:
                    best_eval_loss = avg_eval_loss
                    torch.save({
                        'state_dict': {k: v.cpu() for k, v in model.state_dict().items()},
                        'eval_loss': avg_eval_loss,
                        'eval_f1': avg_eval_f1,
                        'eval_precision': avg_eval_precision,
                        'eval_recall': avg_eval_recall,
                        'step': training_step, 'config': config,
                    }, best_checkpoint_path)
                
                if early_stopping_counter >= args.early_stopping_patience:
                    print(f"early stopping at step {training_step} due to no improvement in eval loss for {args.early_stopping_patience} evaluation intervals.")
                    break
            
    if args.wandb_log: wandb.finish()
