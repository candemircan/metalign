import os
import tomllib
from pathlib import Path

import torch
import wandb
from fastcore.script import Param, bool_arg, call_parse
from fastcore.utils import dict2obj
from schedulefree import AdamWScheduleFree
from torch.amp import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import auroc, average_precision

from metalign.data import FunctionStaticDataset, load_data_mmap
from metalign.model import TwoLinear, TwoLinearConfig

torch.set_float32_matmul_precision('high')

def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    p = torch.sigmoid(inputs)
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    if reduction == "mean": loss = loss.mean()
    elif reduction == "sum":loss = loss.sum()
        
    return loss




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
    weight_decay: float = 1e-5,  # weight decay (L2 regularization) for the optimizer.
    warmup_steps: int = 0,  # number of warmup steps for the learning rate scheduler
    name: str = None,  # name of the model. If provided, it will be used to log the model.
    log_interval_steps: int = 10,  # log training loss every N steps
    eval_interval_steps: int = 100,  # evaluate the model every eval_interval_steps steps
    checkpoint_dir: str = "checkpoints", # directory to save checkpoints. this will be placed under data/checkpoints/{name} if name is provided. If name is None, it will be saved under data/checkpoints
    compile: bool_arg = True,  # whether to compile the model with torch.compile
    train_backbone: str = "coco_train_vit_base_patch16_dinov3.lvd1689m",  # backbone for train data. the name must match data/backbone_reps/{train_backbone}.h5
    eval_backbone: str = "coco_eval_vit_base_patch16_dinov3.lvd1689m",  # backbone for eval data. the name must match data/backbone_reps/{eval_backbone}.h5
    train_features: str = "coco_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",  # features for training data. the name must match data/sae/{train_features}.h5
    eval_features: str = "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",  # features for eval data. the name must match data/sae/{eval_features}.h5
    min_nonzero: int = 120,  # minimum number of non-zero activations per column to keep it in the final array
    tags: Param(help="tags to use for the wandb run. If empty, no tags are used.", type=str, nargs="*") = [],
    early_stopping_patience: int = 50, # number of evaluation intervals to wait for improvement before stopping
    early_stopping_min_delta: float = 0.005, # minimum change in evaluation mAP to be considered an improvement (0.5%)
    early_stopping_max_steps: int = 1000, # after this, early stopping will be considered even if improvement is still happening
    use_focal_loss: bool_arg = True, # whether to use Focal Loss instead of weighted BCE
    focal_loss_alpha: float = 0.25, # alpha parameter for Focal Loss
    focal_loss_gamma: float = 2.0, # gamma parameter for Focal Loss
    use_class_weights: bool_arg = True, # whether to use class weights for imbalanced data (only if use_focal_loss is False)
    pos_weight_factor: float = 1.0, # multiplicative factor for positive class weights
):
    """
    Train a two-layer linear model over multi-label classification tasks.
    Uses threshold-free metrics (mAP, AUC) for both training and evaluation monitoring.
    """
    args = locals()
    if config_file:
        with open(config_file, "rb") as f:
            config_data = tomllib.load(f)
        args.update(config_data)
    args = dict2obj(args)

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(args, flush=True)

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(args.seed)

    full_checkpoint_dir = f"data/checkpoints/{args.checkpoint_dir}" if args.name is None else f"data/checkpoints/{args.name}"
    os.makedirs(full_checkpoint_dir, exist_ok=True)

    # get paths instead of loading data
    train_inputs_path = Path(f"data/backbone_reps/{args.train_backbone}.h5")
    eval_inputs_path = Path(f"data/backbone_reps/{args.eval_backbone}.h5")
    
    # determine feature_dim using mmap (fast)
    train_inputs_mmap = load_data_mmap(train_inputs_path)
    feature_dim = train_inputs_mmap.shape[1]
    
    train_features_path = Path(f"data/sae/{args.train_features}.h5") if "raw" not in args.train_features else Path(f"data/backbone_reps/{args.train_features}.h5")
    eval_features_path = Path(f"data/sae/{args.eval_features}.h5") if "raw" not in args.eval_features else Path(f"data/backbone_reps/{args.eval_features}.h5")

    train_episode_dataset = FunctionStaticDataset(inputs_path=train_inputs_path, features_path=train_features_path, min_nonzero=args.min_nonzero)
    eval_episode_dataset = FunctionStaticDataset(inputs_path=eval_inputs_path, features_path=eval_features_path, min_nonzero=args.min_nonzero, valid_columns=train_episode_dataset.valid_columns)

    print(f"train dataset: {train_episode_dataset.n_samples} samples, {train_episode_dataset.num_functions} functions", flush=True)
    print(f"eval dataset: {eval_episode_dataset.n_samples} samples, {eval_episode_dataset.num_functions} functions", flush=True)
    assert train_episode_dataset.num_functions == eval_episode_dataset.num_functions, "train and eval datasets must have the same number of functions"

    pos_weights = None
    if not args.use_focal_loss and args.use_class_weights:
        pos_freq = train_episode_dataset.Y.float().mean(dim=0)
        pos_weights = ((1 - pos_freq) / pos_freq) * args.pos_weight_factor
        pos_weights = pos_weights.to(device)

    train_loader = DataLoader(
        train_episode_dataset, batch_size=args.batch_size, num_workers=min(4, os.cpu_count()),
        pin_memory=(device == "cuda"), drop_last=True, shuffle=True, persistent_workers=True
    )
    eval_loader = DataLoader(
        eval_episode_dataset, batch_size=args.batch_size, num_workers=min(2, os.cpu_count()),
        pin_memory=(device == "cuda"), drop_last=False, shuffle=False, persistent_workers=True
    )

    config = TwoLinearConfig(x_sz=feature_dim, y_sz=train_episode_dataset.Y.shape[1], bias=args.bias)
    model = TwoLinear(c=config).to(device)
    if args.compile: model = torch.compile(model, fullgraph=True, mode="max-autotune")

    optimizer = AdamWScheduleFree(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, warmup_steps=args.warmup_steps)

    # Early stopping now tracks mean Average Precision (mAP)
    best_eval_map = 0.0
    early_stopping_counter = 0
    best_checkpoint_path = f"{full_checkpoint_dir}/model.pt"

    if args.wandb_log:
        device_name = os.uname()[1]
        wandb.init(project=args.wandb_name, name=args.name, config=args, tags=args.tags, mode="offline" if "juwels" in device_name else "online")
    
    accumulated_train_loss, accumulated_train_map, accumulated_train_auc, accumulated_batches = 0.0, 0.0, 0.0, 0
    train_iterator = iter(train_loader)

    for training_step in range(args.training_steps):
        model.train()
        optimizer.train()

        try:
            X_batch, Y_batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            X_batch, Y_batch = next(train_iterator)
        
        X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)

        with autocast(dtype=torch.bfloat16, device_type=device):
            logits = model(X_batch)
            if args.use_focal_loss:
                loss = sigmoid_focal_loss(logits, Y_batch, alpha=args.focal_loss_alpha, gamma=args.focal_loss_gamma)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, Y_batch, pos_weight=pos_weights)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        accumulated_train_loss += loss.item()
        
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            accumulated_train_map += average_precision(probs, Y_batch.long(), task="multilabel", num_labels=Y_batch.shape[1], average="micro").item()
            accumulated_train_auc += auroc(probs, Y_batch.long(), task="multilabel", num_labels=Y_batch.shape[1], average="micro").item()
        accumulated_batches += 1

        if (training_step + 1) % args.log_interval_steps == 0 and accumulated_batches > 0:
            if args.wandb_log:
                wandb.log({
                    "notmeta/loss_train": accumulated_train_loss / accumulated_batches,
                    "notmeta/mAP_train": accumulated_train_map / accumulated_batches,
                    "notmeta/AUC_train": accumulated_train_auc / accumulated_batches,
                }, step=training_step)
            accumulated_train_loss, accumulated_train_map, accumulated_train_auc, accumulated_batches = 0.0, 0.0, 0.0, 0

        if (training_step + 1) % args.eval_interval_steps == 0:
            with torch.no_grad():
                model.eval()
                optimizer.eval()
                
                sum_eval_loss, sum_eval_map, sum_eval_auc = 0.0, 0.0, 0.0
                eval_batches = 0

                for batch_X, batch_Y in eval_loader:
                    batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)

                    with autocast(dtype=torch.bfloat16, device_type=device if device != "mps" else "cpu"):
                        logits_eval = model(batch_X)
                        loss_eval = sigmoid_focal_loss(logits_eval, batch_Y, alpha=args.focal_loss_alpha, gamma=args.focal_loss_gamma) if args.use_focal_loss else F.binary_cross_entropy_with_logits(logits_eval, batch_Y, pos_weight=pos_weights)
                    
                    sum_eval_loss += loss_eval.item()
                    
                    probs_eval = torch.sigmoid(logits_eval)
                    sum_eval_map += average_precision(probs_eval, batch_Y.long(), task="multilabel", num_labels=batch_Y.shape[1], average="micro").item()
                    sum_eval_auc += auroc(probs_eval, batch_Y.long(), task="multilabel", num_labels=batch_Y.shape[1], average="micro").item()
                    eval_batches += 1

                avg_eval_loss = sum_eval_loss / eval_batches
                avg_eval_map = sum_eval_map / eval_batches
                avg_eval_auc = sum_eval_auc / eval_batches

                if args.wandb_log: 
                    wandb.log({
                        "notmeta/loss_eval": avg_eval_loss,
                        "notmeta/mAP_eval": avg_eval_map,
                        "notmeta/AUC_eval": avg_eval_auc
                    }, step=training_step)

                is_improvement = (avg_eval_map - best_eval_map) > args.early_stopping_min_delta
                
                if training_step >= args.early_stopping_max_steps:
                    if is_improvement:
                        best_eval_map = avg_eval_map
                        early_stopping_counter = 0
                        torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}, 'eval_mAP': avg_eval_map, 'step': training_step, 'config': config}, best_checkpoint_path)
                    else:
                        early_stopping_counter += 1
                elif avg_eval_map > best_eval_map:
                    best_eval_map = avg_eval_map
                    early_stopping_counter = 0 
                    torch.save({'state_dict': {k: v.cpu() for k, v in model.state_dict().items()}, 'eval_mAP': avg_eval_map, 'step': training_step, 'config': config}, best_checkpoint_path)
                else:
                    early_stopping_counter += 1
                
                if early_stopping_counter >= args.early_stopping_patience:
                    print(f"Early stopping at step {training_step} due to no improvement in eval mAP for {args.early_stopping_patience} evaluation intervals.", flush=True)
                    break
            
    if args.wandb_log: wandb.finish()