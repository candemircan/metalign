import os
import tomllib
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import wandb
from fastcore.script import Param, bool_arg, call_parse
from fastcore.utils import dict2obj
from schedulefree import AdamWScheduleFree
from torch.amp import autocast
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from metalign.data import FunctionDataset, ImageFunctionDataset, h5_to_np, load_backbone
from metalign.model import MetaLearnerConfig, get_model
from metalign.utils import calc_r2

torch.set_float32_matmul_precision('high')

@call_parse
def main(
    config_file: str = None,  # path to a config file. If provided, any parameters in the file will override the corresponding command line arguments.
    track_log: bool_arg = True,  # whether to log the model. If False, no logging is done.
    project_name: str = "metalign",  # name of the project. Used to log the model.
    n_heads: int = 12,  # number of attention heads in the transformer model
    int_sz: int = 3072,  # size of the intermediate layer in the MLP of the transformer model
    n_layers: int = 6,  # number of transformer layers
    act: str = "gelu",  # activation function for the MLP in the transformer model
    bias: bool_arg = True,  # whether to use bias in the linear layers of the transformer model
    logit_bias: bool_arg = True,  # whether to use bias in the final linear layer of the transformer model. If False, the final layer will not have a bias term.
    attn_drop: float = 0.0,  # dropout rate for the attention layers in the transformer model
    sl: int = 120,  # maximum number of position embeddings in the transformer model, also the sequence length of the input data
    batch_size: int = 256,  # batch size for training the model
    training_steps: int = 10000,  # number of training steps
    use_mlp: bool_arg = True,  # whether to use an MLP after the transformer layers
    seed: int = 1234, # random seed for reproducibility
    lr: float = 0.0025,  # learning rate for the optimizer
    weight_decay: float = 0.,  # weight decay for the optimizer
    warmup_steps: int = 0,  # number of warmup steps for the learning rate scheduler
    name: str = None,  # name of the model. If provided, it will be used to log the model.
    log_interval_steps: int = 10,  # log training loss every N steps
    eval_interval_steps: int = 100,  # evaluate the model every eval_interval_steps steps
    eval_episodes_per_dim: int = 20,  # number of episodes to sample per function dimension for evaluation
    checkpoint_dir: str = "checkpoints", # directory to save checkpoints. this will be placed under data/checkpoints/{name} if name is provided. If name is None, it will be saved under data/checkpoints
    compile: bool_arg = True,  # whether to compile the model with torch.compile
    train_backbone: str = "coco_train_vit_base_patch16_dinov3.lvd1689m",  # backbone for train data. the name must match data/backbone_reps/{train_backbone}.h5
    eval_backbone: str = "coco_eval_vit_base_patch16_dinov3.lvd1689m",  # backbone for eval data. the name must match data/backbone_reps/{eval_backbone}.h5
    train_features: str = "coco_train_sae-top_k-64-cls_only-layer_11-hook_resid_post",  # features for training data. the name must match data/sae/{train_features}.h5
    eval_features: str = "coco_eval_sae-top_k-64-cls_only-layer_11-hook_resid_post",  # features for eval data. the name must match data/sae/{eval_features}.h5
    min_nonzero: int = 120,  # minimum number of non-zero activations per column to keep it in the final array
    tags: Param(help="tags to use for the run. If empty, no tags are used.", type=str, nargs="*") = [],
    early_stopping_patience: int = 10, # number of evaluation intervals to wait for improvement before stopping
    early_stopping_min_delta: float = 0.001, # minimum change in evaluation loss to be considered an improvement
    early_stopping_max_steps: int = 1000, # after this, early stopping will be considered even if improvement is still happening
    early_stopping_acc_threshold: float = 0.6, # early stopping max_steps is only applied after eval accuracy exceeds this threshold
    no_embed: bool_arg = False,  # if True, skip linear embed; backbone features pass through unchanged, only prev magnitude projected (1 -> x_sz) and added
    model_type: str = "transformer",  # model architecture: "transformer", "lstm", or "static_linear"
):
    """
    train a meta-learning transformer model over function learning tasks.

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
    
    assert torch.cuda.is_available(), "DDP requires CUDA"
    torch.distributed.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)

    if ddp_rank == 0: pprint(args)

    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(args.seed)

    if ddp_rank == 0:
        full_checkpoint_dir = f"data/checkpoints/{args.checkpoint_dir}" if args.name is None else f"data/checkpoints/{args.name}"
        if not os.path.exists(full_checkpoint_dir): os.makedirs(full_checkpoint_dir)

    train_inputs = load_backbone(f"data/backbone_reps/{args.train_backbone}.h5")
    eval_inputs = load_backbone(f"data/backbone_reps/{args.eval_backbone}.h5")
    
    feature_dim = train_inputs.shape[1]
    
    train_features_path = Path(f"data/backbone_reps/{args.train_features}.h5") if "raw" in args.train_features else Path(f"data/sae/{args.train_features}.h5")
    eval_features_path = Path(f"data/backbone_reps/{args.eval_features}.h5") if "raw" in args.eval_features else Path(f"data/sae/{args.eval_features}.h5")

    is_static_linear = args.model_type == "static_linear"

    if is_static_linear:
        Y_train, col_indices = h5_to_np(train_features_path, min_nonzero=args.min_nonzero, return_col_indices=True)
        Y_eval_full = h5_to_np(eval_features_path, min_nonzero=0)
        if len(col_indices) and col_indices.max() >= Y_eval_full.shape[1]:
            Y_eval_full = np.pad(Y_eval_full, ((0, 0), (0, col_indices.max() + 1 - Y_eval_full.shape[1])))
        Y_eval = Y_eval_full[:, col_indices]

        num_dims = Y_train.shape[1]
        Y_train_t = torch.from_numpy(Y_train)
        is_sparse = bool((Y_train_t == 0).float().mean() > 0.5)
        mag_std = 1.0  # not used in StaticLinear (no prev-magnitude input)

        train_episode_dataset = ImageFunctionDataset(train_inputs, Y_train, is_sparse=is_sparse)
        eval_episode_dataset  = ImageFunctionDataset(
            eval_inputs, Y_eval, is_sparse=is_sparse,
            medians=train_episode_dataset.medians if not is_sparse else None,
            mag_stds=train_episode_dataset.mag_stds,
        )
    else:
        num_dims = 1
        train_episode_dataset = FunctionDataset(
            inputs=train_inputs, features_path=train_features_path,
            seq_len=args.sl, min_nonzero=args.min_nonzero,
        )
        is_sparse = train_episode_dataset.is_sparse

        # compute magnitude std from training data
        # for sparse: std of non-zero values (magnitudes are raw activations)
        # for dense: std of (value - median) distances across all values
        if is_sparse:
            nonzero_mags = train_episode_dataset.Y[train_episode_dataset.Y > 0]
            mag_std = nonzero_mags.std().item()
        else:
            distances = train_episode_dataset.Y - train_episode_dataset.medians.unsqueeze(0)
            mag_std = distances.std().item()

        eval_episode_dataset = FunctionDataset(
            inputs=eval_inputs, features_path=eval_features_path,
            seq_len=args.sl, min_nonzero=args.min_nonzero,
            episodes_per_dim=args.eval_episodes_per_dim,
        )

    if ddp_rank == 0: print(f"Magnitude std: {mag_std:.4f} (sparse={is_sparse})")

    per_device_batch_size = args.batch_size // ddp_world_size
    
    train_sampler = DistributedSampler(train_episode_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True)
    eval_sampler = DistributedSampler(eval_episode_dataset, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False)

    train_loader = DataLoader(
        train_episode_dataset, batch_size=per_device_batch_size, num_workers=min(4, os.cpu_count()),
        pin_memory=True, drop_last=True, sampler=train_sampler, persistent_workers=True
    )
    
    eval_loader = DataLoader(
        eval_episode_dataset, batch_size=per_device_batch_size, num_workers=min(2, os.cpu_count()),
        pin_memory=True, drop_last=False, sampler=eval_sampler, persistent_workers=True
    )

    config = MetaLearnerConfig(
        x_sz=feature_dim, n_heads=args.n_heads,
        int_sz=args.int_sz, n_layers=args.n_layers,
        act=args.act, bias=args.bias, logit_bias=args.logit_bias,
        attn_drop=args.attn_drop, sl=args.sl, use_mlp=args.use_mlp,
        mag_std=mag_std, no_embed=args.no_embed,
        num_dims=num_dims, model_type=args.model_type,
    )

    model = get_model(config)
    model.to(device)
    if args.compile: model = torch.compile(model, fullgraph=True, mode="max-autotune")
    model = DDP(model, device_ids=[ddp_local_rank])


    # Exclude initial embedding from weight decay (match 'embed.' anywhere to handle DDP/compile prefixes)
    embed_params = [p for n, p in model.named_parameters() if 'embed.' in n]
    other_params = [p for n, p in model.named_parameters() if 'embed.' not in n]
    optimizer = AdamWScheduleFree([
        {'params': embed_params, 'weight_decay': 0.0},
        {'params': other_params, 'weight_decay': args.weight_decay}
    ], lr=args.lr, warmup_steps=args.warmup_steps)

    best_eval_score = float('-inf')
    early_stopping_counter = 0
    if ddp_rank == 0: best_checkpoint_path = f"{full_checkpoint_dir}/model.pt"

    if args.track_log and ddp_rank == 0:
        wandb.init(project=args.project_name, config=args, name=args.name, tags=args.tags, mode="offline")
        wandb.config.update({"mag_std": mag_std, "is_sparse": is_sparse})
    
    accumulated_train_loss = 0.0
    accumulated_train_loss_binary = 0.0
    accumulated_train_loss_magnitude = 0.0
    accumulated_train_correct = 0
    accumulated_train_total = 0
    accumulated_sigma_ce = 0.0
    accumulated_sigma_mse = 0.0

    train_iterator = iter(train_loader)

    for training_step in range(args.training_steps):
        model.train()
        optimizer.train()

        try: X_batch, Y_binary, Y_magnitude = next(train_iterator)
        except StopIteration:  # ImageFunctionDataset is finite; restart with new shuffle
            train_sampler.set_epoch(training_step)
            train_iterator = iter(train_loader)
            X_batch, Y_binary, Y_magnitude = next(train_iterator)

        X_batch = X_batch.to(device)
        Y_binary, Y_magnitude = Y_binary.to(device), Y_magnitude.to(device)

        with autocast(dtype=torch.bfloat16, device_type="cuda"):
            if is_static_linear:
                binary_logits, magnitude_preds = model(X_batch)
            else:
                binary_logits, magnitude_preds = model(X_batch, Y_magnitude)
                binary_logits   = binary_logits.squeeze(-1)
                magnitude_preds = magnitude_preds.squeeze(-1)

            loss_binary = F.binary_cross_entropy_with_logits(binary_logits, Y_binary)

            # for sparse: only compute magnitude loss on positive examples (negatives are always 0)
            # for dense: compute on all examples (magnitude is signed distance from median)
            if is_sparse:
                nonzero_mask = Y_binary == 1
                if nonzero_mask.sum() > 0: loss_magnitude = F.mse_loss(magnitude_preds[nonzero_mask], Y_magnitude[nonzero_mask])
                else: loss_magnitude = torch.tensor(0.0, device=device)
            else:
                loss_magnitude = F.mse_loss(magnitude_preds, Y_magnitude)

            # Kendall uncertainty weighting
            loss, sigma_ce, sigma_mse = model.module.compute_loss(loss_binary, loss_magnitude)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        accumulated_train_loss += loss.item()
        accumulated_train_loss_binary += loss_binary.item()
        accumulated_train_loss_magnitude += loss_magnitude.item()
        accumulated_sigma_ce += sigma_ce.item()
        accumulated_sigma_mse += sigma_mse.item()

        predictions = (torch.sigmoid(binary_logits) > 0.5).float()
        accumulated_train_correct += (predictions == Y_binary).sum().item()
        accumulated_train_total += Y_binary.numel()

        if (training_step + 1) % args.log_interval_steps == 0:
            # [loss_total, correct, total, loss_binary, loss_magnitude, sigma_ce, sigma_mse]
            metrics = torch.tensor([
                accumulated_train_loss,
                accumulated_train_correct,
                accumulated_train_total,
                accumulated_train_loss_binary,
                accumulated_train_loss_magnitude,
                accumulated_sigma_ce,
                accumulated_sigma_mse
            ], device=device)
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

            if args.track_log and ddp_rank == 0:
                steps = args.log_interval_steps * ddp_world_size
                global_loss = metrics[0].item() / steps
                global_accuracy = metrics[1].item() / metrics[2].item()
                global_loss_binary = metrics[3].item() / steps
                global_loss_magnitude = metrics[4].item() / steps
                global_sigma_ce = metrics[5].item() / steps
                global_sigma_mse = metrics[6].item() / steps

                log_dict = {
                    "train/loss": global_loss,
                    "train/accuracy": global_accuracy,
                    "train/loss_binary": global_loss_binary,
                    "train/loss_magnitude": global_loss_magnitude,
                    "uncertainty/sigma_ce": global_sigma_ce,
                    "uncertainty/sigma_mse": global_sigma_mse,
                }

                wandb.log(log_dict, step=training_step)

            accumulated_train_loss = 0.0
            accumulated_train_loss_binary = 0.0
            accumulated_train_loss_magnitude = 0.0
            accumulated_train_correct = 0
            accumulated_train_total = 0
            accumulated_sigma_ce = 0.0
            accumulated_sigma_mse = 0.0

        if (training_step + 1) % args.eval_interval_steps == 0:
            with torch.no_grad():
                model.eval()
                optimizer.eval()

                local_sum_eval_loss = 0.0
                local_sum_eval_loss_binary = 0.0
                local_sum_eval_loss_magnitude = 0.0
                local_sum_eval_r2 = 0.0
                local_correct_predictions = 0
                local_total_predictions = 0
                local_total_nonzero_predictions = 0

                for batch_X, batch_Y_binary, batch_Y_magnitude in eval_loader:
                    batch_X = batch_X.to(device)
                    batch_Y_binary, batch_Y_magnitude = batch_Y_binary.to(device), batch_Y_magnitude.to(device)

                    with autocast(dtype=torch.bfloat16, device_type="cuda"):
                        if is_static_linear:
                            binary_logits_eval, magnitude_preds_eval = model(batch_X)
                        else:
                            binary_logits_eval, magnitude_preds_eval = model(batch_X, batch_Y_magnitude)
                            binary_logits_eval   = binary_logits_eval.squeeze(-1)
                            magnitude_preds_eval = magnitude_preds_eval.squeeze(-1)

                        loss_binary_eval = F.binary_cross_entropy_with_logits(binary_logits_eval, batch_Y_binary)

                        # for sparse: only compute magnitude loss/r2 on positive examples
                        # for dense: compute on all examples
                        if is_sparse:
                            nonzero_mask_eval = batch_Y_binary == 1
                            n_nonzero = nonzero_mask_eval.sum().item()

                            if n_nonzero > 0:
                                loss_magnitude_eval = F.mse_loss(magnitude_preds_eval[nonzero_mask_eval], batch_Y_magnitude[nonzero_mask_eval])
                                r2_eval = calc_r2(magnitude_preds_eval[nonzero_mask_eval], batch_Y_magnitude[nonzero_mask_eval])
                            else:
                                loss_magnitude_eval = torch.tensor(0.0, device=device)
                                r2_eval = 0.0
                        else:
                            n_nonzero = batch_Y_magnitude.numel()
                            loss_magnitude_eval = F.mse_loss(magnitude_preds_eval, batch_Y_magnitude)
                            r2_eval = calc_r2(magnitude_preds_eval, batch_Y_magnitude)

                        # Kendall uncertainty weighting
                        loss_eval, _, _ = model.module.compute_loss(loss_binary_eval, loss_magnitude_eval)

                    local_sum_eval_loss += loss_eval.item() * batch_Y_binary.numel() # weighted sum
                    local_sum_eval_loss_binary += loss_binary_eval.item() * batch_Y_binary.numel()
                    local_sum_eval_loss_magnitude += loss_magnitude_eval.item() * batch_Y_binary.numel() 
                    
                    if n_nonzero > 0:
                        local_sum_eval_r2 += r2_eval * n_nonzero
                        local_total_nonzero_predictions += n_nonzero

                    predictions = (torch.sigmoid(binary_logits_eval) > 0.5).float()
                    local_correct_predictions += (predictions == batch_Y_binary).sum().item()
                    local_total_predictions += batch_Y_binary.numel()

                # [loss_total, correct, total, loss_binary, loss_magnitude, r2_weighted_sum, total_nonzero]
                metrics = torch.tensor([
                    local_sum_eval_loss, 
                    local_correct_predictions, 
                    local_total_predictions,
                    local_sum_eval_loss_binary,
                    local_sum_eval_loss_magnitude,
                    local_sum_eval_r2,
                    local_total_nonzero_predictions
                ], device=device)
                torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

                stop_training = torch.tensor(0, device=device)
                if ddp_rank == 0:
                    global_total = metrics[2].item()
                    global_sum_loss = metrics[0].item()
                    global_correct = metrics[1].item()
                    global_sum_loss_binary = metrics[3].item()
                    global_sum_loss_magnitude = metrics[4].item()
                    global_sum_r2 = metrics[5].item()
                    global_total_nonzero = metrics[6].item()
                    
                    eval_accuracy = global_correct / global_total
                    avg_eval_loss = global_sum_loss / global_total
                    avg_eval_loss_binary = global_sum_loss_binary / global_total
                    avg_eval_loss_magnitude = global_sum_loss_magnitude / global_total

                    avg_eval_r2 = global_sum_r2 / global_total_nonzero if global_total_nonzero > 0 else 0.0
                    
                    # Combined Score: Average of Accuracy and R2
                    eval_score = 0.5 * eval_accuracy + 0.5 * avg_eval_r2

                    log_dict = {
                        "eval/loss": avg_eval_loss,
                        "eval/accuracy": eval_accuracy,
                        "eval/loss_binary": avg_eval_loss_binary,
                        "eval/loss_magnitude": avg_eval_loss_magnitude,
                        "eval/r2": avg_eval_r2,
                    }

                    if args.track_log: wandb.log(log_dict, step=training_step)

                    if training_step >= args.early_stopping_max_steps and eval_accuracy >= args.early_stopping_acc_threshold:
                        if eval_score - best_eval_score > args.early_stopping_min_delta:
                            best_eval_score = eval_score
                            early_stopping_counter = 0
                            torch.save({
                                'state_dict': {k: v.cpu() for k, v in model.module.state_dict().items()},
                                'eval_accuracy': eval_accuracy, 'eval_r2': avg_eval_r2,
                                'eval_loss': avg_eval_loss,
                                'step': training_step, 'config': config,
                            }, best_checkpoint_path)
                        else: early_stopping_counter += 1

                    elif eval_score > best_eval_score:
                        best_eval_score = eval_score
                        torch.save({
                            'state_dict': {k: v.cpu() for k, v in model.module.state_dict().items()},
                            'eval_accuracy': eval_accuracy, 'eval_r2': avg_eval_r2,
                            'eval_loss': avg_eval_loss,
                            'step': training_step, 'config': config,
                        }, best_checkpoint_path)
                    
                    if early_stopping_counter >= args.early_stopping_patience:
                        print(f"early stopping at step {training_step} due to no improvement in eval score for {args.early_stopping_patience} evaluation intervals.")
                        stop_training.fill_(1)

            torch.distributed.broadcast(stop_training, src=0)
            if stop_training.item() == 1: break
            
    torch.distributed.destroy_process_group()
    if args.track_log and ddp_rank == 0: wandb.finish()
