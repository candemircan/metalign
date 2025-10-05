import os
import tomllib
from pathlib import Path
from pprint import pprint

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

from metalign.data import FunctionDataset, load_backbone
from metalign.model import Transformer, TransformerConfig

torch.set_float32_matmul_precision('high')

@call_parse
def main(
    config_file: str = None,  # path to a config file. If provided, any parameters in the file will override the corresponding command line arguments.
    wandb_log: bool_arg = True,  # whether to log the model to wandb. If False, no logging is done.
    wandb_name: str = "metalign",  # name of the wandb project. Used to log the model.
    n_heads: int = 12,  # number of attention heads in the transformer model
    int_sz: int = 3072,  # size of the intermediate layer in the MLP of the transformer model
    n_layers: int = 6,  # number of transformer layers
    act: str = "gelu",  # activation function for the MLP in the transformer model
    bias: bool_arg = True,  # whether to use bias in the linear layers of the transformer model
    logit_bias: bool_arg = True,  # whether to use bias in the final linear layer of the transformer model. If False, the final layer will not have a bias term.
    attn_drop: float = 0.0,  # dropout rate for the attention layers in the transformer model
    sl: int = 120,  # maximum number of position embeddings in the transformer model, also the sequence length of the input data
    batch_size: int = 256,  # batch size for training the model
    training_steps: int = 50000,  # number of training steps per epoch
    use_mlp: bool_arg = True,  # whether to use an MLP after the transformer layers
    seed: int = 1234, # random seed for reproducibility
    lr: float = 0.0025,  # learning rate for the optimizer
    weight_decay: float = 0.,  # weight decay for the optimizer
    warmup_steps: int = 1000,  # number of warmup steps for the learning rate scheduler
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
    early_stopping_patience: int = 20, # number of evaluation intervals to wait for improvement before stopping
    early_stopping_min_delta: float = 0.001, # minimum change in evaluation loss to be considered an improvement
    early_stopping_max_steps: int = 3000, # after this, early stopping will be considered even if improvement is still happening
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
    
    train_features_path = Path(f"data/sae/{args.train_features}.h5") if "raw" not in args.train_features else Path(f"data/backbone_reps/{args.train_features}.h5")
    eval_features_path = Path(f"data/sae/{args.eval_features}.h5") if "raw" not in args.eval_features else Path(f"data/backbone_reps/{args.eval_features}.h5")

    train_episode_dataset = FunctionDataset(
        inputs=train_inputs, features_path=train_features_path,
        seq_len=args.sl, min_nonzero=args.min_nonzero,
    )
    
    eval_episode_dataset = FunctionDataset(
        inputs=eval_inputs, features_path=eval_features_path,
        seq_len=args.sl, min_nonzero=args.min_nonzero,
        epoch_size=args.num_eval_episodes
    )

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

    config = TransformerConfig(
        x_sz=feature_dim, n_heads=args.n_heads,
        int_sz=args.int_sz, n_layers=args.n_layers,
        act=args.act, bias=args.bias, logit_bias=args.logit_bias,
        attn_drop=args.attn_drop, sl=args.sl, use_mlp=args.use_mlp
    )

    model = Transformer(config)
    model.to(device)
    if args.compile: model = torch.compile(model, fullgraph=True, mode="max-autotune")
    model = DDP(model, device_ids=[ddp_local_rank])


    optimizer = AdamWScheduleFree(model.parameters(),lr=args.lr,weight_decay=args.weight_decay, warmup_steps=args.warmup_steps)

    best_eval_loss = float('inf')
    early_stopping_counter = 0
    if ddp_rank == 0: best_checkpoint_path = f"{full_checkpoint_dir}/model.pt"

    if args.wandb_log and ddp_rank == 0:
        device_name = os.uname()[1]
        # use offline mode if running on a juwels bc there compute nodes don't have internet
        wandb.init(project=args.wandb_name, name=args.name, config=args, tags=args.tags, mode="offline" if "juwels" in device_name else "online")
    
    accumulated_train_loss = 0.0
    accumulated_train_correct = 0
    accumulated_train_total = 0

    train_iterator = iter(train_loader)

    for training_step in range(args.training_steps):
        model.train()
        optimizer.train()

        X_batch, Y_batch = next(train_iterator)
        X_batch = X_batch.to(device)
        Y_batch = Y_batch.to(device)

        with autocast(dtype=torch.bfloat16, device_type="cuda"):
            logits = model(X_batch, Y_batch).squeeze(-1)
            loss = F.binary_cross_entropy_with_logits(logits, Y_batch)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        accumulated_train_loss += loss.item()
        predictions = (torch.sigmoid(logits) > 0.5).float()
        accumulated_train_correct += (predictions == Y_batch).sum().item()
        accumulated_train_total += Y_batch.numel()

        if (training_step + 1) % args.log_interval_steps == 0:
            metrics = torch.tensor([accumulated_train_loss, accumulated_train_correct, accumulated_train_total], device=device)
            torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

            if args.wandb_log and ddp_rank == 0:
                global_loss = metrics[0].item() / (args.log_interval_steps * ddp_world_size)
                global_accuracy = metrics[1].item() / metrics[2].item()
                
                wandb.log({"loss_train": global_loss,"accuracy_train": global_accuracy}, step=training_step)

            accumulated_train_loss = 0.0
            accumulated_train_correct = 0
            accumulated_train_total = 0

        if (training_step + 1) % args.eval_interval_steps == 0:
            with torch.no_grad():
                model.eval()
                optimizer.eval()
                
                local_sum_eval_loss = 0.0
                local_correct_predictions = 0
                local_total_predictions = 0

                for batch_X, batch_Y in eval_loader:
                    batch_X = batch_X.to(device)
                    batch_Y = batch_Y.to(device)

                    with autocast(dtype=torch.bfloat16, device_type="cuda"):
                        logits_eval = model(batch_X, batch_Y).squeeze(-1)
                        loss_eval = F.binary_cross_entropy_with_logits(logits_eval, batch_Y)
                    
                    local_sum_eval_loss += loss_eval.item() * batch_Y.numel()
                    predictions = (torch.sigmoid(logits_eval) > 0.5).float()
                    local_correct_predictions += (predictions == batch_Y).sum().item()
                    local_total_predictions += batch_Y.numel()

                metrics = torch.tensor([local_sum_eval_loss, local_correct_predictions, local_total_predictions], device=device)
                torch.distributed.all_reduce(metrics, op=torch.distributed.ReduceOp.SUM)

                stop_training = torch.tensor(0, device=device)
                if ddp_rank == 0:
                    global_sum_loss = metrics[0].item()
                    global_correct = metrics[1].item()
                    global_total = metrics[2].item()
                    eval_accuracy = global_correct / global_total
                    avg_eval_loss = global_sum_loss / global_total

                    if args.wandb_log: wandb.log({"loss_eval": avg_eval_loss,"accuracy_eval": eval_accuracy}, step=training_step)

                    if training_step >= args.early_stopping_max_steps:
                        if best_eval_loss - avg_eval_loss > args.early_stopping_min_delta:
                            best_eval_loss = avg_eval_loss
                            early_stopping_counter = 0
                            torch.save({
                                'state_dict': {k: v.cpu() for k, v in model.module.state_dict().items()},
                                'eval_loss': avg_eval_loss, 'eval_accuracy': eval_accuracy,
                                'step': training_step, 'config': config,
                            }, best_checkpoint_path)
                        else: early_stopping_counter += 1
                    
                    elif avg_eval_loss < best_eval_loss:
                        best_eval_loss = avg_eval_loss
                        torch.save({
                            'state_dict': {k: v.cpu() for k, v in model.module.state_dict().items()},
                            'eval_loss': avg_eval_loss, 'eval_accuracy': eval_accuracy,
                            'step': training_step, 'config': config,
                        }, best_checkpoint_path)
                    
                    if early_stopping_counter >= args.early_stopping_patience:
                        print(f"early stopping at step {training_step} due to no improvement in eval loss for {args.early_stopping_patience} evaluation intervals.")
                        stop_training.fill_(1)

            torch.distributed.broadcast(stop_training, src=0)
            if stop_training.item() == 1: break
            
    torch.distributed.destroy_process_group()
    if args.wandb_log and ddp_rank == 0: wandb.finish()