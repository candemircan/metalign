import json
from glob import glob
from pathlib import Path

import pandas as pd
import torch
from fastcore.script import call_parse
from nnsight import NNsight
from tqdm import tqdm

from metalign.data import load_backbone
from metalign.model import Transformer, TwoLinear


@call_parse
def main(
    experiment_name: str, # has to be one of main, raw, midsae
    backbone_name: str, # has to be one of mae, clip, siglip2, dinov3
    force: bool = False # if True, will overwrite existing eval files
    ):

    eval_path = Path("data/evals/brain")
    eval_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{experiment_name}_{backbone_name}"
    eval_file = eval_path / f"{file_name}.csv"

    if eval_file.exists() and not force:
        print(f"Eval file {eval_file} already exists. Use --force to overwrite.")
        return
    
    best_models = json.load(open(Path("data/checkpoints") / "best_models.json"))
    backbone_dict = json.load(open(Path("data/backbone_reps") / "backbones.json"))
    ckpt = best_models[f"[{experiment_name.upper()}]"][backbone_name]
    things_reps = f"data/backbone_reps/things_{backbone_dict[backbone_name]}.h5"

    ckpt = torch.load(ckpt,  weights_only=False)
    config, state_dict = ckpt['config'], fix_state_dict(ckpt['state_dict'])
    
    human_data = pd.read_csv("data/external/category_learning.csv")
    backbone_reps = load_backbone(things_reps)
    
    if experiment_name.lower() == 'notmeta':
        model = TwoLinear(c=config)
        model.load_state_dict(state_dict)
        model.eval()
        # for TwoLinear, we extract representations from the embed layer
        metalign_reps = model.embed(torch.from_numpy(backbone_reps))
    else:
        model = Transformer(c=config)
        model.load_state_dict(state_dict)
        model.eval()
        model = NNsight(model)
        # for Transformer, we extract representations from the embed layer using nnsight bc there is special processing of inputs
        with model.trace(torch.from_numpy(backbone_reps).unsqueeze(1)): 
            metalign_reps = model.embed.output.squeeze().save()

    imgs = sorted(glob("data/external/THINGS/*/*jpg"))
    brain_root = Path("data/external/brain_data")
    subs = brain_root.glob("sub-*ResponseData.h5")
    subs = [str(x).split("-")[-1].split("_")[0] for x in subs]

    for sub in tqdm(subs):
        stim_df = pd.read_csv(brain_root/f"sub-{sub}_StimulusMetadata.csv")
        response_df = pd.read_hdf(brain_root/f"sub-{sub}_ResponseData.h5")
        voxel_meta_df = pd.read_csv(brain_root/f"sub-{sub}_VoxelMetadata.csv")


        it_voxels = voxel_meta_df[voxel_meta_df.IT == 1]["voxel_id"].tolist()
        response_df = response_df[response_df.voxel_id.isin(it_voxels)].reset_index(drop=True)
        assert len(it_voxels) == len(response_df)
        del response_df["voxel_id"]
        
        all_responses = response_df.to_numpy().T
        
