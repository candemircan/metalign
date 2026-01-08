import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastcore.script import call_parse
from nnsight import NNsight
from tqdm import tqdm

from metalign.cognitive_model import RewardLearner
from metalign.data import Things, load_backbone
from metalign.model import Transformer, TwoLinear
from metalign.utils import fix_state_dict

_ = torch.set_grad_enabled(False)

@call_parse
def main(
    experiment_name: str, # has to be one of main, raw, midsae
    backbone_name: str, # has to be one of clip, siglip2, dinov3
    force: bool = False # if True, will overwrite existing eval files
):
    """
    Compare metalign to baselines with linear probes in how aligned they are with human reward learning
    """

    eval_path = Path("data/evals/rewardlearning")
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
    
    human_data = pd.read_csv("data/external/reward_learning.csv")
    backbone_reps = load_backbone(things_reps)
    
    if experiment_name.lower() == 'notmeta':
        model = TwoLinear(c=config)
        model.load_state_dict(state_dict)
        model.eval()
        metalign_reps = model.embed(torch.from_numpy(backbone_reps))
    else:
        model = Transformer(c=config)
        model.load_state_dict(state_dict)
        model.eval()
        model = NNsight(model)
        with model.trace(torch.from_numpy(backbone_reps).unsqueeze(1)): 
            metalign_reps = model.embed.output.squeeze().save()

    imgs = [str(x) for x in Things().images]
    results = []

    for participant in tqdm(human_data.participant.unique()):
        participant_data = human_data[human_data.participant == participant]
        left_images = participant_data["left_image"].tolist()
        left_images = [f"data/external/THINGS/{image.split("stimuli/")[-1]}" for image in left_images]
        right_images = participant_data["right_image"].tolist()
        right_images = [f"data/external/THINGS/{image.split("stimuli/")[-1]}" for image in right_images]
        left_img_locs = [imgs.index(image) for image in left_images]
        right_img_locs = [imgs.index(image) for image in right_images]

        X = np.array([[backbone_reps[left], backbone_reps[right]] for left, right in zip(left_img_locs, right_img_locs)])
        y = participant_data[["left_reward", "right_reward"]].values
        participant_choices = participant_data["choice"].values
        correct_choices = np.argmax(y, axis=1)

        # base   
        learner = RewardLearner()
        learner.fit(X, y)
        base_choices = np.argmax(learner.values, axis=1)
        base_values = learner.values # (n_trials, 2)

        # metalign
        X = np.array([[metalign_reps[left], metalign_reps[right]] for left, right in zip(left_img_locs, right_img_locs)])
        learner = RewardLearner()
        learner.fit(X, y)
        metalign_choices = np.argmax(learner.values, axis=1)
        metalign_values = learner.values # (n_trials, 2)

        for i in range(len(participant_choices)):
            results.append({
                "participant": participant,
                "trial": i,
                "human_choice": participant_choices[i],
                "correct_choice": correct_choices[i],
                "base_choice": base_choices[i],
                "metalign_choice": metalign_choices[i],
                "base_val_0": base_values[i, 0],
                "base_val_1": base_values[i, 1],
                "metalign_val_0": metalign_values[i, 0],
                "metalign_val_1": metalign_values[i, 1],
                "base_correct": base_choices[i] == correct_choices[i],
                "metalign_correct": metalign_choices[i] == correct_choices[i],
                "base_align_human": base_choices[i] == participant_choices[i],
                "metalign_align_human": metalign_choices[i] == participant_choices[i]
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(eval_file, index=False)

