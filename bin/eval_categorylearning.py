import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastcore.script import call_parse
from nnsight import NNsight
from tqdm import tqdm

from metalign.cognitive_model import CategoryLearner
from metalign.data import Things, load_backbone
from metalign.model import Transformer, TwoLinear
from metalign.utils import fix_state_dict

_ = torch.set_grad_enabled(False)

@call_parse
def main(
    experiment_name: str, # has to be one of main, raw, midsae
    backbone_name: str, # has to be one of  clip, siglip2, dinov3
    force: bool = False # if True, will overwrite existing eval files
):
    """
    Compare metalign to baselines with linear probes in how aligned they are with human category learning
    """

    eval_path = Path("data/evals/categorylearning")
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
        images = participant_data["image"].tolist()
        images = [f"data/external/THINGS/{image.split("stimuli/")[-1]}" for image in images]
        img_locs = [imgs.index(image) for image in images]
        participant_choices = participant_data["choice"].values
        true_labels = participant_data.true_category_binary.values
        
        # base linear probe
        X = backbone_reps[img_locs]
        learner = CategoryLearner()
        learner.fit(X, true_labels)
        base_preds = np.argmax(learner.values, axis=1)

        # metalign linear probe
        X = metalign_reps[img_locs].numpy()
        learner = CategoryLearner()
        learner.fit(X, true_labels)
        metalign_preds = np.argmax(learner.values, axis=1)

        for i in range(len(participant_choices)):
            results.append({
                "participant": participant,
                "base_choice": base_preds[i],
                "metalign_choice": metalign_preds[i],
                "human_choice": participant_choices[i],
                "true_label": true_labels[i],
                "base_correct": base_preds[i] == true_labels[i],
                "metalign_correct": metalign_preds[i] == true_labels[i],
                "base_align_human": base_preds[i] == participant_choices[i],
                "metalign_align_human": metalign_preds[i] == participant_choices[i]
            })

    result_df = pd.DataFrame(results)
    print(f"Metalign average accuracy: {result_df.metalign_align_human.mean():.4f}")
    print(f"Base average accuracy: {result_df.base_align_human.mean():.4f}")
    result_df.to_csv(eval_file, index=False)

