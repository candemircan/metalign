import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastcore.script import call_parse
from nnsight import NNsight
from tqdm import tqdm

from metalign.cognitive_model import CategoryLearner
from metalign.data import load_backbone_representations
from metalign.model import Transformer
from metalign.utils import fix_state_dict

_ = torch.set_grad_enabled(False)

@call_parse
def main(
    experiment_name: str, # has to be one of main, raw, midsae
    backbone_name: str, # has to be one of  clip, siglip2, dinov3, or mae
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
    model = Transformer(c=config)
    model.load_state_dict(state_dict)
    model.eval()
    model = NNsight(model)

    human_data = pd.read_csv("data/external/category_learning.csv")
    backbone_reps = load_backbone_representations(things_reps)
    with model.trace(torch.from_numpy(backbone_reps).unsqueeze(1)): 
        metalign_reps = model.embed.output.squeeze().save()

    imgs = sorted(glob("data/external/THINGS/*/*jpg"))
    metalign_accuracies, base_accuracies, metalign_linear_accuracies = [], [], []


    for participant in tqdm(human_data.participant.unique()):
        participant_data = human_data[human_data.participant == participant]
        images = participant_data["image"].tolist()
        images = [f"data/external/THINGS/{image.split("stimuli/")[-1]}" for image in images]  # Extract the file names
        img_locs = [imgs.index(image) for image in images]

        # metalign
        X = torch.from_numpy(backbone_reps[img_locs]).unsqueeze(0)
        y = torch.from_numpy(participant_data.true_category_binary.values).unsqueeze(0)
        participant_choices = participant_data["choice"].values
        raw_logits = model(X, y).squeeze().detach().numpy()
        final_probabilities = 1 / (1 + np.exp(-raw_logits))
        answers = (final_probabilities > 0.5).astype(int)
        acc = np.mean(participant_choices == answers)
        metalign_accuracies.append(acc)

        # base
        # no batch dim and back to numpy for the category learner
        X = X.squeeze().numpy()
        y = y.squeeze().numpy()       
        learner = CategoryLearner()
        learner.fit(X, y)
        model_preds = np.argmax(learner.values, axis=1)
        acc = np.mean(participant_choices == model_preds)
        base_accuracies.append(acc)

        # linear probe on metalign
        X = metalign_reps[img_locs].numpy()
        y = participant_data.true_category_binary.values
        learner = CategoryLearner()
        learner.fit(X, y)
        model_preds = np.argmax(learner.values, axis=1)
        acc = np.mean(participant_choices == model_preds)
        metalign_linear_accuracies.append(acc)


    result_df = pd.DataFrame({"participant": human_data.participant.unique(),"metalign_accuracy": metalign_accuracies,  "base_accuracy": base_accuracies, "metalign_linear_accuracy": metalign_linear_accuracies})

    result_df.to_csv(eval_file, index=False)
    print(f"Average metalign accuracy: {np.mean(metalign_accuracies)}")
    print(f"Average base accuracy: {np.mean(base_accuracies)}")
    print(f"Average metalign linear probe accuracy: {np.mean(metalign_linear_accuracies)}")
