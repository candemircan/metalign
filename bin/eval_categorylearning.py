from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastcore.script import call_parse
from tqdm import tqdm

from metalign.cognitive_model import CategoryLearner
from metalign.data import load_backbone_representations
from metalign.model import Transformer

_ = torch.set_grad_enabled(False)

@call_parse
def main(
    checkpoint_name: str, # path to model checkpoint file
):
    """
    Compare metalign to baselines with linear probes in how aligned they are with human category learning
    """
    
    checkpoint_path = Path("data/checkpoints") / checkpoint_name
    model_name = checkpoint_name.split("_")[1]
    things_reps = glob(f"data/backbone_reps/things_{model_name}*.h5")[0]

    human_data = pd.read_csv("data/external/category_learning.csv")
    backbone_reps = load_backbone_representations(things_reps)

    ckpt = torch.load(checkpoint_path / "best.pt", weights_only=False)
    config, state_dict = ckpt['config'], ckpt['state_dict']
    model = Transformer(config=config)
    model.load_state_dict(state_dict)
    model.eval()

    imgs = sorted(glob("data/external/THINGS/*/*jpg"))
    metalign_accuracies = []
    base_accuracies = []


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

        # no batch dim and back to numpy for the category learner
        X = X.squeeze().numpy()
        y = y.squeeze().numpy()       
        learner = CategoryLearner()
        learner.fit(X, y)
        model_preds = np.argmax(learner.values, axis=1)
        acc = np.mean(participant_choices == model_preds)
        base_accuracies.append(acc)

    result_df = pd.DataFrame({"participant": human_data.participant.unique(),"metalign_accuracy": metalign_accuracies,"base_accuracy": base_accuracies})
    eval_path = Path("data/evals/categorylearning")
    eval_path.mkdir(parents=True, exist_ok=True)
    eval_file = eval_path / f"{checkpoint_name}.csv"
    result_df.to_csv(eval_file, index=False)
    print(f"Average metalign accuracy: {np.mean(metalign_accuracies)}")
    print(f"Average base accuracy: {np.mean(base_accuracies)}")
    print(f"{(result_df.metalign_accuracy > result_df.base_accuracy).mean()} of participants were better modeled by metalign")