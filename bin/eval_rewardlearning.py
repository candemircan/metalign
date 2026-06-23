import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from fastcore.script import call_parse
from scipy.optimize import minimize
from scipy.special import softmax
from tqdm import tqdm

from metalign.cognitive_model import RewardLearner
from metalign.data import Things, load_backbone
from metalign.model import get_model
from metalign.utils import fix_state_dict

_ = torch.set_grad_enabled(False)
warnings.filterwarnings('ignore')

def _load_model_reps(ckpt_path, backbone_reps):
    "Load model and extract representations via model.encode()"
    ckpt = torch.load(ckpt_path, weights_only=False)
    model = get_model(ckpt['config'])
    model.load_state_dict(fix_state_dict(ckpt['state_dict']))
    model.eval()
    x = backbone_reps if isinstance(backbone_reps, torch.Tensor) else torch.from_numpy(backbone_reps)
    return model.encode(x)


def fit_temperature(values, y):
    "Fit temperature parameter to minimize NLL of choices"
    def nll(log_temp):
        temp = np.exp(log_temp)
        probs = softmax(values / temp, axis=1)
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        return -np.mean(np.log(probs[np.arange(len(y)), y]))

    res = minimize(nll, x0=[0.0], method='L-BFGS-B', bounds=[(-4.6, 4.6)])
    return np.exp(res.x[0])

def compute_nll(values, y, temp):
    "Compute per-trial NLL given values (logits), choices, and temperature"
    probs = softmax(values / temp, axis=1)
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    return -np.log(probs[np.arange(len(y)), y])

@call_parse
def main(
    experiment_name: str, # One of main, raw, midsae, static_linear, 8heads, 32heads, mlp, wd, lstm, 2layers
    backbone_name: str, # clip, siglip2, dinov3, or mae
    force: bool = False
):
    "Evaluate how well base vs metalign predict human reward learning choices"

    eval_path = Path("data/evals/rewardlearning")
    eval_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{experiment_name}_{backbone_name}"
    output_file = eval_path / f"{file_name}_trials.h5"

    if output_file.exists() and not force:
        print(f"Output for {file_name} already exists, use --force to overwrite")
        return

    best_models = json.load(open(Path("data/checkpoints") / "best_models.json"))
    backbone_dict = json.load(open(Path("data/backbone_reps") / "backbones.json"))
    things_reps = f"data/backbone_reps/things_{backbone_dict[backbone_name]}.h5"

    human_data = pd.read_csv("data/external/reward_learning.csv")
    backbone_reps = load_backbone(things_reps)

    # Load experiment model reps
    exp_reps = _load_model_reps(best_models[f"[{experiment_name}]"][backbone_name], backbone_reps)

    # Baseline: backbone always, plus main-model for ablations
    base_reps = backbone_reps
    main_reps = None if experiment_name == "main" else _load_model_reps(best_models["[main]"][backbone_name], backbone_reps)

    imgs = [str(x) for x in Things().images]
    trial_rows = []

    for participant in tqdm(human_data.participant.unique()):
        participant_data = human_data[human_data.participant == participant]
        left_images = participant_data["left_image"].tolist()
        left_images = [f"data/external/THINGS/{image.split('stimuli/')[-1]}" for image in left_images]
        right_images = participant_data["right_image"].tolist()
        right_images = [f"data/external/THINGS/{image.split('stimuli/')[-1]}" for image in right_images]
        left_img_locs = [imgs.index(image) for image in left_images]
        right_img_locs = [imgs.index(image) for image in right_images]

        # Helper to get numpy array from tensor or numpy
        def to_np(x): return x.numpy() if hasattr(x, 'numpy') else x

        X_base = np.array([[to_np(base_reps[left]), to_np(base_reps[right])] for left, right in zip(left_img_locs, right_img_locs)])
        y = participant_data[["left_reward", "right_reward"]].values
        human_choices = participant_data["choice"].values

        # Fit base model
        base_learner = RewardLearner()
        base_learner.fit(X_base, y)
        base_values = base_learner.values

        # Fit temperature for base (minimize NLL of human choices)
        temp_base = fit_temperature(base_values, human_choices)
        base_nlls = compute_nll(base_values, human_choices, temp_base)

        # Fit main model (if ablation)
        main_nlls = None
        if main_reps is not None:
            X_main = np.array([[main_reps[left].numpy(), main_reps[right].numpy()] for left, right in zip(left_img_locs, right_img_locs)])
            main_learner = RewardLearner()
            main_learner.fit(X_main, y)
            main_values = main_learner.values
            temp_main = fit_temperature(main_values, human_choices)
            main_nlls = compute_nll(main_values, human_choices, temp_main)

        # Fit experiment model
        X_exp = np.array([[exp_reps[left].numpy(), exp_reps[right].numpy()] for left, right in zip(left_img_locs, right_img_locs)])
        exp_learner = RewardLearner()
        exp_learner.fit(X_exp, y)
        exp_values = exp_learner.values

        # Fit temperature for experiment (minimize NLL of human choices)
        temp_exp = fit_temperature(exp_values, human_choices)
        meta_nlls = compute_nll(exp_values, human_choices, temp_exp)

        # Save trial-level data
        for i in range(len(human_choices)):
            row = {'participant_id': participant, 'base_nll': base_nlls[i], 'meta_nll': meta_nlls[i]}
            if main_nlls is not None: row['main_nll'] = main_nlls[i]
            trial_rows.append(row)

    # Save trial-level results
    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_hdf(output_file, key='trials', mode='w')
    print(f"Saved {len(trial_df)} trials to {output_file}")
