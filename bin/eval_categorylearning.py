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

from metalign.cognitive_model import CategoryLearner
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


def fit_temperature(logits, y):
    "Fit temperature parameter to minimize NLL of choices"
    def nll(log_temp):
        temp = np.exp(log_temp)
        probs = softmax(logits / temp, axis=1)
        eps = 1e-15
        probs = np.clip(probs, eps, 1 - eps)
        return -np.mean(np.log(probs[np.arange(len(y)), y]))

    res = minimize(nll, x0=[0.0], method='L-BFGS-B', bounds=[(-4.6, 4.6)])
    return np.exp(res.x[0])

def compute_nll(logits, y, temp):
    "Compute per-trial NLL given logits, choices, and temperature"
    probs = softmax(logits / temp, axis=1)
    eps = 1e-15
    probs = np.clip(probs, eps, 1 - eps)
    return -np.log(probs[np.arange(len(y)), y])

@call_parse
def main(
    experiment_name: str, # One of main, raw, midsae, static_linear, 8heads, 32heads, mlp, wd, lstm, 2layers
    backbone_name: str, # clip, siglip2, dinov3, or mae
    force: bool = False
):
    "Evaluate how well base vs metalign predict human category learning choices"

    eval_path = Path("data/evals/categorylearning")
    eval_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{experiment_name}_{backbone_name}"
    output_file = eval_path / f"{file_name}_trials.h5"

    if output_file.exists() and not force:
        print(f"Output for {file_name} already exists, use --force to overwrite")
        return

    best_models = json.load(open(Path("data/checkpoints") / "best_models.json"))
    backbone_dict = json.load(open(Path("data/backbone_reps") / "backbones.json"))
    things_reps = f"data/backbone_reps/things_{backbone_dict[backbone_name]}.h5"

    human_data = pd.read_csv("data/external/category_learning.csv")
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
        images = participant_data["image"].tolist()
        images = [f"data/external/THINGS/{image.split('stimuli/')[-1]}" for image in images]
        img_locs = [imgs.index(image) for image in images]
        human_choices = participant_data["choice"].values
        true_labels = participant_data.true_category_binary.values

        X_base = base_reps[img_locs].numpy() if hasattr(base_reps, 'numpy') else base_reps[img_locs]
        # Fit base model
        base_learner = CategoryLearner().fit(X_base, true_labels)
        base_probs = base_learner.values
        base_logits = np.log(np.clip(base_probs, 1e-15, 1-1e-15))

        # Fit temperature for base (minimize NLL of human choices)
        temp_base = fit_temperature(base_logits, human_choices)
        base_nlls = compute_nll(base_logits, human_choices, temp_base)

        # Fit main model (if ablation)
        main_nlls = None
        if main_reps is not None:
            X_main = main_reps[img_locs].numpy()
            main_learner = CategoryLearner().fit(X_main, true_labels)
            main_probs = main_learner.values
            main_logits = np.log(np.clip(main_probs, 1e-15, 1-1e-15))
            temp_main = fit_temperature(main_logits, human_choices)
            main_nlls = compute_nll(main_logits, human_choices, temp_main)

        # Fit experiment model
        X_exp = exp_reps[img_locs].numpy()
        exp_learner = CategoryLearner().fit(X_exp, true_labels)
        exp_probs = exp_learner.values
        exp_logits = np.log(np.clip(exp_probs, 1e-15, 1-1e-15))

        # Fit temperature for experiment (minimize NLL of human choices)
        temp_exp = fit_temperature(exp_logits, human_choices)
        meta_nlls = compute_nll(exp_logits, human_choices, temp_exp)

        # Save trial-level data
        for i in range(len(human_choices)):
            row = {'participant_id': participant, 'base_nll': base_nlls[i], 'meta_nll': meta_nlls[i]}
            if main_nlls is not None: row['main_nll'] = main_nlls[i]
            trial_rows.append(row)

    # Save trial-level results
    trial_df = pd.DataFrame(trial_rows)
    trial_df.to_hdf(output_file, key='trials', mode='w')
    print(f"Saved {len(trial_df)} trials to {output_file}")
