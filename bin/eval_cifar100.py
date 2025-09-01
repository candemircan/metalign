import json
from pathlib import Path

import numpy as np
import torch
from fastcore.script import call_parse
from nnsight import NNsight
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torchvision.datasets import CIFAR100
from tqdm import tqdm

from metalign.data import load_backbone_representations
from metalign.model import Transformer

_ = torch.set_grad_enabled(False)

def _evaluate_few_shot(train_backbone_reps, train_metalign_reps, train_targets, 
                      test_backbone_reps, test_metalign_reps, test_targets, n_shot, n_runs=20):
    """Evaluate few-shot learning performance"""
    
    backbone_results = []
    metalign_results = []
    
    for run in tqdm(range(n_runs), desc=f"{n_shot}-shot evaluation"):
        # Sample n_shot instances per class from training data
        train_indices = []
        val_indices = []
        
        for class_id in range(100):  # CIFAR-100 has 100 classes
            class_indices = np.where(train_targets == class_id)[0]
            np.random.shuffle(class_indices)
            
            train_indices.extend(class_indices[:n_shot])
            val_indices.extend(class_indices[n_shot:])
        
        train_indices = np.array(train_indices)
        val_indices = np.array(val_indices)
        
        # Prepare training and validation data
        X_train_backbone = train_backbone_reps[train_indices]
        X_train_metalign = train_metalign_reps[train_indices]
        y_train = train_targets[train_indices]
        
        X_val_backbone = train_backbone_reps[val_indices] 
        X_val_metalign = train_metalign_reps[val_indices]
        y_val = train_targets[val_indices]
        
        # Hyperparameter search space
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'use_scaler': [True, False]
        }
        
        # Evaluate backbone representations
        best_backbone_acc = 0
        best_backbone_params = None
        for C in param_grid['C']:
            for use_scaler in param_grid['use_scaler']:
                if use_scaler:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_backbone)
                    X_val_scaled = scaler.transform(X_val_backbone)
                else:
                    X_train_scaled = X_train_backbone
                    X_val_scaled = X_val_backbone
                    scaler = None
                
                clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
                clf.fit(X_train_scaled, y_train)
                val_acc = clf.score(X_val_scaled, y_val)
                
                if val_acc > best_backbone_acc:
                    best_backbone_acc = val_acc
                    best_backbone_params = {'C': C, 'scaler': scaler}
        
        # Evaluate metalign representations  
        best_metalign_acc = 0
        best_metalign_params = None
        for C in param_grid['C']:
            for use_scaler in param_grid['use_scaler']:
                if use_scaler:
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_metalign)
                    X_val_scaled = scaler.transform(X_val_metalign)
                else:
                    X_train_scaled = X_train_metalign
                    X_val_scaled = X_val_metalign
                    scaler = None
                
                clf = LogisticRegression(C=C, max_iter=1000, random_state=42)
                clf.fit(X_train_scaled, y_train)
                val_acc = clf.score(X_val_scaled, y_val)
                
                if val_acc > best_metalign_acc:
                    best_metalign_acc = val_acc
                    best_metalign_params = {'C': C, 'scaler': scaler}
        
        # Test on test set with best hyperparameters
        # Backbone test
        if best_backbone_params['scaler'] is not None:
            X_train_backbone_scaled = best_backbone_params['scaler'].fit_transform(X_train_backbone)
            X_test_backbone_scaled = best_backbone_params['scaler'].transform(test_backbone_reps)
        else:
            X_train_backbone_scaled = X_train_backbone
            X_test_backbone_scaled = test_backbone_reps
            
        backbone_clf = LogisticRegression(C=best_backbone_params['C'], max_iter=1000, random_state=42)
        backbone_clf.fit(X_train_backbone_scaled, y_train)
        backbone_test_acc = backbone_clf.score(X_test_backbone_scaled, test_targets)
        
        # Metalign test
        if best_metalign_params['scaler'] is not None:
            X_train_metalign_scaled = best_metalign_params['scaler'].fit_transform(X_train_metalign)
            X_test_metalign_scaled = best_metalign_params['scaler'].transform(test_metalign_reps)
        else:
            X_train_metalign_scaled = X_train_metalign
            X_test_metalign_scaled = test_metalign_reps
            
        metalign_clf = LogisticRegression(C=best_metalign_params['C'], max_iter=1000, random_state=42)
        metalign_clf.fit(X_train_metalign_scaled, y_train)
        metalign_test_acc = metalign_clf.score(X_test_metalign_scaled, test_targets)
        
        backbone_results.append(backbone_test_acc)
        metalign_results.append(metalign_test_acc)
    
    return {
        'backbone_mean': np.mean(backbone_results),
        'backbone_std': np.std(backbone_results),
        'metalign_mean': np.mean(metalign_results), 
        'metalign_std': np.std(metalign_results),
        'backbone_results': backbone_results,
        'metalign_results': metalign_results
    }

@call_parse
def main(
    experiment_name: str, # has to be one of main, raw, midsae
    backbone_name: str, # has to be one of vit, clip, siglip2, dinov2
    n_runs: int = 20 # number of runs for few-shot evaluation
):
    """
    Evaluate 1-shot and 10-shot accuracy on CIFAR-100 classification task
    """
    
    best_models = json.load(open(Path("data/checkpoints") / "best_models.json"))
    backbone_dict = json.load(open(Path("data/backbone_reps") / "backbones.json"))
    ckpt = best_models[f"[{experiment_name.upper()}]"][backbone_name]
    
    # Load train and test representations
    train_reps_path = f"data/backbone_reps/cifar100_train_{backbone_dict[backbone_name]}.h5"
    test_reps_path = f"data/backbone_reps/cifar100_test_{backbone_dict[backbone_name]}.h5"
    
    train_backbone_reps = load_backbone_representations(train_reps_path)
    test_backbone_reps = load_backbone_representations(test_reps_path)
    
    # Get CIFAR-100 labels (we need the actual labels, not just representations)
    train_dataset = CIFAR100(root="data/external", train=True, download=False)
    test_dataset = CIFAR100(root="data/external", train=False, download=False)
    
    train_targets = np.array(train_dataset.targets)
    test_targets = np.array(test_dataset.targets)
    
    # Load metalign model
    ckpt = torch.load(ckpt, weights_only=False)
    config, state_dict = ckpt['config'], ckpt['state_dict']
    model = Transformer(config=config)
    model.load_state_dict(state_dict)
    model.eval()
    model = NNsight(model)
    
    # Get metalign representations for train and test
    with model.trace(torch.from_numpy(train_backbone_reps).unsqueeze(1)): 
        train_metalign_reps = model.embedding.output.squeeze().save()
    
    with model.trace(torch.from_numpy(test_backbone_reps).unsqueeze(1)):
        test_metalign_reps = model.embedding.output.squeeze().save()
    
    train_metalign_reps = train_metalign_reps.cpu().numpy()
    test_metalign_reps = test_metalign_reps.cpu().numpy()
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Evaluate 1-shot and 10-shot
    results_1_shot = _evaluate_few_shot(
        train_backbone_reps, train_metalign_reps, train_targets, 
        test_backbone_reps, test_metalign_reps, test_targets, 1, n_runs
    )
    
    results_10_shot = _evaluate_few_shot(
        train_backbone_reps, train_metalign_reps, train_targets,
        test_backbone_reps, test_metalign_reps, test_targets, 10, n_runs
    )
    
    # Save results
    eval_path = Path("data/evals/cifar100")
    eval_path.mkdir(parents=True, exist_ok=True)
    file_name = f"{experiment_name}_{backbone_name}"
    eval_file = eval_path / f"{file_name}.json"
    
    eval_data = {
        "model_name": backbone_name,
        "checkpoint_name": file_name,
        "n_runs": n_runs,
        "1_shot": results_1_shot,
        "10_shot": results_10_shot
    }
    
    with open(eval_file, "w") as f: 
        json.dump(eval_data, f, indent=4)
    
    print("1-shot results:")
    print(f"  Backbone: {results_1_shot['backbone_mean']:.4f} ± {results_1_shot['backbone_std']:.4f}")
    print(f"  Metalign: {results_1_shot['metalign_mean']:.4f} ± {results_1_shot['metalign_std']:.4f}")
    
    print("10-shot results:")
    print(f"  Backbone: {results_10_shot['backbone_mean']:.4f} ± {results_10_shot['backbone_std']:.4f}")
    print(f"  Metalign: {results_10_shot['metalign_mean']:.4f} ± {results_10_shot['metalign_std']:.4f}")
