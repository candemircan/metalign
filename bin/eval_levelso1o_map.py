"""
Mixed conditional logit evaluation using NumPyro MAP estimation.
Replicates eval_levelso1o.R analysis in Python.
"""
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
from fastcore.script import call_parse
from jax.nn import log_softmax
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoDelta
from numpyro.optim import Adam

from metalign.utils import calculate_log10_bf, calculate_lrt

numpyro.set_host_device_count(1)

def z_score(x):
    return (x - np.mean(x)) / np.std(x)

def prepare_data(df, is_main, df_main=None):
    "Prepare data arrays for NumPyro model"
    df = df.copy()
    n_trials = len(df)

    if not is_main and df_main is not None:
        for i in range(3):
            df[f'metalign_sim_main_{i}'] = df_main[f'metalign_sim_{i}'].values

    # Build predictor matrices (n_trials, 3)
    base_sim = np.stack([df[f'base_sim_{i}'].values for i in range(3)], axis=1)
    metalign_sim = np.stack([df[f'metalign_sim_{i}'].values for i in range(3)], axis=1)
    asc1 = np.array([[0, 1, 0]] * n_trials, dtype=np.float64)
    asc2 = np.array([[0, 0, 1]] * n_trials, dtype=np.float64)

    # Z-score
    base_sim = (base_sim - base_sim.mean()) / base_sim.std()
    metalign_sim = (metalign_sim - metalign_sim.mean()) / metalign_sim.std()

    # Participant indices
    participants = df['participant_id'].unique()
    p2idx = {p: i for i, p in enumerate(participants)}
    p_idx = df['participant_id'].map(p2idx).values

    data = {
        'base_sim': base_sim,
        'metalign_sim': metalign_sim,
        'asc1': asc1,
        'asc2': asc2,
        'n_participants': len(participants),
        'p_idx': p_idx,
        'choices': df['y'].values,
        'n_trials': n_trials,
    }

    if not is_main:
        metalign_sim_main = np.stack([df[f'metalign_sim_main_{i}'].values for i in range(3)], axis=1)
        metalign_sim_main = (metalign_sim_main - metalign_sim_main.mean()) / metalign_sim_main.std()
        data['metalign_sim_main'] = metalign_sim_main

    return data

def mixed_logit_model(X, p_idx, choices, n_p, pred_names):
    """
    Mixed conditional logit with random effects.
    X: dict of predictor arrays (n_trials, 3)
    p_idx: participant index per trial (n_trials,)
    choices: chosen alternative per trial (n_trials,)
    n_p: number of participants
    pred_names: list of predictor names
    """
    n_pred = len(pred_names)
    n_t = len(choices)

    # Fixed effects
    beta = numpyro.sample('beta', dist.Normal(0, 2).expand([n_pred]))

    # Random effect std devs
    sigma = numpyro.sample('sigma', dist.HalfNormal(1).expand([n_pred]))

    # Random effects per participant: shape (n_p, n_pred)
    re = numpyro.sample('re', dist.Normal(0, 1).expand([n_p, n_pred]))

    # Scale random effects by sigma
    re_scaled = re * sigma  # (n_p, n_pred)

    # Compute utilities
    utilities = jnp.zeros((n_t, 3))
    for i, name in enumerate(pred_names):
        coef = beta[i] + re_scaled[p_idx, i]
        utilities = utilities + coef[:, None] * X[name]

    # Likelihood
    numpyro.sample('obs', dist.Categorical(logits=utilities), obs=choices)

def fit_map(data, pred_names, n_steps=5000, lr=0.01):
    "Fit mixed logit via MAP using SVI with AutoDelta guide"
    X = {name: jnp.array(data[name]) for name in pred_names}
    p_idx = jnp.array(data['p_idx'])
    choices = jnp.array(data['choices'])
    n_p = data['n_participants']
    n_t = data['n_trials']

    def model():
        mixed_logit_model(X, p_idx, choices, n_p, pred_names)

    guide = AutoDelta(model)
    optimizer = Adam(lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # Use run() for JIT-compiled optimization loop
    rng_key = jax.random.PRNGKey(0)
    svi_result = svi.run(rng_key, n_steps, progress_bar=True)
    params = svi_result.params

    # Extract MAP estimates
    beta = np.array(params['beta_auto_loc'])
    sigma = np.array(params['sigma_auto_loc'])
    re = np.array(params['re_auto_loc'])

    # Compute log-likelihood at MAP
    re_scaled = re * sigma
    utilities = np.zeros((n_t, 3))
    for i, name in enumerate(pred_names):
        coef = beta[i] + re_scaled[data['p_idx'], i]
        utilities += coef[:, None] * data[name]

    log_probs = np.array(log_softmax(jnp.array(utilities), axis=1))
    ll = np.sum(log_probs[np.arange(n_t), data['choices']])

    # BIC: count fixed effects + variance params (like R mlogit)
    k = 2 * len(pred_names)  # betas + sigmas
    bic = -2 * ll + k * np.log(n_t)
    aic = -2 * ll + 2 * k

    return {
        'betas': {name: float(beta[i]) for i, name in enumerate(pred_names)},
        'sigmas': {name: float(sigma[i]) for i, name in enumerate(pred_names)},
        'loglik': float(ll),
        'bic': float(bic),
        'aic': float(aic),
        'k': k,
        'n_obs': n_t,
    }

@call_parse
def main(
    experiment_name: str,  # MAIN, RAW, MIDSAE, NOTMETA
    backbone_name: str,    # clip, siglip2, dinov3
):
    """Fit mixed conditional logit models using NumPyro MAP estimation."""
    print(f"Running eval_levelso1o_map.py for {experiment_name} {backbone_name}")

    stats_file = Path(f"data/evals/levelso1o/{experiment_name}_{backbone_name}_stats.csv")
    if not stats_file.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_file}")

    df = pd.read_csv(stats_file)
    is_main = experiment_name == "MAIN"

    df_main = None
    if not is_main:
        main_file = Path(f"data/evals/levelso1o/MAIN_{backbone_name}_stats.csv")
        if not main_file.exists():
            raise FileNotFoundError("Main model stats file not found")
        df_main = pd.read_csv(main_file)

    data = prepare_data(df, is_main, df_main)
    results = {'experiment_name': experiment_name, 'backbone_name': backbone_name}

    # Fit null model for R2
    print("Fitting Null Model...")
    null_stats = fit_map(data, ['asc1', 'asc2'])
    null_ll = null_stats['loglik']

    if is_main:
        print("Fitting Model 0 (Base only)...")
        stats0 = fit_map(data, ['base_sim', 'asc1', 'asc2'])
        stats0['r2'] = 1 - stats0['loglik'] / null_ll
        print(f"Model 0 LogLik: {stats0['loglik']:.2f}, BIC: {stats0['bic']:.2f}")

        print("Fitting Model M (Metalign only)...")
        stats_m = fit_map(data, ['metalign_sim', 'asc1', 'asc2'])
        stats_m['r2'] = 1 - stats_m['loglik'] / null_ll
        print(f"Model M LogLik: {stats_m['loglik']:.2f}, BIC: {stats_m['bic']:.2f}")

        print("Fitting Model 1 (Base + Metalign)...")
        stats1 = fit_map(data, ['base_sim', 'metalign_sim', 'asc1', 'asc2'])
        stats1['r2'] = 1 - stats1['loglik'] / null_ll
        print(f"Model 1 LogLik: {stats1['loglik']:.2f}, BIC: {stats1['bic']:.2f}")

        log10_bf_0_1 = calculate_log10_bf(stats0['bic'], stats1['bic'])
        log10_bf_0_m = calculate_log10_bf(stats0['bic'], stats_m['bic'])
        log10_bf_m_1 = calculate_log10_bf(stats_m['bic'], stats1['bic'])
        lrt_0_1 = calculate_lrt(stats0, stats1)
        print(f"Log10 BF (M1/M0): {log10_bf_0_1:.2f}, Log10 BF (M/M0): {log10_bf_0_m:.2f}")

        results['model_0'] = stats0
        results['model_m'] = stats_m
        results['model_1'] = stats1
        results['comparison_0_vs_1'] = {'log10_bayes_factor': log10_bf_0_1, 'lrt': lrt_0_1}
        results['comparison_0_vs_m'] = {'log10_bayes_factor': log10_bf_0_m}
        results['comparison_m_vs_1'] = {'log10_bayes_factor': log10_bf_m_1}
    else:
        print("Fitting Model 1 (Base + Ablation)...")
        stats1 = fit_map(data, ['base_sim', 'metalign_sim', 'asc1', 'asc2'])
        stats1['r2'] = 1 - stats1['loglik'] / null_ll

        print("Fitting Model 2 (Base + Ablation + Main)...")
        stats2 = fit_map(data, ['base_sim', 'metalign_sim', 'metalign_sim_main', 'asc1', 'asc2'])
        stats2['r2'] = 1 - stats2['loglik'] / null_ll
        print(f"Model 1 BIC: {stats1['bic']:.2f}, Model 2 BIC: {stats2['bic']:.2f}")

        log10_bf = calculate_log10_bf(stats1['bic'], stats2['bic'])
        lrt = calculate_lrt(stats1, stats2)
        print(f"Log10 BF (M2/M1): {log10_bf:.2f}, LRT p-value: {lrt['p_value']:.3e}")

        results['model_1'] = stats1
        results['model_2'] = stats2
        results['comparison_1_vs_2'] = {'log10_bayes_factor': log10_bf, 'lrt': lrt}

    output_file = Path(f"data/evals/levelso1o/{experiment_name}_{backbone_name}_analysis_map.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")
