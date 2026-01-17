"""
Mixed logistic regression evaluation using NumPyro MAP estimation.
Replicates eval_categorylearning.R analysis in Python.
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

    if not is_main and df_main is not None:
        df['main_logit_1'] = df_main['metalign_logit_1'].values

    # Z-score predictors
    df['base_logit_1'] = z_score(df['base_logit_1'])
    df['metalign_logit_1'] = z_score(df['metalign_logit_1'])
    if 'main_logit_1' in df.columns:
        df['main_logit_1'] = z_score(df['main_logit_1'])

    # Participant indices
    participants = df['participant'].unique()
    p2idx = {p: i for i, p in enumerate(participants)}
    p_idx = df['participant'].map(p2idx).values

    return df, p_idx, len(participants)

def mixed_logistic_model(X, p_idx, y, n_p, pred_names):
    """
    Mixed logistic regression with random intercept and slopes.
    X: dict of predictor arrays (n_obs,)
    p_idx: participant index per observation (n_obs,)
    y: binary outcome (n_obs,)
    n_p: number of participants
    pred_names: list of predictor names (not including intercept)
    """
    n_pred = len(pred_names) + 1  # +1 for intercept
    n_obs = len(y)

    # Fixed effects (intercept + slopes)
    beta = numpyro.sample('beta', dist.Normal(0, 2).expand([n_pred]))

    # Random effect std devs (intercept + slopes)
    sigma = numpyro.sample('sigma', dist.HalfNormal(1).expand([n_pred]))

    # Random effects per participant: shape (n_p, n_pred)
    # Using independent random effects (like uncorrelated in lme4)
    re = numpyro.sample('re', dist.Normal(0, 1).expand([n_p, n_pred]))
    re_scaled = re * sigma  # (n_p, n_pred)

    # Compute linear predictor
    # Intercept
    eta = beta[0] + re_scaled[p_idx, 0]
    # Slopes
    for i, name in enumerate(pred_names):
        coef = beta[i + 1] + re_scaled[p_idx, i + 1]
        eta = eta + coef * X[name]

    # Likelihood
    numpyro.sample('obs', dist.Bernoulli(logits=eta), obs=y)

def fit_map(df, p_idx, n_p, pred_names, n_steps=5000, lr=0.01):
    "Fit mixed logistic regression via MAP using SVI with AutoDelta guide"
    X = {name: jnp.array(df[name].values) for name in pred_names}
    y = jnp.array(df['choice'].values)
    n_obs = len(y)

    def model():
        mixed_logistic_model(X, jnp.array(p_idx), y, n_p, pred_names)

    guide = AutoDelta(model)
    optimizer = Adam(lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rng_key = jax.random.PRNGKey(0)
    svi_result = svi.run(rng_key, n_steps, progress_bar=False)
    params = svi_result.params

    # Extract MAP estimates
    beta = np.array(params['beta_auto_loc'])
    sigma = np.array(params['sigma_auto_loc'])
    re = np.array(params['re_auto_loc'])

    # Compute log-likelihood at MAP
    re_scaled = re * sigma
    eta = beta[0] + re_scaled[p_idx, 0]
    for i, name in enumerate(pred_names):
        coef = beta[i + 1] + re_scaled[p_idx, i + 1]
        eta = eta + coef * df[name].values

    # Bernoulli log-likelihood
    y_np = df['choice'].values
    prob = 1 / (1 + np.exp(-eta))
    ll = np.sum(y_np * np.log(prob + 1e-10) + (1 - y_np) * np.log(1 - prob + 1e-10))

    # BIC: count fixed effects + variance params
    n_fixed = len(pred_names) + 1  # intercept + slopes
    n_var = len(pred_names) + 1    # variance for intercept + slopes
    k = n_fixed + n_var
    bic = -2 * ll + k * np.log(n_obs)
    aic = -2 * ll + 2 * k

    return {
        'betas': {'intercept': float(beta[0]), **{name: float(beta[i + 1]) for i, name in enumerate(pred_names)}},
        'sigmas': {'intercept': float(sigma[0]), **{name: float(sigma[i + 1]) for i, name in enumerate(pred_names)}},
        'loglik': float(ll),
        'bic': float(bic),
        'aic': float(aic),
        'k': k,
        'n_obs': n_obs,
    }

def fit_null(df, p_idx, n_p, n_steps=5000, lr=0.01):
    "Fit null model (intercept only with random intercept)"
    y = jnp.array(df['choice'].values)
    n_obs = len(y)

    def model():
        beta0 = numpyro.sample('beta0', dist.Normal(0, 2))
        sigma0 = numpyro.sample('sigma0', dist.HalfNormal(1))
        re0 = numpyro.sample('re0', dist.Normal(0, 1).expand([n_p]))
        eta = beta0 + sigma0 * re0[jnp.array(p_idx)]
        numpyro.sample('obs', dist.Bernoulli(logits=eta), obs=y)

    guide = AutoDelta(model)
    optimizer = Adam(lr)
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    rng_key = jax.random.PRNGKey(0)
    svi_result = svi.run(rng_key, n_steps, progress_bar=False)
    params = svi_result.params

    beta0 = float(params['beta0_auto_loc'])
    sigma0 = float(params['sigma0_auto_loc'])
    re0 = np.array(params['re0_auto_loc'])

    eta = beta0 + sigma0 * re0[p_idx]
    y_np = df['choice'].values
    prob = 1 / (1 + np.exp(-eta))
    ll = np.sum(y_np * np.log(prob + 1e-10) + (1 - y_np) * np.log(1 - prob + 1e-10))

    return {'loglik': float(ll)}

@call_parse
def main(
    experiment_name: str,  # MAIN, RAW, MIDSAE, NOTMETA
    backbone_name: str,    # clip, siglip2, dinov3
):
    """Fit mixed logistic regression models using NumPyro MAP estimation."""
    print(f"Running eval_categorylearning_map.py for {experiment_name} {backbone_name}")

    stats_file = Path(f"data/evals/categorylearning/{experiment_name}_{backbone_name}_stats.csv")
    if not stats_file.exists():
        raise FileNotFoundError(f"Stats file not found: {stats_file}")

    df = pd.read_csv(stats_file)
    is_main = experiment_name == "MAIN"

    df_main = None
    if not is_main:
        main_file = Path(f"data/evals/categorylearning/MAIN_{backbone_name}_stats.csv")
        if not main_file.exists():
            raise FileNotFoundError("Main model stats file not found")
        df_main = pd.read_csv(main_file)

    df, p_idx, n_p = prepare_data(df, is_main, df_main)
    results = {'experiment_name': experiment_name, 'backbone_name': backbone_name}

    print("Fitting Null Model...")
    null_stats = fit_null(df, p_idx, n_p)
    null_ll = null_stats['loglik']

    if is_main:
        print("Fitting Model 0 (Base only)...")
        stats0 = fit_map(df, p_idx, n_p, ['base_logit_1'])
        stats0['r2'] = 1 - stats0['loglik'] / null_ll
        print(f"Model 0 LogLik: {stats0['loglik']:.2f}, BIC: {stats0['bic']:.2f}")

        print("Fitting Model M (Metalign only)...")
        stats_m = fit_map(df, p_idx, n_p, ['metalign_logit_1'])
        stats_m['r2'] = 1 - stats_m['loglik'] / null_ll
        print(f"Model M LogLik: {stats_m['loglik']:.2f}, BIC: {stats_m['bic']:.2f}")

        print("Fitting Model 1 (Base + Metalign)...")
        stats1 = fit_map(df, p_idx, n_p, ['base_logit_1', 'metalign_logit_1'])
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
        stats1 = fit_map(df, p_idx, n_p, ['base_logit_1', 'metalign_logit_1'])
        stats1['r2'] = 1 - stats1['loglik'] / null_ll

        print("Fitting Model 2 (Base + Ablation + Main)...")
        stats2 = fit_map(df, p_idx, n_p, ['base_logit_1', 'metalign_logit_1', 'main_logit_1'])
        stats2['r2'] = 1 - stats2['loglik'] / null_ll
        print(f"Model 1 BIC: {stats1['bic']:.2f}, Model 2 BIC: {stats2['bic']:.2f}")

        log10_bf = calculate_log10_bf(stats1['bic'], stats2['bic'])
        lrt = calculate_lrt(stats1, stats2)
        print(f"Log10 BF (M2/M1): {log10_bf:.2f}, LRT p-value: {lrt['p_value']:.3e}")

        results['model_1'] = stats1
        results['model_2'] = stats2
        results['comparison_1_vs_2'] = {'log10_bayes_factor': log10_bf, 'lrt': lrt}

    output_file = Path(f"data/evals/categorylearning/{experiment_name}_{backbone_name}_analysis_map.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")
