from pathlib import Path

import pandas as pd
from groupBMC import GroupBMC
from scipy.stats import wilcoxon


def run_bmc(df, col1='base_nll', col2='meta_nll'):
    "Run Group Bayesian Model Comparison between two models"
    p_df = df.groupby('participant_id')[[col1, col2]].sum()
    L = -p_df[[col1, col2]].values.T

    bmc = GroupBMC(L)
    res = bmc.get_result()

    _, p_wilcoxon = wilcoxon(p_df[col2], p_df[col1])

    return {
        f'{col1}_total': p_df[col1].sum(),
        f'{col2}_total': p_df[col2].sum(),
        'wilcoxon_p': p_wilcoxon,
        f'freq_{col1.replace("_nll","")}': res.frequency_mean[0],
        f'freq_{col2.replace("_nll","")}': res.frequency_mean[1],
        f'pxp_{col1.replace("_nll","")}': res.protected_exceedance_probability[0],
        f'pxp_{col2.replace("_nll","")}': res.protected_exceedance_probability[1]
    }

def compare_one(h5_path, dataset):
    "Run comparisons for a single trial HDF5"
    trial_df = pd.read_hdf(h5_path, key='trials')
    stem = h5_path.stem.replace('_trials', '')
    experiment_name, backbone_name = stem.rsplit('_', 1)

    has_main = 'main_nll' in trial_df.columns
    stats_rows = []

    # Full BMC: meta vs base
    res = run_bmc(trial_df, 'base_nll', 'meta_nll')
    base_acc, meta_acc = trial_df['base_correct'].mean(), trial_df['meta_correct'].mean()
    stats_rows.append({'dataset': dataset, 'experiment': experiment_name, 'backbone': backbone_name,
                       'subset': 'all', 'comparison': 'meta_vs_base', **res,
                       'base_acc': base_acc, 'meta_acc': meta_acc})

    # Full BMC: meta vs main (ablations only)
    if has_main:
        main_acc = trial_df['main_correct'].mean()
        res_main = run_bmc(trial_df, 'main_nll', 'meta_nll')
        stats_rows.append({'dataset': dataset, 'experiment': experiment_name, 'backbone': backbone_name,
                           'subset': 'all', 'comparison': 'meta_vs_main', **res_main,
                           'main_acc': main_acc, 'meta_acc': meta_acc})

    # Per triplet type (LEVELS only)
    if 'triplet_type' in trial_df.columns:
        for tt in sorted(trial_df['triplet_type'].unique()):
            tt_df = trial_df[trial_df['triplet_type'] == tt]
            if tt_df['participant_id'].nunique() < 3: continue

            tt_base_acc, tt_meta_acc = tt_df['base_correct'].mean(), tt_df['meta_correct'].mean()
            res_tt = run_bmc(tt_df, 'base_nll', 'meta_nll')
            stats_rows.append({'dataset': dataset, 'experiment': experiment_name, 'backbone': backbone_name,
                               'subset': tt, 'comparison': 'meta_vs_base', **res_tt,
                               'base_acc': tt_base_acc, 'meta_acc': tt_meta_acc})

            if has_main:
                tt_main_acc = tt_df['main_correct'].mean()
                res_tt_main = run_bmc(tt_df, 'main_nll', 'meta_nll')
                stats_rows.append({'dataset': dataset, 'experiment': experiment_name, 'backbone': backbone_name,
                                   'subset': tt, 'comparison': 'meta_vs_main', **res_tt_main,
                                   'main_acc': tt_main_acc, 'meta_acc': tt_meta_acc})

    return stats_rows

if __name__ == "__main__":
    all_rows = []
    for dataset in ['things', 'levels']:
        eval_path = Path(f"data/evals/{dataset}o1o")
        h5s = sorted(eval_path.glob("*_trials.h5"))
        if not h5s:
            print(f"No trial HDF5s found in {eval_path}")
            continue
        for h5 in h5s:
            print(f"Processing {h5.name} ({dataset})...")
            all_rows.extend(compare_one(h5, dataset))

    if not all_rows:
        print("No trial CSVs found")
        raise SystemExit(1)

    stats_df = pd.DataFrame(all_rows)
    out = Path("data/evals/o1o_stats.csv")
    stats_df.to_csv(out, index=False)
    print(f"\nSaved combined stats to {out}")
    print(stats_df.to_string(index=False))
