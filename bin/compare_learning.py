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

def compare_one(h5_path):
    "Run comparisons for a single trial HDF5"
    trial_df = pd.read_hdf(h5_path, key='trials')
    stem = h5_path.stem.replace('_trials', '')
    experiment_name, backbone_name = stem.rsplit('_', 1)

    stats_rows = []

    # Full BMC: meta vs base (always present)
    res = run_bmc(trial_df, 'base_nll', 'meta_nll')
    stats_rows.append({'experiment': experiment_name, 'backbone': backbone_name,
                       'comparison': 'meta_vs_base', **res,
                       'n_participants': trial_df['participant_id'].nunique(),
                       'n_trials': len(trial_df)})

    # Full BMC: meta vs main (ablations only)
    if 'main_nll' in trial_df.columns:
        res_main = run_bmc(trial_df, 'main_nll', 'meta_nll')
        stats_rows.append({'experiment': experiment_name, 'backbone': backbone_name,
                           'comparison': 'meta_vs_main', **res_main,
                           'n_participants': trial_df['participant_id'].nunique(),
                           'n_trials': len(trial_df)})

    return stats_rows

if __name__ == "__main__":
    for eval_path in [Path("data/evals/categorylearning"), Path("data/evals/rewardlearning")]:
        h5s = sorted(eval_path.glob("*_trials.h5"))
        if not h5s:
            print("No trial HDF5s found in", eval_path)
            continue

        all_rows = []
        for h5 in h5s:
            print(f"Processing {h5.name}...")
            all_rows.extend(compare_one(h5))

        stats_df = pd.DataFrame(all_rows)
        out = eval_path / "stats.csv"
        stats_df.to_csv(out, index=False)
        print(f"\nSaved combined stats to {out}")
        print(stats_df.to_string(index=False))
