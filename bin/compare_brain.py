from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _load_perm_null(eval_path, stem):
    "Load permutation null distributions (R2 diffs per ROI) for a given eval stem"
    perm_files = sorted(eval_path.glob(f"{stem}_perm_*.npy"))
    if not perm_files: return {}
    null = {}
    for f in perm_files:
        perm_res = np.load(f, allow_pickle=True).item()
        for roi, diff in perm_res.items():
            null.setdefault(roi, []).append(diff)
    return {k: np.array(v) for k, v in null.items()}


def compare_brain():
    "Compare brain encoding r2 across all real (non-permutation) npy files"
    eval_path = Path("data/evals/brain")
    npys = sorted(f for f in eval_path.glob("*.npy") if "_perm_" not in f.stem)
    if not npys:
        print("No npy files found in", eval_path)
        raise SystemExit(1)

    stats_rows, roi_rows = [], []
    for npy in npys:
        stem = npy.stem
        par_no, rest = stem.split('_', 1)
        exp_name, bb_name = rest.rsplit('_', 1)

        res = np.load(npy, allow_pickle=True).item()

        # Voxel-level (summary over all voxels)
        r2_b, r2_m = res['r2_base'].mean(axis=0), res['r2_meta'].mean(axis=0)

        row = {'participant': par_no, 'experiment': exp_name, 'backbone': bb_name,
               'mean_r2_base': r2_b.mean(), 'mean_r2_meta': r2_m.mean(),
               'n_voxels': len(r2_b)}

        _, p_r2 = wilcoxon(r2_m, r2_b); row['wilcoxon_p_r2'] = p_r2
        stats_rows.append(row)

        # ROI-level
        if 'roi_results' in res:
            null = _load_perm_null(eval_path, stem)
            for roi, r in res['roi_results'].items():
                roi_row = {
                    'participant': par_no, 'experiment': exp_name, 'backbone': bb_name, 'roi': roi,
                    'r2_base': r['r2_base'], 'r2_meta': r['r2_meta'],
                    'diff_mean': r['diff_mean'],
                }
                # Permutation p-value for meta vs base diff
                if roi in null and len(null[roi]) > 0:
                    obs_diff = r['r2_meta'] - r['r2_base']
                    roi_row['p_diff'] = (np.abs(null[roi]) >= np.abs(obs_diff)).mean()

                if 'r2_main' in r:
                    roi_row['r2_main'] = r['r2_main']
                    # Permutation p-value for meta vs main diff
                    if roi in null and len(null[roi]) > 0:
                        obs_diff_main = r['r2_meta'] - r['r2_main']
                        roi_row['p_meta_vs_main'] = (np.abs(null[roi]) >= np.abs(obs_diff_main)).mean()

                roi_rows.append(roi_row)

        print(f"{stem}: r2_base={r2_b.mean():.4f}, r2_meta={r2_m.mean():.4f}")

    stats_df = pd.DataFrame(stats_rows)
    stats_df.to_csv(eval_path / "stats.csv", index=False)

    if roi_rows:
        roi_df = pd.DataFrame(roi_rows)
        roi_df.to_csv(eval_path / "roi_stats.csv", index=False)
        print("\nROI Summary (Top 10 ROIs by r2_meta):")
        print(roi_df.sort_values('r2_meta', ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    compare_brain()
