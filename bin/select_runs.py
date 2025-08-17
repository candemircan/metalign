import os
import shutil
import sqlite3
import tomllib
from ast import literal_eval
from pathlib import Path

import pandas as pd
from fastcore.script import call_parse


@call_parse
def main(
    checkpoints_dir: Path = Path("data/checkpoints"),
    configs_dir: Path = Path("data/configs"),
    db_name: str = "metalign.db",
):
    "Finds the best run for each model, removes checkpoints of other runs, and copies the best run configs."
    hf_root = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    db_path = Path(hf_root) / "trackio" / db_name
    if not db_path.exists():
        print(f"Database not found at {db_path}")
        return

    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql("SELECT * from metrics", conn)

    run_names, model_names, best_metrics = [], [], []
    for _, row in df.iterrows():
        if "best_eval_accuracy" in row.metrics:
            metrics_dict = literal_eval(row.metrics)
            run_names.append(row.run_name)
            model_names.append(row.run_name.split("_")[0])
            best_metrics.append(metrics_dict["best_eval_accuracy"])

    if not run_names:
        print("No runs with 'best_eval_accuracy' found in the database.")
        return

    df_summary = pd.DataFrame({"run_name": run_names, "model_name": model_names, "best_eval_accuracy": best_metrics})
    df_best = df_summary.loc[df_summary.groupby("model_name")["best_eval_accuracy"].idxmax()]
    best_runs = set(df_best.run_name.values)


    for folder in checkpoints_dir.iterdir():
        if folder.is_dir() and folder.name not in best_runs:
            shutil.rmtree(folder)

    for folder in checkpoints_dir.iterdir():
        if folder.is_dir() and folder.name in best_runs:
            config_file = folder / "config.toml"
            if not config_file.exists():
                found_config = False
                for toml_file in configs_dir.glob("*.toml"):
                    with open(toml_file, "rb") as f:
                        config = tomllib.load(f)
                    if config.get("name") == folder.name:
                        with open(config_file, "w") as f:
                            f.write(tomllib.dumps(config))
                        found_config = True
                        break
                if not found_config:
                    print(f"Warning: Could not find a matching config for {folder.name} in {configs_dir}")