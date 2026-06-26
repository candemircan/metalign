# metalign

> Meta-learning as a principle for human-like visual representations

`metalign` trains a causal transformer on visual features from pretrained vision models and sparse autoencoders. The model performs online category and reward learning, and its learned representations are compared against human behavioral benchmarks and fMRI responses. The repository contains the model code, experiment scripts, configs, and DataLad-tracked derived artifacts needed to reproduce the main analyses.

## System requirements

- **Python** 3.12.5.
- **Dependencies** are managed with `uv` and fully pinned in `uv.lock`.
- **Operating systems tested:** macOS 15 (Apple Silicon) for evaluation, comparison, and plotting; Linux (SLURM cluster) for feature extraction and training.
- **Non-standard hardware:** an NVIDIA CUDA GPU is required for feature extraction and model training (an optional `flash-attn` extra is installed automatically when a GPU is detected). Evaluation, comparison, and plotting run on CPU.

## Installation

Clone the repository and install dependencies with `uv`:

```bash
git clone https://github.com/candemircan/metalign.git
cd metalign
./setup.sh
```

`setup.sh` installs `uv` (if absent), syncs the pinned dependencies, installs `git-annex`, and sets up pre-commit hooks. Most experiment scripts should be run with `uv run`.

**Typical install time:** a few minutes on a normal desktop, dominated by downloading and building PyTorch and its dependencies (longer on the first run with no cached wheels).

## Data

There are two kinds of data:

- External source datasets are downloaded into `data/external/` and are not tracked.
- Derived artifacts, such as extracted features, checkpoints, configs, evaluation outputs, and generated figures, are tracked with DataLad/git-annex and stored on OSF.

To download external source datasets:

```bash
bash bin/get_data.sh
```

By default, this also points pycortex's user-level filestore config to `data/external/brain_data/surfaces`. To download data without changing pycortex config:

```bash
bash bin/get_data.sh --skip-pycortex-config
```

To retrieve DataLad-tracked derived artifacts:

```bash
uv run datalad get data figures
```

## Demo

The core modules ship self-contained test harnesses that run on small synthetic tensors, requiring no data download and no GPU:

```bash
uv run -m metalign.model          # single module
bash bin/run_tests.sh metalign/*.py   # all modules
```

**Expected output:** each harness runs forward/loss passes and shape/value assertions; a successful run completes silently and exits 0 (a failed assertion raises and exits non-zero). **Expected run time:** a few seconds per module on CPU.

For a demo on real data, fetch the DataLad artifacts (above) and regenerate a paper figure — this needs no GPU or SLURM:

```bash
uv run bin/plot_evals.py
```

**Expected output:** the corresponding figure files written under `figures/`. **Expected run time:** under a minute on CPU.

## Instructions for use

To run the full pipeline on your data — extract features, generate configs, train, evaluate, compare, and plot:

```bash
sbatch bin/extract.slurm
uv run bin/generate_configs.py
bash bin/train.sh
bash bin/eval.sh
bash bin/compare.sh
bash bin/plot.sh
```

The artifacts fetched via `datalad get` already include the extracted features and the paper figures, so feature extraction and plotting are only needed to regenerate them from scratch.

The `*.sh` and `*.slurm` wrappers reflect my SLURM setup and may need local adjustment. For non-SLURM runs, call the corresponding Python scripts directly with `uv run` (the plot scripts take no arguments, e.g. `uv run bin/plot_evals.py`).

## Repository Structure

```text
bin/       experiment, evaluation, comparison, extraction, and plotting scripts
data/      configs, DataLad-tracked derived artifacts, and local external data
figures/   generated figures (DataLad-tracked, stored on OSF)
logs/      local job logs
metalign/  package code and module test harnesses
```

## License

Released under the [MIT License](LICENSE).

## Citation

Citation coming soon.
