# metalign

> Meta-learning as a principle for human-like visual representations

`metalign` trains a causal transformer on visual features from pretrained vision models and sparse autoencoders. The model performs online category and reward learning, and its learned representations are compared against human behavioral benchmarks and fMRI responses. The repository contains the model code, experiment scripts, configs, and DataLad-tracked derived artifacts needed to reproduce the main analyses.

## Setup

Clone the repository and install dependencies with `uv`:

```bash
git clone https://github.com/candemircan/metalign.git
cd metalign
./setup.sh
```

The project uses Python 3.12.5. Most experiment scripts should be run with `uv run`.

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
git annex enableremote osf-storage
uv run datalad get data figures
```

## Running Experiments

Generate configs, train models, evaluate them, and compare model families:

```bash
uv run bin/generate_configs.py
bash bin/train.sh
bash bin/eval.sh
bash bin/compare.sh
```

The `*.sh` and `*.slurm` wrappers reflect my SLURM setup and may need local adjustment. For non-SLURM runs, call the corresponding Python scripts directly with `uv run`.

## Tests

```bash
bash bin/run_tests.sh metalign/*.py
uv run -m metalign.model
```

## Repository Structure

```text
bin/       experiment, evaluation, comparison, extraction, and plotting scripts
data/      configs, DataLad-tracked derived artifacts, and local external data
figures/   generated figures (DataLad-tracked, stored on OSF)
logs/      local job logs
metalign/  package code and module test harnesses
```

## Citation

Citation coming soon.
