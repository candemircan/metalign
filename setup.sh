#!/bin/bash

# Check if uv is installed
if command -v uv >/dev/null 2>&1; then
    echo "uv is already installed."

# If uv is not installed, try to install it
elif command -v curl >/dev/null 2>&1; then
    echo "installing uv using curl..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
elif command -v wget >/dev/null 2>&1; then
    echo "installing uv using wget..."
    wget -qO- https://astral.sh/uv/install.sh | sh
else
    echo "error: neither curl nor wget is installed. please install one to continue or install uv yourself before using this template." >&2
    exit 1
fi

export UV_TORCH_BACKEND=auto

# Check if --disposable flag is provided
if [[ "$1" == "--disposable" ]]; then
    export TMPDIR=/localscratch/
    cd /localscratch/
    uv venv -p 3.12.5 .metarep
    source .metarep/bin/activate
    cd /lustre/groups/hcai/workspace/can.demircan/metarep
else
    uv venv -p 3.12.5  .venv
    source .venv/bin/activate
fi

uv pip install -e '.[dev]'
uv pip install --no-build-isolation '.[linux]'
pre-commit install
pre-commit run --all-files

# add example* to .gitignore
echo "example*" >> .gitignore

