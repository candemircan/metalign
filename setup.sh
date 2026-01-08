#!/bin/bash

# check if uv is installed
if command -v uv >/dev/null 2>&1; then
    echo "uv is already installed."
# if uv is not installed, try to install it
elif command -v curl >/dev/null 2>&1; then
    echo "installing uv using curl..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
elif command -v wget >/dev/null 2>&1; then
    echo "installing uv using wget..."
    wget -qO- https://astral.sh/uv/install.sh | sh
    
else
    echo "error: neither curl nor wget is installed. please install one to continue or install uv yourself before carrying on" >&2
    exit 1
fi

export PATH="$HOME/.local/bin:$PATH"
export UV_TORCH_BACKEND=auto
uv sync --no-install-package flash-attn


if command -v nvidia-smi >/dev/null 2>&1; then
    echo "CUDA is available. Syncing with base, dev, and cuda extras..."
    uv sync --extra dev,cuda
else
    echo "CUDA is NOT available. Syncing with base and dev extras only..."
    uv sync --extra dev
fi

uv run pre-commit install
uv run pre-commit run --all-files
uv run bin/fix_pycortex.py

# i use jq for some json processing
curl -sS https://webi.sh/jq | sh; \
source ~/.config/envman/PATH.env


# julia for mixed models
if ! command -v julia >/dev/null 2>&1; then
    curl -fsSL https://install.julialang.org | sh
else
    echo "julia is already installed."
fi
julia bin/setup_julia.jl
