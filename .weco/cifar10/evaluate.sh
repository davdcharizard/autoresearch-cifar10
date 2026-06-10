#!/usr/bin/env bash
# Weco eval command for the CIFAR-10 training task.
# Runs from the repo root so `uv run` picks up pyproject.toml and prepare.py.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Pin to a single GPU (task constraint). Defaults to GPU 1 because GPU 0 is in
# use by another experiment here; override by exporting CUDA_VISIBLE_DEVICES.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

exec uv run python .weco/cifar10/evaluate.py
