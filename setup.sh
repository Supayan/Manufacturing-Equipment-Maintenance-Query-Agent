#!/usr/bin/env bash
# Bootstraps the project: installs uv if missing, creates a .venv on the
# pinned Python version, and installs requirements.txt into it.
set -euo pipefail

PYTHON_VERSION="3.11"
VENV_DIR=".venv"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if ! command -v uv >/dev/null 2>&1; then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Creating virtual environment (Python $PYTHON_VERSION)..."
uv venv --python "$PYTHON_VERSION" "$VENV_DIR"

echo "Installing pinned requirements..."
uv pip install -r requirements.txt --python "$VENV_DIR"

echo ""
echo "Setup complete. Activate with:"
echo "  source $VENV_DIR/bin/activate"
