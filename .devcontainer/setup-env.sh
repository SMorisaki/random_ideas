#!/usr/bin/env bash
# .devcontainer/setup-env.sh
# コンテナ作成時（postCreateCommand）に自動実行。
# /workspace/.conda-linux が無ければリポジトリの定義から作る。
# Dockerfile 側で PATH / CONDA_PREFIX がこの env を指しているので、
# これ以降は Claude Code も Codex も `python` `pip` `pytest` がこの env になる。
set -euo pipefail

WORKSPACE="${WORKSPACE_DIR:-/workspace}"
ENV_DIR="$WORKSPACE/.conda-linux"
cd "$WORKSPACE"

if [ -d "$ENV_DIR" ]; then
  echo "[env] $ENV_DIR already exists"
else
  if [ -f environment.yml ]; then
    echo "[env] creating $ENV_DIR from environment.yml"
    mamba env create -y -p "$ENV_DIR" -f environment.yml
  else
    # environment.yml が無いリポジトリ向けの最低限（必要に応じて書き換える）
    echo "[env] creating $ENV_DIR (python only; no environment.yml found)"
    mamba create -y -p "$ENV_DIR" python=3.12 pip
  fi
  if [ -f requirements.txt ]; then
    echo "[env] pip install -r requirements.txt"
    "$ENV_DIR/bin/pip" install -r requirements.txt
  fi
fi

echo "[env] python = $(command -v python)"
