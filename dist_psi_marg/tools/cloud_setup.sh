#!/usr/bin/env bash
# dist_psi_marg/tools/cloud_setup.sh
# One-time setup for a *cloud* session (no devcontainer volumes, no conda preinstalled).
# Run from the repository root:  bash dist_psi_marg/tools/cloud_setup.sh
#
#  1. Codex authentication: if the environment variable CODEX_AUTH_JSON is set (a cloud
#     environment secret holding the content of the local ~/.codex/auth.json), write it to
#     $CODEX_HOME/auth.json (mode 600). The value is never printed.
#  2. Codex config defaults (same as .devcontainer/codex-check.sh): ChatGPT auth, network in
#     the workspace-write sandbox.
#  3. micromamba: micro.mamba.pm and GitHub releases are blocked by the network policy, but
#     conda.anaconda.org is reachable, so the binary is taken from the conda-forge package.
#  4. Conda environment from the repository's environment.yml at $CONDA_ENV_DIR
#     (default /home/user/conda-env). Prepend "$CONDA_ENV_DIR/bin" to PATH afterwards.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
CONDA_ENV_DIR="${CONDA_ENV_DIR:-/home/user/conda-env}"
MM_BIN="${MM_BIN:-$HOME/.local/bin/micromamba}"
MM_PKG="${MM_PKG:-micromamba-2.9.0-0.tar.bz2}"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.micromamba}"
if [ -f /root/.ccr/ca-bundle.crt ]; then
  export SSL_CERT_FILE=/root/.ccr/ca-bundle.crt REQUESTS_CA_BUNDLE=/root/.ccr/ca-bundle.crt
fi

# --- 1. Codex auth from secret -------------------------------------------------------
mkdir -p "$CODEX_HOME"
if [ -n "${CODEX_AUTH_JSON:-}" ]; then
  if [ ! -s "$CODEX_HOME/auth.json" ]; then
    umask 077
    printf '%s' "$CODEX_AUTH_JSON" > "$CODEX_HOME/auth.json"
    umask 022
    echo "[setup] wrote $CODEX_HOME/auth.json from CODEX_AUTH_JSON"
  else
    echo "[setup] $CODEX_HOME/auth.json already exists; left unchanged"
  fi
else
  echo "[setup] CODEX_AUTH_JSON not set; Codex authentication not configured"
fi

# --- 2. Codex config defaults --------------------------------------------------------
CFG="$CODEX_HOME/config.toml"
touch "$CFG"
grep -q '^preferred_auth_method' "$CFG" || echo 'preferred_auth_method = "chatgpt"' >> "$CFG"
if ! grep -q '^\[sandbox_workspace_write\]' "$CFG"; then
  printf '\n[sandbox_workspace_write]\nnetwork_access = true\n' >> "$CFG"
fi

# --- 3. micromamba from conda-forge --------------------------------------------------
if [ ! -x "$MM_BIN" ]; then
  TMPD="$(mktemp -d)"
  echo "[setup] downloading $MM_PKG from conda-forge"
  curl -sS -o "$TMPD/$MM_PKG" "https://conda.anaconda.org/conda-forge/linux-64/$MM_PKG"
  mkdir -p "$TMPD/x" && tar xjf "$TMPD/$MM_PKG" -C "$TMPD/x"
  mkdir -p "$(dirname "$MM_BIN")" && cp "$TMPD/x/bin/micromamba" "$MM_BIN" && chmod +x "$MM_BIN"
  rm -rf "$TMPD"
fi
"$MM_BIN" --version

# --- 4. conda env from environment.yml -----------------------------------------------
if [ ! -x "$CONDA_ENV_DIR/bin/python" ]; then
  echo "[setup] creating $CONDA_ENV_DIR from $REPO_ROOT/environment.yml"
  "$MM_BIN" create -y -p "$CONDA_ENV_DIR" -f "$REPO_ROOT/environment.yml" >/dev/null
else
  echo "[setup] $CONDA_ENV_DIR exists; syncing with environment.yml"
  "$MM_BIN" install -y -p "$CONDA_ENV_DIR" -f "$REPO_ROOT/environment.yml" >/dev/null
fi
"$CONDA_ENV_DIR/bin/python" -c "import numpy, scipy, numba, matplotlib, h5py, pytest; print('[setup] env OK')"
echo "[setup] prepend to PATH: $CONDA_ENV_DIR/bin"
