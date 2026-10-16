#!/usr/bin/env bash
# .devcontainer/codex-check.sh
# コンテナ起動時（postStartCommand）に自動実行する Codex の点検スクリプト。
#   1. ログイン状態を確認し、未ログインならデバイスコード認証の案内を出す
#      （認証そのものは人がブラウザで承認する必要があるので自動化しない。
#        ~/.codex を volume にしてあれば一度で済む）
#   2. ~/.codex/config.toml に既定値を書く（無い場合のみ）
#   3. Codex のサンドボックスがこのコンテナで動くか実地に確認し、
#      結果を ~/.codex/sandbox_mode に書く（run_codex.sh が読む）
set -u

CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
WORKSPACE="${WORKSPACE_DIR:-/workspace}"
mkdir -p "$CODEX_HOME"

# --- 1. login -------------------------------------------------------------
if codex login status >/dev/null 2>&1; then
  echo "[codex] logged in"
else
  cat <<'MSG'
[codex] NOT logged in.
        コンテナ内で一度だけ次を実行し、表示された URL を手元のブラウザで開いてコードを入力してください:
            sh run_codex.sh login --device-auth
MSG
fi

# --- 2. config defaults ----------------------------------------------------
CFG="$CODEX_HOME/config.toml"
touch "$CFG"
# ChatGPT 認証を既定にする
grep -q '^preferred_auth_method' "$CFG" || echo 'preferred_auth_method = "chatgpt"' >> "$CFG"
# workspace-write サンドボックス内でもネットワークを許可する
# （pip / mamba install をエージェントにさせるため。外向き通信の制限はコンテナの
#   ファイアウォール側で行う）
if ! grep -q '^\[sandbox_workspace_write\]' "$CFG"; then
  printf '\n[sandbox_workspace_write]\nnetwork_access = true\n' >> "$CFG"
fi

# --- 3. sandbox probe ------------------------------------------------------
# workspace-write 相当の設定で、作業ディレクトリへの書き込みが通るかを確認する。
# bwrap / Landlock がコンテナで使えない場合や、bind mount に書けない場合はここで失敗する。
cd "$WORKSPACE" || exit 0
if codex sandbox -c 'sandbox_mode="workspace-write"' \
     sh -c 'touch .codex_sandbox_probe && rm .codex_sandbox_probe' >/dev/null 2>&1; then
  echo workspace-write > "$CODEX_HOME/sandbox_mode"
  echo "[codex] sandbox OK -> workspace-write"
else
  echo danger-full-access > "$CODEX_HOME/sandbox_mode"
  echo "[codex] sandbox NOT available in this container -> run_codex.sh will use danger-full-access"
  echo "        (isolation is provided by the container: firewall + mounts)"
fi
