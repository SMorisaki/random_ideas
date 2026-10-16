#!/bin/sh
# Codex を ChatGPT アカウントで動かすラッパー（API キー課金に切り替わらない）。
# 使い方は記事と同じ:  sh run_codex.sh exec --sandbox workspace-write "..."
#
# 追加機能: コンテナ起動時に .devcontainer/codex-check.sh が書いた
#   $CODEX_HOME/sandbox_mode  （workspace-write または danger-full-access）
# を読み、Codex のサンドボックスが使えない環境では、引数中の
#   --sandbox <mode> / -s <mode> / --sandbox=<mode>
# を danger-full-access に置き換える（隔離はコンテナ側が担う）。
# ファイルが無ければ何も置き換えない（= 記事と同じ挙動）。

unset OPENAI_API_KEY CODEX_API_KEY

CODEX_HOME="${CODEX_HOME:-$HOME/.codex}"
MODE=$(cat "$CODEX_HOME/sandbox_mode" 2>/dev/null || echo workspace-write)

if [ "$MODE" = "danger-full-access" ]; then
  n=$#
  i=0
  replace_next=0
  while [ "$i" -lt "$n" ]; do
    a=$1
    shift
    if [ "$replace_next" = 1 ]; then
      a=danger-full-access
      replace_next=0
    else
      case "$a" in
        --sandbox|-s)       replace_next=1 ;;
        --sandbox=*|-s=*)   a="--sandbox=danger-full-access" ;;
      esac
    fi
    set -- "$@" "$a"
    i=$((i + 1))
  done
  echo "[run_codex] sandbox unavailable in this container -> using danger-full-access (isolation = container)" >&2
fi

exec codex --config 'preferred_auth_method="chatgpt"' "$@"
