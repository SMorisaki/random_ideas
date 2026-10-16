#!/usr/bin/env bash
# .devcontainer/claude-setup.sh
# Runs at container start (postStartCommand).
#
# Merges the Claude Code defaults in .devcontainer/claude-settings.json
# (model, effort, Remote Control at startup) into the *user* settings file
# ~/.claude/settings.json. Keys that are not listed there (theme, ...) are kept.
#
# Why here and not in the Dockerfile: ~/.claude is a Docker volume, so anything
# written at image-build time is shadowed once the volume is mounted.
# Why user settings: `remoteControlAtStartup` is only honoured in user settings,
# a `true` in project settings (.claude/settings.json) is ignored.
set -euo pipefail

CLAUDE_HOME="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
SRC="$(cd "$(dirname "$0")" && pwd)/claude-settings.json"
DST="$CLAUDE_HOME/settings.json"

mkdir -p "$CLAUDE_HOME"
[ -s "$DST" ] || echo '{}' > "$DST"

# Deep merge: values from claude-settings.json win, everything else is preserved.
jq -s '.[0] * .[1]' "$DST" "$SRC" > "$DST.tmp"
mv "$DST.tmp" "$DST"

echo "[claude] defaults merged into $DST:"
jq -c '{model, modelSettings, remoteControlAtStartup}' "$DST"

if [ ! -s "$CLAUDE_HOME/.credentials.json" ]; then
  echo "[claude] not logged in: start 'claude' and run /login (Remote Control needs a claude.ai login)"
fi
