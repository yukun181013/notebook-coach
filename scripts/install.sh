#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
CODEX_ROOT=${CODEX_HOME:-"$HOME/.codex"}
SKILLS_DIR="$CODEX_ROOT/skills"
DESTINATION="$SKILLS_DIR/notebook-coach"
MODE=install

if [ "${1:-}" = "--link-only" ]; then
    MODE=link-only
    shift
fi
if [ "$#" -ne 0 ]; then
    echo "usage: $0 [--link-only]" >&2
    exit 2
fi

destination_matches=false
if [ -e "$DESTINATION" ] || [ -L "$DESTINATION" ]; then
    if resolved=$(CDPATH= cd -- "$DESTINATION" 2>/dev/null && pwd -P) && [ "$resolved" = "$ROOT" ]; then
        destination_matches=true
    else
        echo "refusing to replace unrelated destination: $DESTINATION" >&2
        exit 1
    fi
fi

if [ "$MODE" = install ]; then
    if ! command -v python3.11 >/dev/null 2>&1; then
        echo "Python 3.11 is required." >&2
        exit 1
    fi
    if [ ! -x "$ROOT/.venv/bin/python" ]; then
        python3.11 -m venv "$ROOT/.venv"
    fi
    "$ROOT/.venv/bin/python" -m pip install -e "$ROOT"
fi

if [ "$destination_matches" = false ]; then
    mkdir -p "$SKILLS_DIR"
    ln -s "$ROOT" "$DESTINATION"
fi

echo "Notebook Coach linked at $DESTINATION"
