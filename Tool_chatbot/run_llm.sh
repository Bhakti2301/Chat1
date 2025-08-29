#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/.venv"

PYTHON_BIN="python3"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "[setup] Creating virtual environment..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

echo "[setup] Ensuring dependencies..."
python -m pip install --upgrade pip >/dev/null 2>&1
python -m pip install langgraph openai >/dev/null 2>&1

echo "[run] Starting LLM + memory chatbot..."
python "$SCRIPT_DIR/LangGraph_llm_memory_chatbot.py" "$@"

