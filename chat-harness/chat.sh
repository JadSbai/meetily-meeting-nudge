#!/bin/bash
# Chat with your meetings (local prototype). Usage: ./chat.sh [demo]
set -e
GGUF="$HOME/Library/Application Support/com.meetily.ai/models/summary/Qwen3.5-4B-Q4_K_M.gguf"
URL="http://localhost:8091"

# Ensure the local model server is running.
if ! curl -s "$URL/health" 2>/dev/null | grep -q ok; then
  echo "Starting Qwen 4B (llama-server) ..."
  nohup llama-server -m "$GGUF" --port 8091 -c 16384 --jinja --no-webui >/tmp/llama_server.log 2>&1 &
  for i in $(seq 1 40); do
    curl -s "$URL/health" 2>/dev/null | grep -q ok && break; sleep 3
  done
fi

DB_ARG=()
if [ "$1" = "demo" ]; then
  python3 chat_harness.py seed-demo >/dev/null
  DB_ARG=(--db demo.sqlite)
  echo "== DEMO corpus (8 synthetic meetings) =="
else
  echo "== YOUR real recordings =="
fi
exec python3 chat_harness.py chat --llm-url "$URL" "${DB_ARG[@]}"
