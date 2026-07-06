# Chat-with-meetings — local prototype harness

Validates the "chat with your meetings" **retrieval brain** before building any
Rust/UI. Runs 100% locally against your real Meetily SQLite DB + your Qwen 4B
GGUF. **No embeddings / no vector DB** — retrieval is agentic "grep": SQLite
**FTS5** ranked full-text search + read-on-demand (the Claude Code model).

## 1. Serve the model (uses the GGUF Meetily already downloaded)

```bash
llama-server \
  -m ~/Library/Application\ Support/com.meetily.ai/models/summary/Qwen3.5-4B-Q4_K_M.gguf \
  --port 8091 -c 16384 --jinja --no-webui
```
(install once: `brew install llama.cpp`)

## 2. Drive it

```bash
cd chat-harness
LLM=http://localhost:8091

# synthetic multi-topic corpus (for testing cross-app recall)
python3 chat_harness.py seed-demo

# interactive REPL — cross-app agentic chat
python3 chat_harness.py chat --db demo.sqlite --llm-url $LLM
#   /meeting <id>  scope to one meeting   ·   /all  back to all   ·   /list  ·  /quit

# one-shot
python3 chat_harness.py ask "What did we decide about the Q3 budget?" --db demo.sqlite --llm-url $LLM --verbose
python3 chat_harness.py ask --meeting demo-06 "Embeddings or FTS, and why?" --db demo.sqlite --llm-url $LLM

# your REAL meetings (default --db is the live Meetily DB)
python3 chat_harness.py stats
python3 chat_harness.py ask --meeting <id> "3-bullet summary + action items" --llm-url $LLM

# test the grep primitive alone (no LLM)
python3 chat_harness.py search "security incident root cause" --db demo.sqlite
```

`--verbose` prints every tool call (search/read) so you can watch the retrieval.

## What this proves → what ports into the app

- **Single meeting** = whole transcript in context (no retrieval). Direct + cited.
- **Cross-app** = agentic loop over 3 primitives: `search` (FTS5 BM25), `read`
  (summary/full), `final` (cited answer) + a cheap summary index for routing.
- Runs on a 4B local model with `enable_thinking:false`. Handles multi-hop,
  disambiguation, and honestly says "not found".

Port plan: FTS5 virtual table (migration) + these 3 tools as Tauri commands +
the loop + a chat UI panel (3rd tab) + streaming. No new model, no vector store.
