#!/usr/bin/env python3
"""
Meetily "chat with your meetings" — local prototype harness.

Purpose: validate the RETRIEVAL BRAIN before building any Rust/UI. Runs entirely
locally against your real Meetily SQLite DB and your Qwen 4B GGUF (served by
llama-server). No embeddings, no vector DB — the retrieval is agentic "grep":
SQLite FTS5 ranked full-text search + read-on-demand, exactly the Claude Code
model (corpus on disk, agent searches and reads).

Two modes:
  - single meeting  : stuff the whole transcript in context (transcripts are small).
  - across all       : agentic loop with search/read tools + a cheap meeting index.

Everything is stdlib. The only external piece is an OpenAI-compatible LLM server
(llama-server) on --llm-url (default http://localhost:8080).

CLI:
  python chat_harness.py stats                 # corpus + index stats
  python chat_harness.py search "budget"       # test the grep primitive alone
  python chat_harness.py ask "..."             # cross-app agentic answer
  python chat_harness.py ask --meeting <id> "..."   # single-meeting answer
  python chat_harness.py chat                   # interactive REPL
  python chat_harness.py seed-demo              # write a synthetic multi-topic demo.sqlite
Flags: --db <path>  --llm-url <url>  --model <name>  --verbose
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import urllib.request
from dataclasses import dataclass, field

DEFAULT_DB = os.path.expanduser(
    "~/Library/Application Support/com.meetily.ai/meeting_minutes.sqlite"
)
DEFAULT_LLM_URL = "http://localhost:8080"
DEFAULT_MODEL = "qwen"  # llama-server ignores the name; kept for OpenAI-compat.

# --------------------------------------------------------------------------- #
# Corpus loading (robust to schema variance across Meetily migrations)
# --------------------------------------------------------------------------- #


@dataclass
class Segment:
    meeting_id: str
    text: str
    ts: str  # display timestamp (mm:ss or raw)
    speaker: str
    order: float  # sort key (audio_start_time secs, else index)


@dataclass
class Meeting:
    id: str
    title: str
    created_at: str
    segments: list[Segment] = field(default_factory=list)
    summary: str = ""

    @property
    def full_text(self) -> str:
        return "\n".join(
            f"[{s.ts}] {s.speaker+': ' if s.speaker else ''}{s.text}"
            for s in self.segments
        )

    def blurb(self, n: int = 160) -> str:
        base = self.summary.strip() or self.full_text
        base = re.sub(r"\s+", " ", base).strip()
        return base[:n] + ("…" if len(base) > n else "")


def _cols(con: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}
    except sqlite3.Error:
        return set()


def _fmt_ts(audio_start, raw_ts) -> str:
    if audio_start is not None:
        try:
            s = int(float(audio_start))
            return f"{s // 60:02d}:{s % 60:02d}"
        except (TypeError, ValueError):
            pass
    return str(raw_ts or "")


def _flatten_summary(result_json: str) -> str:
    """summary_processes.result is JSON (sections/blocks). Flatten to plain text."""
    if not result_json:
        return ""
    try:
        d = json.loads(result_json)
    except (json.JSONDecodeError, TypeError):
        return str(result_json)
    out: list[str] = []

    def walk(x):
        if isinstance(x, str):
            if x.strip():
                out.append(x.strip())
        elif isinstance(x, list):
            for i in x:
                walk(i)
        elif isinstance(x, dict):
            for k in ("title", "name", "heading"):
                if isinstance(x.get(k), str):
                    out.append(x[k].strip())
            for k, v in x.items():
                if k in ("title", "name", "heading"):
                    continue
                walk(v)

    walk(d)
    # de-dupe consecutive
    seen, dedup = set(), []
    for line in out:
        if line not in seen:
            dedup.append(line)
            seen.add(line)
    return "\n".join(dedup)


def load_corpus(db_path: str) -> list[Meeting]:
    """Load meetings + segments + summaries. Copies the DB to avoid WAL locks."""
    if not os.path.exists(db_path):
        raise SystemExit(f"DB not found: {db_path}")
    tmp = tempfile.mkdtemp(prefix="meetily_harness_")
    copy = os.path.join(tmp, "db.sqlite")
    shutil.copy2(db_path, copy)
    for ext in ("-wal", "-shm"):
        if os.path.exists(db_path + ext):
            shutil.copy2(db_path + ext, copy + ext)
    con = sqlite3.connect(f"file:{copy}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row

    meetings: list[Meeting] = []
    mcols = _cols(con, "meetings")
    title_col = "title" if "title" in mcols else next(iter(mcols - {"id"}), "id")
    rows = con.execute(
        f"SELECT id, {title_col} AS title, "
        f"{'created_at' if 'created_at' in mcols else 'id'} AS created_at "
        f"FROM meetings ORDER BY created_at DESC"
    ).fetchall()

    tcols = _cols(con, "transcripts")
    has_astart = "audio_start_time" in tcols
    has_speaker = "speaker" in tcols
    has_tsc = "transcript_chunks" in {
        r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    scols = _cols(con, "summary_processes")

    for r in rows:
        m = Meeting(id=r["id"], title=r["title"] or "(untitled)", created_at=str(r["created_at"]))
        seg_rows = con.execute(
            "SELECT transcript AS text, timestamp AS ts, "
            f"{'audio_start_time' if has_astart else 'NULL'} AS astart, "
            f"{'speaker' if has_speaker else 'NULL'} AS speaker "
            "FROM transcripts WHERE meeting_id = ? "
            f"ORDER BY {'audio_start_time' if has_astart else 'rowid'}",
            (m.id,),
        ).fetchall()
        for i, sr in enumerate(seg_rows):
            txt = (sr["text"] or "").strip()
            if not txt:
                continue
            m.segments.append(
                Segment(
                    meeting_id=m.id,
                    text=txt,
                    ts=_fmt_ts(sr["astart"], sr["ts"]),
                    speaker=(sr["speaker"] or "").strip(),
                    order=float(sr["astart"]) if (has_astart and sr["astart"] is not None) else float(i),
                )
            )
        # Fallback to the concatenated blob if per-segment rows are absent.
        if not m.segments and has_tsc:
            row = con.execute(
                "SELECT transcript_text FROM transcript_chunks WHERE meeting_id = ? LIMIT 1",
                (m.id,),
            ).fetchone()
            if row and row[0]:
                for i, line in enumerate(str(row[0]).split("\n")):
                    if line.strip():
                        m.segments.append(Segment(m.id, line.strip(), "", "", float(i)))
        if scols:
            sr = con.execute(
                "SELECT result FROM summary_processes WHERE meeting_id = ? LIMIT 1", (m.id,)
            ).fetchone()
            if sr:
                m.summary = _flatten_summary(sr["result"])
        meetings.append(m)
    con.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return meetings


# --------------------------------------------------------------------------- #
# Retrieval primitives — the "grep": FTS5 ranked full-text search + read
# --------------------------------------------------------------------------- #


class Corpus:
    def __init__(self, meetings: list[Meeting]):
        self.meetings = meetings
        self.by_id = {m.id: m for m in meetings}
        self.fts = sqlite3.connect(":memory:")
        self.fts.execute(
            "CREATE VIRTUAL TABLE seg USING fts5("
            "meeting_id UNINDEXED, seg_idx UNINDEXED, ts UNINDEXED, speaker UNINDEXED, text)"
        )
        for m in meetings:
            for i, s in enumerate(m.segments):
                self.fts.execute(
                    "INSERT INTO seg VALUES (?,?,?,?,?)",
                    (m.id, i, s.ts, s.speaker, s.text),
                )
        self.fts.commit()

    @staticmethod
    def _fts_query(q: str) -> str:
        # Make a lenient OR query of the salient terms → high recall.
        terms = re.findall(r"[A-Za-z0-9']+", q.lower())
        stop = {"the", "a", "an", "of", "to", "in", "on", "and", "or", "what",
                "did", "we", "was", "is", "about", "for", "how", "when", "who",
                "which", "that", "our", "do", "does", "with"}
        terms = [t for t in terms if t not in stop and len(t) > 1]
        if not terms:
            terms = re.findall(r"[A-Za-z0-9']+", q.lower())
        return " OR ".join(f'"{t}"' for t in terms) or '""'

    def search(self, query: str, k: int = 8) -> list[dict]:
        """Ranked snippets across ALL meetings. The grep-with-recall primitive."""
        try:
            rows = self.fts.execute(
                "SELECT meeting_id, seg_idx, ts, speaker, text, bm25(seg) AS score "
                "FROM seg WHERE seg MATCH ? ORDER BY score LIMIT ?",
                (self._fts_query(query), k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        out = []
        for mid, idx, ts, speaker, text, score in rows:
            m = self.by_id.get(mid)
            out.append({
                "meeting_id": mid,
                "meeting_title": m.title if m else mid,
                "ts": ts,
                "speaker": speaker,
                "snippet": text if len(text) < 320 else text[:320] + "…",
                "score": round(-score, 2),
            })
        return out

    def list_meetings(self) -> list[dict]:
        return [
            {"id": m.id, "title": m.title, "date": m.created_at[:10], "topic": m.blurb()}
            for m in self.meetings
        ]

    def read_meeting(self, meeting_id: str, mode: str = "summary") -> str:
        m = self.by_id.get(meeting_id)
        if not m:
            # tolerate the model passing a title or prefix
            for mm in self.meetings:
                if meeting_id.lower() in mm.title.lower() or mm.id.startswith(meeting_id):
                    m = mm
                    break
        if not m:
            return f"(no meeting matching '{meeting_id}')"
        if mode == "full":
            return f"# {m.title} ({m.created_at[:10]})\n\n{m.full_text}"
        body = m.summary or m.full_text[:2000]
        return f"# {m.title} ({m.created_at[:10]}) — summary\n\n{body}"


# --------------------------------------------------------------------------- #
# LLM client (OpenAI-compatible; llama-server)
# --------------------------------------------------------------------------- #


class LLM:
    def __init__(self, url: str, model: str):
        self.url = url.rstrip("/")
        self.model = model

    def chat(self, messages: list[dict], *, json_schema: dict | None = None,
             temperature: float = 0.2, max_tokens: int = 1024, no_think: bool = False) -> str:
        # Qwen3 is a reasoning model: left on, it burns the token budget on
        # visible reasoning and never reaches a clean answer. Disable thinking
        # via the Qwen chat-template switch (honored by llama-server --jinja).
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if json_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "action", "strict": True, "schema": json_schema},
            }
        req = urllib.request.Request(
            f"{self.url}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())
        except urllib.error.URLError as e:
            raise SystemExit(
                f"\n[LLM unreachable at {self.url}] {e}\n"
                "Start it:  llama-server -m <Qwen3.5-4B-Q4_K_M.gguf> --port 8091 -c 16384 --jinja\n"
            )
        msg = data["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        # Some builds route the <think> block to reasoning_content, leaving
        # content empty; fall back to it. Always strip any inline think block.
        if not content and msg.get("reasoning_content"):
            content = msg["reasoning_content"].strip()
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        return content


# --------------------------------------------------------------------------- #
# Agentic loop (cross-app) — search → read → answer, constrained JSON actions
# --------------------------------------------------------------------------- #

ACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "thought": {"type": "string"},
        "action": {"type": "string", "enum": ["search", "read", "final"]},
        "query": {"type": "string"},
        "meeting_id": {"type": "string"},
        "mode": {"type": "string", "enum": ["summary", "full"]},
        "answer": {"type": "string"},
    },
    "required": ["thought", "action"],
}

SYSTEM = """You answer questions about the user's recorded meetings.
You cannot see the meetings directly — you must retrieve with tools, one action per turn.

Tools (respond with ONE JSON object per turn):
- {{"thought": "...", "action": "search", "query": "keywords"}}
    Full-text search across ALL meetings. Returns ranked snippets with meeting title + timestamp.
- {{"thought": "...", "action": "read", "meeting_id": "<id>", "mode": "summary"|"full"}}
    Read a meeting's summary (cheap) or full transcript (when you need detail/quotes).
- {{"thought": "...", "action": "final", "answer": "..."}}
    Give the final answer. Ground EVERY claim in what you retrieved and cite as
    [Meeting Title @ mm:ss]. If nothing relevant was found, say so honestly.

Rules:
- Prefer `search` first to locate evidence; then `read` the most relevant meeting(s).
- Do not invent facts. Only state what appears in retrieved snippets/transcripts.
- Be concise. Stop and answer as soon as you have enough evidence.

Meeting index (id — title — date — topic):
{index}
"""


def agentic_answer(corpus: Corpus, llm: LLM, question: str, *, max_steps: int = 6,
                   verbose: bool = False) -> str:
    index = "\n".join(
        f'- {m["id"]} — {m["title"]} — {m["date"]} — {m["topic"]}'
        for m in corpus.list_meetings()[:40]
    )
    messages = [
        {"role": "system", "content": SYSTEM.format(index=index)},
        {"role": "user", "content": question},
    ]
    for step in range(max_steps):
        raw = llm.chat(messages, json_schema=ACTION_SCHEMA, temperature=0.1)
        try:
            act = json.loads(raw)
        except json.JSONDecodeError:
            act = {"action": "final", "answer": raw}
        messages.append({"role": "assistant", "content": raw})
        a = act.get("action")
        if verbose:
            print(f"  · step {step+1}: {a}  {act.get('thought','')[:90]}", file=sys.stderr)

        if a == "search":
            hits = corpus.search(act.get("query", question), k=8)
            if verbose:
                for h in hits[:5]:
                    print(f"      ↳ {h['meeting_title']} @ {h['ts']}: {h['snippet'][:70]}", file=sys.stderr)
            obs = json.dumps(hits, ensure_ascii=False) if hits else "no matches"
            messages.append({"role": "user", "content": f"SEARCH RESULTS:\n{obs}"})
        elif a == "read":
            text = corpus.read_meeting(act.get("meeting_id", ""), act.get("mode", "summary"))
            if verbose:
                print(f"      ↳ read {act.get('meeting_id','')} ({act.get('mode','summary')}), {len(text)} chars", file=sys.stderr)
            messages.append({"role": "user", "content": f"MEETING CONTENT:\n{text[:8000]}"})
        elif a == "final":
            ans = (act.get("answer") or "").strip()
            if ans:
                return ans
            # Small models sometimes emit `final` with an empty answer after
            # several reads. Synthesize a free-form answer from what we gathered
            # (no schema constraint so the whole budget goes to prose).
            if verbose:
                print("      ↳ empty final → synthesizing from gathered context", file=sys.stderr)
            messages.append({"role": "user", "content":
                             "Write the complete final answer now, grounded in the retrieved "
                             "content above, with [Meeting Title @ mm:ss] citations."})
            return llm.chat(messages, temperature=0.2, max_tokens=900)
        else:
            messages.append({"role": "user", "content": "Unknown action. Use search, read, or final."})
    # Force a final answer from what we have.
    messages.append({"role": "user", "content": "Give your best final answer now with citations."})
    return llm.chat(messages, temperature=0.2)


def single_meeting_answer(m: Meeting, llm: LLM, question: str) -> str:
    ctx = f"# {m.title} ({m.created_at[:10]})\n"
    if m.summary:
        ctx += f"\n## Summary\n{m.summary}\n"
    ctx += f"\n## Transcript\n{m.full_text}"
    sys_prompt = (
        "You answer questions about ONE meeting, using only the transcript/summary below. "
        "Answer directly and concisely — do NOT show your reasoning. "
        "Cite timestamps like [@ mm:ss] when quoting. If the answer isn't in it, say so.\n\n"
        + ctx[:28000]
    )
    return llm.chat(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": question}],
        temperature=0.2, max_tokens=1536, no_think=True,
    )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description="Meetily chat-with-meetings harness")
    ap.add_argument("cmd", choices=["stats", "search", "ask", "chat", "seed-demo"])
    ap.add_argument("text", nargs="?", default="")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--llm-url", default=DEFAULT_LLM_URL)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--meeting", default=None, help="scope to a single meeting id")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.cmd == "seed-demo":
        from demo_corpus import seed_demo
        path = seed_demo()
        print(f"Wrote synthetic demo corpus → {path}\nUse it: python chat_harness.py ask --db {path} \"...\"")
        return

    meetings = load_corpus(args.db)
    corpus = Corpus(meetings)

    if args.cmd == "stats":
        nseg = sum(len(m.segments) for m in meetings)
        print(f"DB: {args.db}")
        print(f"Meetings: {len(meetings)} | segments (FTS rows): {nseg} | "
              f"with-summary: {sum(1 for m in meetings if m.summary)}")
        for m in meetings[:40]:
            print(f"  - {m.id}  {m.title[:44]:44}  {len(m.segments):4d} seg  {m.created_at[:10]}")
        return

    if args.cmd == "search":
        for h in corpus.search(args.text or "", k=10):
            print(f"[{h['score']:5}] {h['meeting_title'][:30]:30} @ {h['ts']:>6}  {h['snippet'][:90]}")
        return

    llm = LLM(args.llm_url, args.model)

    if args.cmd == "ask":
        if args.meeting:
            m = corpus.by_id.get(args.meeting)
            if not m:
                raise SystemExit(f"no meeting {args.meeting}")
            print(single_meeting_answer(m, llm, args.text))
        else:
            print(agentic_answer(corpus, llm, args.text, verbose=args.verbose))
        return

    if args.cmd == "chat":
        scope = args.meeting
        print("Interactive. Commands: /meeting <id>, /all, /list, /quit")
        while True:
            try:
                q = input("\n› ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not q:
                continue
            if q in ("/quit", "/exit"):
                break
            if q == "/list":
                for m in meetings:
                    print(f"  {m.id}  {m.title}")
                continue
            if q == "/all":
                scope = None; print("(scope: all meetings)"); continue
            if q.startswith("/meeting "):
                scope = q.split(" ", 1)[1].strip(); print(f"(scope: {scope})"); continue
            if scope:
                m = corpus.by_id.get(scope)
                print(single_meeting_answer(m, llm, q) if m else f"no meeting {scope}")
            else:
                print(agentic_answer(corpus, llm, q, verbose=args.verbose))


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
