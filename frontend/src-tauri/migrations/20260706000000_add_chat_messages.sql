-- Chat with your meetings — persisted conversation history.
--
-- Search (FTS5) is intentionally NOT defined here: it is built in-memory at
-- query time from the transcripts table (mirrors the validated harness), so the
-- recording-critical transcripts insert path is never touched by triggers.
--
-- `scope` is either a meeting id (chat with one recording) or the literal
-- 'all' (chat across every recording). `citations` is a JSON array of
-- {meeting_id, title, ts} the assistant grounded its answer in.

CREATE TABLE IF NOT EXISTS chat_messages (
    id          TEXT PRIMARY KEY,
    scope       TEXT NOT NULL,
    role        TEXT NOT NULL,            -- 'user' | 'assistant'
    content     TEXT NOT NULL,
    citations   TEXT,                     -- JSON array, nullable
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chat_messages_scope
    ON chat_messages (scope, created_at);
