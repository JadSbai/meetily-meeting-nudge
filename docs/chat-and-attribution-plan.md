# Chat with Attributed Meetings — Delivery Plan

Status legend: ✅ done · 🔨 in progress · ⏳ planned · ⚠ risk/attention

Goal: **"chat with your meetings"** — ask questions to one recording or all of them —
where the transcript is **attributed** (who said what: you, and each named other
person), fully **local** (Qwen 4B + local audio), **no bot**.

The chat *brain* is already validated (see `chat-harness/`) and the chat *backend*
is already ported to Rust and compiles green on CI (Wave A). This plan covers the
remaining program: attribution (the hard, novel part) + the chat UI + wiring.

---

## 0. Current state (what already exists)

- ✅ **Meeting-detection nudge** — shipped, working (mic-in-use → overlay → start recording).
- ✅ **Chat backend (Wave A)** — `bog/src-tauri/src/chat/{retrieval,agent,commands}.rs` +
  migration `20260706000000_add_chat_messages.sql`; compiles + links GREEN on CI.
  Single-meeting (full transcript in context) + cross-app (agentic FTS-grep loop),
  reusing `summary::llm_client::generate_summary`. Pure-Rust ranked search (no FTS5
  triggers on the recording path). Qwen `<think>` scrubbing.
- ✅ **Validated prototype** — `chat-harness/` (Python) proved the retrieval brain on the
  real Qwen 4B GGUF: single-meeting, multi-hop, disambiguation, honesty all pass.
- ✅ **CI toolchain fixed** — build-macos pinned to `macos-15` + explicit **Xcode 16.4**
  select + rust-cache key bump (GitHub rolled to Xcode 26.5 which breaks cidre's
  `clang_rt.osx` link). Build is green and durable. **No local compile** (needs full
  Xcode which isn't installed) → **CI is the compiler**; verify every wave via a build.

## 1. Architecture — the attribution stack

"What did **John** say about X" decomposes into layers, each degrade-safe:

```
 mic stream  ── VAD+STT ─────────────► segments tagged  speaker = "me"        (YOU, unambiguous)
 system stream ─ VAD+STT ─► diarize ─► segments tagged  speaker = "them:S1|S2|…"  (each other voice)
                                          │
                                          ▼  identification (bind label → name)
                              people store + calendar roster + LLM cues + confirm
                                          │
                                          ▼
                              transcript segments carry a resolved display name
```

- **Separation (B2):** transcribe mic and system **independently** (not the current
  single mixed stream). Mic = you (unambiguous). System = everyone else.
- **Diarization:** cluster voices within the system stream → `them:S1, them:S2…`, with a
  voice embedding per cluster.
- **Identification:** bind each `them:Sn` → a real name, cheapest signal first:
  1. **1:1** — calendar shows you + one attendee → that attendee **is** the one "them". Done, no ask.
  2. **Voice-store match** — embedding matches a previously-labeled person → auto.
  3. **LLM binding** — extract self-intros ("Hi I'm John") + addressee cues ("Sarah, …"),
     constrained to the **calendar roster** (closed name set → no hallucination).
  4. **Confirm card** — only the residual unknowns: play a 3-sec clip + one-tap calendar
     names. Each confirmation **writes the voice→person store** → asked at most once per
     person, ever. This is the moat.

Everything degrades: no diarization → still me/them; no name → still `Speaker 2`;
wrong name → one-tap fix that teaches the store.

## 2. Data model (new migrations)

- `chat_messages` — ✅ already added (`20260706000000`).
- `people` — the durable identity store.
  `id, display_name, emails (json), voice_embeddings (json: list of f32 vectors), created_at, updated_at`.
- `transcripts.speaker` — already exists (empty today). Repurpose to a **stable label**:
  `"me"`, or `"them:S1"` etc. NEVER a name (names are late-bound + mutable).
- `meeting_participants` — the per-meeting binding (label → identity), editable + provenance.
  `id, meeting_id, speaker_label, person_id (nullable), display_name, source (calendar|intro|voice|user|onlyother), confidence, created_at`.
- `meeting_calendar` (optional) — cached roster per meeting joined from EventKit.
  `meeting_id, title, attendees (json: [{name,email}]), event_start, event_end, matched_at`.

All migrations: NOT NULL + server defaults where possible; additive; never a trigger on
the recording insert path.

## 3. Waves (dependency-ordered)

### Wave A — Chat backend ✅ DONE (green on CI)

### Wave B — Separation (B2: dual-stream transcription) 🔨 NEXT ⚠ core recording path
Transcribe mic and system **separately**, tag each segment `speaker="me"|"them"`.
- Pipeline (`audio/pipeline.rs`) currently mixes mic+system → one VAD/STT stream. Change:
  run VAD+STT on **each source** (mic path → "me", system path → "them"), OR (safer first
  cut) keep one mixed STT for text quality but derive per-segment me/them from a
  **source-dominance track** computed at mix time (both slices present in
  `extract_window`). ← decision below.
- **Decision (accuracy-first, per Jad):** go **true dual-stream**. Add a second VAD+STT
  path for the system source; mic path is cheap (usually just you). Feature-flag
  (`attribution_enabled`) with the current mixed path as fallback so recording never
  regresses. Thread `speaker` through `AudioChunk` → `worker::TranscriptUpdate` →
  `recording_saver::TranscriptSegment` → `save_transcript` INSERT → `transcripts.speaker`.
- Files: `audio/pipeline.rs`, `audio/transcription/worker.rs`, `audio/recording_state.rs`
  (`AudioChunk`), `audio/recording_commands.rs` (listener), `audio/recording_saver.rs`
  (`TranscriptSegment`), `database/repositories/transcript.rs` (INSERT), + the other
  `TranscriptSegment` construction sites (import/retranscription) for compile.
- ⚠ Delicate + un-testable locally → implement additively behind a flag, then **CI build →
  Jad records a 1-min two-person call → verify `speaker` populates me/them correctly**.
- Verification: `sqlite3 … "SELECT speaker,count(*) FROM transcripts GROUP BY speaker"`.

### Wave C — Diarization of the system stream ⏳ ⚠ build, not wire
Split "them" into distinct voices + capture a voice embedding per cluster.
- **CORRECTION (verified):** `audio/stt.rs` is **dead code** — it is NOT declared in
  `audio/mod.rs`, and `crate::pyannote` does not exist (no module, no crate). That's why
  the app builds despite `stt.rs` importing it. So there is nothing to "wire" — diarization
  must be **built**.
- **Foundation is present though:** `ort` (ONNX Runtime, `2.0.0-rc.10`) IS a dependency —
  Parakeet already runs ONNX via `ort::session::Session` (see `parakeet_engine/model.rs`).
  The dead `stt.rs` is a useful DESIGN TEMPLATE: it intended pyannote **ONNX** models —
  `get_or_download_model(Segmentation|Embedding)` → `EmbeddingExtractor` → `EmbeddingManager`
  clustering, `speaker_embedding: Vec<f32>` per segment.
- **Build plan:** a new `audio/diarize/` module using `ort`: download the pyannote
  segmentation ONNX + a speaker-embedding ONNX (wespeaker/pyannote) → per system-segment
  embedding → online clustering (cosine threshold) → `them:S1/S2…`. Reuse Parakeet's `ort`
  session pattern + model-download infra (`get_or_download_model` equivalent).
- Output: system segments sub-labeled + `speaker_embedding` per (meeting, label).
- Degrade-safe: diarization off/unavailable → all system = single "them" (Wave B already
  gives this). So Wave C is strictly additive on top of a working me/them.

### Wave D — Identification backend ⏳
- `people` + `meeting_participants` (+ optional `meeting_calendar`) migrations.
- **Calendar via EventKit** (local, no cloud): a Tauri/Swift bridge or an `objc2`/`cidre`
  EventKit read → find the event whose window contains the recording start → attendees.
  (Reuse the meeting-nudge's start signal for the time join.) Permission prompt handled once.
- **1:1 auto-bind** — you + one attendee → the one "them" = that attendee.
- **Voice-store match** — cosine match new cluster embedding vs `people.voice_embeddings`.
- **LLM binding** — a `chat`-style call: given the attributed transcript + roster, propose
  `label→name` from self-intros/addressee cues (constrained to roster; abstain if unsure).
- Produce a per-meeting **proposed mapping** + confidences; `resolve_participants` /
  `confirm_participant` commands; confirming writes back to `people` (learn the voice).

### Wave E — Chat UI + attribution UI ⏳
- Chat panel as a **3rd tab** in `meeting-details/page-content.tsx` (per-meeting) + a global
  **"Chat all meetings"** surface; message thread + input; subscribe to `chat-token`.
- **Confirm-speakers card** (post-meeting): residual unknown voices → play clip + one-tap
  calendar names; renames flow to `meeting_participants` + teach `people`.
- Show resolved names in transcript + chat citations.

### Wave F — Streaming + attribution-aware chat ⏳
- Real token streaming: add a streaming variant to the LLM path (Ollama/BuiltInAI SSE) →
  emit incremental `chat-token` deltas (today it emits the whole answer once).
- Chat retrieval uses resolved names so "what did John say / what did they tell me about
  Company X" work; citations show names + timestamps.

### Wave G — Build + verify ⏳
Combined dmg via CI → Jad records a real multi-person call → verify attribution + chat end
to end. Iterate.

## 4. Cross-cutting rules
- **CI is the compiler.** After each wave: commit → push → build-macos → fix red → repeat.
  Keep waves attributable (don't stack two un-verified changes into one build).
- **Recording is sacred.** Every attribution/pipeline change is additive + feature-flagged
  with the current mixed path as fallback. A bug must degrade to "empty speaker", never a
  broken recorder.
- **Names are late-bound + mutable.** Transcript stores stable labels; names live in
  `meeting_participants`, editable, learned into `people`.
- **No PHI/cloud.** All local. Voice embeddings + names stay on-device.
- **Reuse.** `generate_summary` for LLM; existing pyannote code for diarization; the
  meeting-nudge start signal for the calendar time-join; the proven harness logic for chat.

## 5. Open items to confirm while building
- ✅ pyannote buildability — RESOLVED: absent/dead; build diarization on `ort` (present) with
  pyannote ONNX models, using `stt.rs` as the design template. (Wave C, above.)
- EventKit access from Rust (objc2/cidre) vs a small Swift helper — spike at Wave D start.
- Dual-stream STT cost on Qwen/whisper — measure; if too heavy, fall back to
  mixed-STT-for-text + dominance-track-for-me/them (documented B1 fallback).
