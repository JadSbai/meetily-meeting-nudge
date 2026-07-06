//! Chat with your meetings (Wave A — backend only).
//!
//! Ported faithfully from the validated Python prototype in
//! `chat-harness/chat_harness.py`. Two answer modes:
//!   * single-meeting — stuff one recording's transcript + summary into context
//!     and answer directly with timestamp citations.
//!   * cross-app — an agentic search/read/final loop over pure-Rust retrieval
//!     primitives, then a grounded, cited answer.
//!
//! Retrieval is in-memory keyword ranking built at query time from the
//! `transcripts` table (mirrors the harness; NO FTS5, NO embeddings). The
//! recording/audio path is never touched — this module is strictly additive.
//!
//! * [`retrieval`] — corpus loading + ranked keyword search + read primitives.
//! * [`agent`]     — the agentic loop, single-meeting path, prompts, LLM glue.
//! * [`commands`]  — Tauri commands (send / history / clear) + persistence.

pub mod agent;
pub mod commands;
pub mod retrieval;
