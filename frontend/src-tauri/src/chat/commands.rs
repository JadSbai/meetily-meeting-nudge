//! Tauri commands for "chat with your meetings" — send / history / clear.
//!
//! Backend-only + additive. Provider/model arrive from the frontend the same
//! way summaries do (see `api_process_transcript`). Both the user message and
//! the assistant reply are persisted to `chat_messages`.

use crate::chat::agent::{agentic_answer, single_meeting_answer, LlmRunner};
use crate::chat::retrieval::{load_corpus, Citation, Corpus};
use crate::database::manager::DatabaseManager;
use serde::{Deserialize, Serialize};
use tauri::{AppHandle, Emitter, Manager};
use uuid::Uuid;

/// The final answer returned to the frontend.
#[derive(Debug, Clone, Serialize)]
pub struct ChatResponse {
    pub answer: String,
    pub citations: Vec<Citation>,
}

/// A persisted chat message row.
#[derive(Debug, Clone, Serialize, Deserialize, sqlx::FromRow)]
pub struct ChatMessage {
    pub id: String,
    pub scope: String,
    pub role: String,
    pub content: String,
    pub citations: Option<String>, // JSON array string, nullable
    pub created_at: String,
}

/// Streaming progress payload (`chat-token` event).
#[derive(Debug, Clone, Serialize)]
struct TokenPayload {
    scope: String,
    delta: String,
}

fn now_stamp() -> String {
    // Space-separated, subsecond precision → lexically sortable + consistent
    // with the table's `datetime('now')` DEFAULT format.
    chrono::Utc::now().format("%Y-%m-%d %H:%M:%S%.6f").to_string()
}

/// Chat with one meeting (`scope` = meeting id) or across all (`scope` = "all").
#[tauri::command]
pub async fn chat_send(
    app: AppHandle,
    scope: String,
    message: String,
    provider: String,
    model: String,
) -> Result<ChatResponse, String> {
    let db = DatabaseManager::new_from_app_handle(&app)
        .await
        .map_err(|e| e.to_string())?;
    let pool = db.pool();

    let corpus = Corpus::new(load_corpus(pool).await?);
    let app_data_dir = app.path().app_data_dir().ok();
    let llm = LlmRunner::resolve(pool, &provider, &model, app_data_dir).await?;

    let result = if scope == "all" {
        agentic_answer(&llm, &corpus, &message).await?
    } else {
        let meeting = corpus
            .get(&scope)
            .ok_or_else(|| format!("no meeting matching '{}'", scope))?;
        single_meeting_answer(&llm, meeting, &message).await?
    };

    // Stream progress. TODO: emit true token-by-token deltas once
    // generate_summary exposes a streaming API; for now emit the whole answer.
    let _ = app.emit(
        "chat-token",
        TokenPayload { scope: scope.clone(), delta: result.answer.clone() },
    );

    // Persist the user message, then the assistant reply.
    persist_message(pool, &scope, "user", &message, None).await?;
    let citations_json = serde_json::to_string(&result.citations).ok();
    persist_message(pool, &scope, "assistant", &result.answer, citations_json).await?;

    Ok(ChatResponse { answer: result.answer, citations: result.citations })
}

/// Read the conversation for a scope, oldest first.
#[tauri::command]
pub async fn chat_history(app: AppHandle, scope: String) -> Result<Vec<ChatMessage>, String> {
    let db = DatabaseManager::new_from_app_handle(&app)
        .await
        .map_err(|e| e.to_string())?;
    let pool = db.pool();

    sqlx::query_as::<_, ChatMessage>(
        "SELECT id, scope, role, content, citations, created_at \
         FROM chat_messages WHERE scope = ? ORDER BY created_at ASC",
    )
    .bind(scope.as_str())
    .fetch_all(pool)
    .await
    .map_err(|e| e.to_string())
}

/// Clear the conversation for a scope.
#[tauri::command]
pub async fn chat_clear(app: AppHandle, scope: String) -> Result<(), String> {
    let db = DatabaseManager::new_from_app_handle(&app)
        .await
        .map_err(|e| e.to_string())?;
    let pool = db.pool();

    sqlx::query("DELETE FROM chat_messages WHERE scope = ?")
        .bind(scope.as_str())
        .execute(pool)
        .await
        .map_err(|e| e.to_string())?;
    Ok(())
}

async fn persist_message(
    pool: &sqlx::SqlitePool,
    scope: &str,
    role: &str,
    content: &str,
    citations: Option<String>,
) -> Result<(), String> {
    sqlx::query(
        "INSERT INTO chat_messages (id, scope, role, content, citations, created_at) \
         VALUES (?, ?, ?, ?, ?, ?)",
    )
    .bind(Uuid::new_v4().to_string())
    .bind(scope)
    .bind(role)
    .bind(content)
    .bind(citations)
    .bind(now_stamp())
    .execute(pool)
    .await
    .map_err(|e| e.to_string())?;
    Ok(())
}
