//! The answer engine — single-meeting path + cross-app agentic loop + prompts.
//!
//! Ported from the harness `single_meeting_answer` / `agentic_answer`. The one
//! structural difference: `generate_summary` takes a single (system, user)
//! pair rather than a running message list, so the agentic loop folds the
//! search/read history into an accumulating scratchpad passed as the user turn.

use crate::chat::retrieval::{Citation, Corpus, Meeting, SearchHit};
use crate::database::repositories::setting::SettingsRepository;
use crate::summary::llm_client::{generate_summary, LLMProvider};
use once_cell::sync::Lazy;
use regex::Regex;
use reqwest::Client;
use sqlx::SqlitePool;
use std::path::PathBuf;

const MAX_STEPS: usize = 6;

/// A finished answer plus the citations it was grounded in.
#[derive(Debug, Clone)]
pub struct AgentAnswer {
    pub answer: String,
    pub citations: Vec<Citation>,
}

/// Resolved LLM configuration + client, built the same way summaries are.
pub struct LlmRunner {
    client: Client,
    provider: LLMProvider,
    model_name: String,
    api_key: String,
    ollama_endpoint: Option<String>,
    custom_openai_endpoint: Option<String>,
    top_p: Option<f32>,
    app_data_dir: Option<PathBuf>,
}

impl LlmRunner {
    /// Mirror `process_transcript_background`: parse provider, resolve api key /
    /// ollama endpoint / custom-openai config from settings.
    pub async fn resolve(
        pool: &SqlitePool,
        provider_str: &str,
        model: &str,
        app_data_dir: Option<PathBuf>,
    ) -> Result<Self, String> {
        let provider = LLMProvider::from_str(provider_str)?;

        let keyless = matches!(
            provider,
            LLMProvider::Ollama | LLMProvider::BuiltInAI | LLMProvider::CustomOpenAI
        );
        let api_key = if keyless {
            String::new()
        } else {
            match SettingsRepository::get_api_key(pool, provider_str).await {
                Ok(Some(k)) if !k.is_empty() => k,
                Ok(_) => return Err(format!("API key not found for {}", provider_str)),
                Err(e) => return Err(format!("Failed to retrieve API key for {}: {}", provider_str, e)),
            }
        };

        let ollama_endpoint = if provider == LLMProvider::Ollama {
            SettingsRepository::get_model_config(pool)
                .await
                .ok()
                .flatten()
                .and_then(|c| c.ollama_endpoint)
        } else {
            None
        };

        let (custom_openai_endpoint, custom_key, top_p) = if provider == LLMProvider::CustomOpenAI {
            match SettingsRepository::get_custom_openai_config(pool).await {
                Ok(Some(c)) => (Some(c.endpoint), c.api_key, c.top_p),
                Ok(None) => return Err("Custom OpenAI provider selected but no configuration found".to_string()),
                Err(e) => return Err(format!("Failed to retrieve custom OpenAI config: {}", e)),
            }
        } else {
            (None, None, None)
        };

        let final_api_key = if provider == LLMProvider::CustomOpenAI {
            custom_key.unwrap_or_default()
        } else {
            api_key
        };

        Ok(Self {
            client: Client::new(),
            provider,
            model_name: model.to_string(),
            api_key: final_api_key,
            ollama_endpoint,
            custom_openai_endpoint,
            top_p,
            app_data_dir,
        })
    }

    /// One (system, user) completion, defensively cleaned of Qwen3 `<think>`.
    async fn chat(&self, system: &str, user: &str, max_tokens: u32, temperature: f32) -> Result<String, String> {
        let raw = generate_summary(
            &self.client,
            &self.provider,
            &self.model_name,
            &self.api_key,
            system,
            user,
            self.ollama_endpoint.as_deref(),
            self.custom_openai_endpoint.as_deref(),
            Some(max_tokens),
            Some(temperature),
            self.top_p,
            self.app_data_dir.as_ref(),
            None,
        )
        .await?;
        Ok(clean_response(&raw))
    }
}

/// Strip any `<think>…</think>` block (DOTALL). Qwen3 is a reasoning model and
/// `generate_summary` exposes no `enable_thinking` switch, so we scrub here.
pub fn clean_response(raw: &str) -> String {
    static THINK_RE: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?s)<think>.*?</think>").expect("valid think regex"));
    THINK_RE.replace_all(raw, "").trim().to_string()
}

const NO_THINK: &str =
    "Answer directly and concisely. Do NOT show your reasoning or thinking. Output only the final answer.";

const SYSTEM_AGENT: &str = "You answer questions about the user's recorded meetings. \
You cannot see the meetings directly — you must retrieve with tools, one action per turn.\n\n\
Respond with ONLY ONE compact JSON object per turn and nothing else. Do NOT show your reasoning. Shapes:\n\
- {\"thought\":\"...\",\"action\":\"search\",\"query\":\"keywords\"} — full-text search across ALL meetings; returns ranked snippets with meeting title + timestamp.\n\
- {\"thought\":\"...\",\"action\":\"read\",\"meeting_id\":\"<id>\",\"mode\":\"summary\"|\"full\"} — read a meeting's summary (cheap) or full transcript (for detail/quotes).\n\
- {\"thought\":\"...\",\"action\":\"final\",\"answer\":\"...\"} — give the final answer. Ground EVERY claim in what you retrieved and cite as [Meeting Title @ mm:ss]. If nothing relevant was found, say so honestly.\n\n\
Rules: prefer search first to locate evidence, then read the most relevant meeting(s); do not invent facts; be concise; stop and answer as soon as you have enough evidence.\n\n\
Meeting index (id — title — date — topic):\n{index}";

/// Single-meeting answer: stuff the whole transcript + summary into context.
pub async fn single_meeting_answer(
    llm: &LlmRunner,
    m: &Meeting,
    question: &str,
) -> Result<AgentAnswer, String> {
    let mut ctx = format!("# {} ({})\n", m.title, m.date);
    if !m.summary.is_empty() {
        ctx.push_str(&format!("\n## Summary\n{}\n", m.summary));
    }
    ctx.push_str(&format!("\n## Transcript\n{}", m.full_text()));
    let ctx: String = ctx.chars().take(28000).collect();

    let system = format!(
        "You answer questions about ONE meeting, using only the transcript/summary below. \
         {} Cite timestamps like [@ mm:ss] when quoting. If the answer isn't in it, say so.\n\n{}",
        NO_THINK, ctx
    );
    let answer = llm.chat(&system, question, 1536, 0.2).await?;
    Ok(AgentAnswer {
        answer,
        citations: vec![Citation { meeting_id: m.id.clone(), title: m.title.clone(), ts: String::new() }],
    })
}

struct Action {
    action: String,
    query: String,
    meeting_id: String,
    mode: String,
    answer: String,
}

/// Robustly parse the model's action: find the first `{ … }`, parse it; on any
/// failure treat the whole raw text as a free-form `final` answer.
fn parse_action(raw: &str) -> Action {
    let cleaned = raw.trim();
    if let (Some(a), Some(b)) = (cleaned.find('{'), cleaned.rfind('}')) {
        if b > a {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&cleaned[a..=b]) {
                let get = |k: &str| v.get(k).and_then(|x| x.as_str()).unwrap_or("").trim().to_string();
                let mut action = get("action");
                if action.is_empty() {
                    action = "final".to_string();
                }
                let mut mode = get("mode");
                if mode.is_empty() {
                    mode = "summary".to_string();
                }
                return Action { action, query: get("query"), meeting_id: get("meeting_id"), mode, answer: get("answer") };
            }
        }
    }
    Action { action: "final".to_string(), query: String::new(), meeting_id: String::new(), mode: "summary".to_string(), answer: cleaned.to_string() }
}

fn push_citation(citations: &mut Vec<Citation>, c: Citation) {
    if citations.len() >= 8 {
        return;
    }
    if !citations.iter().any(|x| x.meeting_id == c.meeting_id && x.ts == c.ts) {
        citations.push(c);
    }
}

/// Cross-app agentic loop: search → read → final, grounded + cited.
pub async fn agentic_answer(
    llm: &LlmRunner,
    corpus: &Corpus,
    question: &str,
) -> Result<AgentAnswer, String> {
    let index = corpus
        .list_meetings()
        .iter()
        .take(40)
        .map(|m| format!("- {} — {} — {} — {}", m.id, m.title, m.date, m.topic))
        .collect::<Vec<_>>()
        .join("\n");
    let system = SYSTEM_AGENT.replace("{index}", &index);

    let mut scratch = String::new();
    let mut citations: Vec<Citation> = Vec::new();

    for _ in 0..MAX_STEPS {
        let user = format!("QUESTION: {}\n\n{}\nRespond with ONE JSON action now.", question, scratch);
        let raw = llm.chat(&system, &user, 640, 0.1).await?;
        let act = parse_action(&raw);

        match act.action.as_str() {
            "search" => {
                let q = if act.query.is_empty() { question.to_string() } else { act.query.clone() };
                let hits = corpus.search(&q, 8);
                for h in &hits {
                    push_citation(&mut citations, hit_citation(h));
                }
                let obs = if hits.is_empty() {
                    "no matches".to_string()
                } else {
                    serde_json::to_string(&hits).unwrap_or_else(|_| "no matches".to_string())
                };
                scratch.push_str(&format!("ACTION: search \"{}\"\nRESULTS: {}\n\n", q, obs));
            }
            "read" => {
                let text = corpus.read_meeting(&act.meeting_id, &act.mode);
                let text: String = text.chars().take(8000).collect();
                if let Some(m) = corpus.get(&act.meeting_id) {
                    push_citation(&mut citations, Citation { meeting_id: m.id.clone(), title: m.title.clone(), ts: String::new() });
                }
                scratch.push_str(&format!("ACTION: read {} ({})\nCONTENT: {}\n\n", act.meeting_id, act.mode, text));
            }
            "final" => {
                let ans = act.answer.trim().to_string();
                if !ans.is_empty() {
                    return Ok(AgentAnswer { answer: ans, citations });
                }
                // Empty final (a small-model quirk) → synthesize free-form prose
                // from everything gathered, with the whole budget for the answer.
                let answer = self_synthesize(llm, &system, question, &scratch).await?;
                return Ok(AgentAnswer { answer, citations });
            }
            _ => {
                scratch.push_str("NOTE: unknown action; use search, read, or final.\n\n");
            }
        }
    }

    // Budget exhausted — force a best-effort final answer.
    let answer = self_synthesize(llm, &system, question, &scratch).await?;
    Ok(AgentAnswer { answer, citations })
}

async fn self_synthesize(
    llm: &LlmRunner,
    system: &str,
    question: &str,
    scratch: &str,
) -> Result<String, String> {
    let user = format!(
        "QUESTION: {}\n\n{}\nWrite the complete final answer now, grounded in the retrieved content \
         above, with [Meeting Title @ mm:ss] citations. {}",
        question, scratch, NO_THINK
    );
    llm.chat(system, &user, 1536, 0.2).await
}

fn hit_citation(h: &SearchHit) -> Citation {
    Citation { meeting_id: h.meeting_id.clone(), title: h.meeting_title.clone(), ts: h.ts.clone() }
}
