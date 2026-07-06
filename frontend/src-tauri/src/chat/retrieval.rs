//! Retrieval brain — corpus loading + ranked keyword search + read primitives.
//!
//! Pure Rust port of the harness `load_corpus` / `Corpus` (SQLite FTS5 replaced
//! by an in-memory BM25-ish ranker so we never depend on sqlx FTS5 availability
//! and never add a trigger to the recording-critical transcripts insert path).

use crate::database::models::MeetingModel;
use crate::database::repositories::meeting::MeetingsRepository;
use serde::{Deserialize, Serialize};
use sqlx::{Row, SqlitePool};

/// One transcript line for a meeting.
#[derive(Debug, Clone)]
pub struct Segment {
    pub text: String,
    pub ts: String,      // display timestamp (mm:ss or raw)
    pub speaker: String, // may be empty
    pub order: f64,      // sort key (audio_start_time secs, else index)
}

/// A meeting with its loaded segments + flattened summary.
#[derive(Debug, Clone)]
pub struct Meeting {
    pub id: String,
    pub title: String,
    pub date: String, // yyyy-mm-dd
    pub segments: Vec<Segment>,
    pub summary: String,
}

impl Meeting {
    /// Full transcript rendered as `[ts] speaker: text` lines.
    pub fn full_text(&self) -> String {
        self.segments
            .iter()
            .map(|s| {
                let sp = if s.speaker.is_empty() {
                    String::new()
                } else {
                    format!("{}: ", s.speaker)
                };
                format!("[{}] {}{}", s.ts, sp, s.text)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Short topic blurb for the meeting index (summary first, else transcript).
    pub fn blurb(&self, n: usize) -> String {
        let base = if self.summary.trim().is_empty() {
            self.full_text()
        } else {
            self.summary.clone()
        };
        let collapsed = base.split_whitespace().collect::<Vec<_>>().join(" ");
        let chars: Vec<char> = collapsed.chars().collect();
        if chars.len() > n {
            format!("{}…", chars.iter().take(n).collect::<String>())
        } else {
            chars.into_iter().collect()
        }
    }
}

/// A grounding citation the assistant relied on.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Citation {
    pub meeting_id: String,
    pub title: String,
    pub ts: String,
}

/// A ranked search snippet (serialized as the agent's observation).
#[derive(Debug, Clone, Serialize)]
pub struct SearchHit {
    pub meeting_id: String,
    pub meeting_title: String,
    pub ts: String,
    pub speaker: String,
    pub snippet: String,
    pub score: f64,
}

/// A meeting index row for the agent's cross-app prompt.
#[derive(Debug, Clone)]
pub struct MeetingInfo {
    pub id: String,
    pub title: String,
    pub date: String,
    pub topic: String,
}

const STOPWORDS: &[&str] = &[
    "the", "a", "an", "of", "to", "in", "on", "and", "or", "what", "did", "we", "was", "is",
    "about", "for", "how", "when", "who", "which", "that", "our", "do", "does", "with",
];

fn tokenize(s: &str) -> Vec<String> {
    let lower = s.to_lowercase();
    lower
        .split(|c: char| !(c.is_ascii_alphanumeric() || c == '\''))
        .filter(|t| !t.is_empty())
        .map(|t| t.to_string())
        .collect()
}

/// Salient, deduped query terms (stopwords + single chars dropped, with a
/// lenient fallback to all terms so recall never collapses to zero).
fn query_terms(q: &str) -> Vec<String> {
    let all = tokenize(q);
    let mut terms: Vec<String> = all
        .iter()
        .filter(|t| t.len() > 1 && !STOPWORDS.contains(&t.as_str()))
        .cloned()
        .collect();
    if terms.is_empty() {
        terms = all;
    }
    // distinct, preserve order
    let mut seen = std::collections::HashSet::new();
    terms.retain(|t| seen.insert(t.clone()));
    terms
}

fn fmt_ts(astart: Option<f64>, raw: Option<&str>) -> String {
    if let Some(a) = astart {
        let s = a as i64;
        return format!("{:02}:{:02}", s / 60, s % 60);
    }
    raw.unwrap_or("").to_string()
}

/// Flatten `summary_processes.result` JSON (sections/blocks) to plain text.
/// Mirrors `_flatten_summary` in the harness (title/name/heading first).
fn flatten_summary(result_json: &str) -> String {
    if result_json.trim().is_empty() {
        return String::new();
    }
    let value: serde_json::Value = match serde_json::from_str(result_json) {
        Ok(v) => v,
        Err(_) => return result_json.to_string(),
    };
    let mut out: Vec<String> = Vec::new();
    walk_summary(&value, &mut out);
    let mut seen = std::collections::HashSet::new();
    out.into_iter().filter(|l| seen.insert(l.clone())).collect::<Vec<_>>().join("\n")
}

fn walk_summary(x: &serde_json::Value, out: &mut Vec<String>) {
    match x {
        serde_json::Value::String(s) => {
            let t = s.trim();
            if !t.is_empty() {
                out.push(t.to_string());
            }
        }
        serde_json::Value::Array(a) => a.iter().for_each(|i| walk_summary(i, out)),
        serde_json::Value::Object(o) => {
            for k in ["title", "name", "heading"] {
                if let Some(serde_json::Value::String(s)) = o.get(k) {
                    let t = s.trim();
                    if !t.is_empty() {
                        out.push(t.to_string());
                    }
                }
            }
            for (k, v) in o {
                if matches!(k.as_str(), "title" | "name" | "heading") {
                    continue;
                }
                walk_summary(v, out);
            }
        }
        _ => {}
    }
}

/// Load every meeting with its segments + summary from the live SQLite pool.
pub async fn load_corpus(pool: &SqlitePool) -> Result<Vec<Meeting>, String> {
    let rows: Vec<MeetingModel> = MeetingsRepository::get_meetings(pool)
        .await
        .map_err(|e| format!("failed to load meetings: {}", e))?;

    let mut meetings = Vec::with_capacity(rows.len());
    for m in rows {
        let title = if m.title.trim().is_empty() {
            "(untitled)".to_string()
        } else {
            m.title.clone()
        };
        let date = m.created_at.0.format("%Y-%m-%d").to_string();
        let mut meeting = Meeting { id: m.id.clone(), title, date, segments: Vec::new(), summary: String::new() };

        let seg_rows = sqlx::query(
            "SELECT transcript, timestamp, audio_start_time, speaker \
             FROM transcripts WHERE meeting_id = ? ORDER BY audio_start_time",
        )
        .bind(meeting.id.as_str())
        .fetch_all(pool)
        .await
        .map_err(|e| format!("failed to load transcripts: {}", e))?;

        for (i, row) in seg_rows.iter().enumerate() {
            let text = row.try_get::<Option<String>, _>("transcript").ok().flatten().unwrap_or_default();
            let text = text.trim().to_string();
            if text.is_empty() {
                continue;
            }
            let astart = row.try_get::<Option<f64>, _>("audio_start_time").ok().flatten();
            let ts_raw = row.try_get::<Option<String>, _>("timestamp").ok().flatten();
            let speaker = row.try_get::<Option<String>, _>("speaker").ok().flatten().unwrap_or_default().trim().to_string();
            let order = astart.unwrap_or(i as f64);
            meeting.segments.push(Segment { text, ts: fmt_ts(astart, ts_raw.as_deref()), speaker, order });
        }

        // Fallback: the concatenated blob if per-segment rows are absent.
        if meeting.segments.is_empty() {
            if let Some(row) = sqlx::query(
                "SELECT transcript_text FROM transcript_chunks WHERE meeting_id = ? LIMIT 1",
            )
            .bind(meeting.id.as_str())
            .fetch_optional(pool)
            .await
            .map_err(|e| format!("failed to load transcript_chunks: {}", e))?
            {
                let blob = row.try_get::<Option<String>, _>("transcript_text").ok().flatten().unwrap_or_default();
                for (i, line) in blob.split('\n').enumerate() {
                    let line = line.trim();
                    if !line.is_empty() {
                        meeting.segments.push(Segment { text: line.to_string(), ts: String::new(), speaker: String::new(), order: i as f64 });
                    }
                }
            }
        }

        if let Some(row) = sqlx::query("SELECT result FROM summary_processes WHERE meeting_id = ? LIMIT 1")
            .bind(meeting.id.as_str())
            .fetch_optional(pool)
            .await
            .map_err(|e| format!("failed to load summary: {}", e))?
        {
            if let Some(result) = row.try_get::<Option<String>, _>("result").ok().flatten() {
                meeting.summary = flatten_summary(&result);
            }
        }

        // Keep segments in chronological order (query may sort NULLs first).
        meeting.segments.sort_by(|a, b| a.order.partial_cmp(&b.order).unwrap_or(std::cmp::Ordering::Equal));
        meetings.push(meeting);
    }
    Ok(meetings)
}

struct SegRef {
    m: usize,
    s: usize,
    tokens: Vec<String>,
}

/// The retrieval corpus — ranked keyword search + read-on-demand.
pub struct Corpus {
    meetings: Vec<Meeting>,
    seg_index: Vec<SegRef>,
    avg_len: f64,
}

impl Corpus {
    pub fn new(meetings: Vec<Meeting>) -> Self {
        let mut seg_index = Vec::new();
        let mut total = 0usize;
        for (mi, m) in meetings.iter().enumerate() {
            for (si, seg) in m.segments.iter().enumerate() {
                let tokens = tokenize(&seg.text);
                total += tokens.len();
                seg_index.push(SegRef { m: mi, s: si, tokens });
            }
        }
        let avg_len = if seg_index.is_empty() { 1.0 } else { total as f64 / seg_index.len() as f64 };
        Self { meetings, seg_index, avg_len }
    }

    /// Ranked snippets across ALL meetings (light BM25 + multi-term boost).
    pub fn search(&self, query: &str, k: usize) -> Vec<SearchHit> {
        let terms = query_terms(query);
        if terms.is_empty() {
            return Vec::new();
        }
        let avg = self.avg_len.max(1.0);
        let (k1, b) = (1.5_f64, 0.75_f64);
        let mut scored: Vec<(f64, usize)> = Vec::new();
        for (i, sr) in self.seg_index.iter().enumerate() {
            let doc_len = sr.tokens.len() as f64;
            let mut score = 0.0;
            let mut distinct = 0;
            for term in &terms {
                let tf = sr.tokens.iter().filter(|w| w.as_str() == term.as_str()).count() as f64;
                if tf > 0.0 {
                    distinct += 1;
                    let denom = tf + k1 * (1.0 - b + b * (doc_len / avg));
                    score += (tf * (k1 + 1.0)) / denom;
                }
            }
            if distinct > 1 {
                score += 0.5 * (distinct as f64 - 1.0);
            }
            if score > 0.0 {
                scored.push((score, i));
            }
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
            .into_iter()
            .map(|(score, i)| {
                let sr = &self.seg_index[i];
                let m = &self.meetings[sr.m];
                let seg = &m.segments[sr.s];
                SearchHit {
                    meeting_id: m.id.clone(),
                    meeting_title: m.title.clone(),
                    ts: seg.ts.clone(),
                    speaker: seg.speaker.clone(),
                    snippet: snippet(&seg.text, 320),
                    score: (score * 100.0).round() / 100.0,
                }
            })
            .collect()
    }

    pub fn list_meetings(&self) -> Vec<MeetingInfo> {
        self.meetings
            .iter()
            .map(|m| MeetingInfo { id: m.id.clone(), title: m.title.clone(), date: m.date.clone(), topic: m.blurb(160) })
            .collect()
    }

    pub fn get(&self, meeting_id: &str) -> Option<&Meeting> {
        self.resolve(meeting_id)
    }

    /// Tolerant lookup: exact id, else title-contains / id-prefix (the model
    /// occasionally passes a title or a partial id).
    fn resolve(&self, meeting_id: &str) -> Option<&Meeting> {
        if let Some(m) = self.meetings.iter().find(|m| m.id == meeting_id) {
            return Some(m);
        }
        let needle = meeting_id.to_lowercase();
        self.meetings
            .iter()
            .find(|m| m.title.to_lowercase().contains(&needle) || m.id.starts_with(meeting_id))
    }

    pub fn read_meeting(&self, meeting_id: &str, mode: &str) -> String {
        let m = match self.resolve(meeting_id) {
            Some(m) => m,
            None => return format!("(no meeting matching '{}')", meeting_id),
        };
        if mode == "full" {
            return format!("# {} ({})\n\n{}", m.title, m.date, m.full_text());
        }
        let body = if m.summary.is_empty() {
            snippet(&m.full_text(), 2000)
        } else {
            m.summary.clone()
        };
        format!("# {} ({}) — summary\n\n{}", m.title, m.date, body)
    }
}

fn snippet(text: &str, max: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() > max {
        format!("{}…", chars.iter().take(max).collect::<String>())
    } else {
        text.to_string()
    }
}
