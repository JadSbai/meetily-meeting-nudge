// audio/diarize/ — speaker diarization of the "them" (system) stream (Wave C).
//
// Strictly additive on top of Wave B's me/them attribution and touches NOTHING on the
// recording-critical path:
//   • during recording the transcription worker calls `record_segment` with each VAD
//     segment's samples → we buffer a voice embedding + loudness in memory (no DB, no
//     frontend round-trip — 200-float vectors never leave the backend);
//   • after `save_transcript` persists a meeting, `finalize_meeting` clusters that
//     buffer into distinct voices, writes the durable voice vectors to
//     `speaker_embeddings`, and refines the transcript labels ("them" → "them:S1"…),
//     matching rows back by audio_start_time.
//
// Degrade-safe end to end: no model / no embeddings / a single voice → the flat "them"
// from Wave B stands and nothing breaks.

pub mod cluster;
pub mod embed;

use cluster::SegEmbed;
use log::{info, warn};
use once_cell::sync::Lazy;
use sqlx::SqlitePool;
use std::collections::HashSet;
use std::sync::Mutex;

/// Recording-scoped buffer of per-segment embeddings. Cleared at recording start.
static BUFFER: Lazy<Mutex<Vec<SegEmbed>>> = Lazy::new(|| Mutex::new(Vec::new()));

/// Clear the buffer for a new recording session. Called from the record-start path.
pub fn reset_buffer() {
    if let Ok(mut b) = BUFFER.lock() {
        b.clear();
    }
}

fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    (samples.iter().map(|&x| x * x).sum::<f32>() / samples.len() as f32).sqrt()
}

/// Buffer one speech segment's voice embedding + loudness during recording.
///
/// `speaker` is the Wave-B label ("me" | "them"); anything else is ignored. Cheap
/// and best-effort: if the embedding model isn't ready the segment is still buffered
/// with an empty vector (it just won't participate in clustering).
pub fn record_segment(audio_start_time: f64, speaker: &str, samples: &[f32], sample_rate: u32) {
    if speaker != "me" && speaker != "them" {
        return;
    }
    let energy = rms(samples);
    let embedding = embed::compute_embedding(samples, sample_rate).unwrap_or_default();
    if let Ok(mut b) = BUFFER.lock() {
        b.push(SegEmbed {
            audio_start_time,
            speaker: speaker.to_string(),
            embedding,
            energy,
        });
    }
}

fn drain_buffer() -> Vec<SegEmbed> {
    match BUFFER.lock() {
        Ok(mut b) => std::mem::take(&mut *b),
        Err(_) => Vec::new(),
    }
}

/// Nearest DB row (by audio_start_time) to a buffered segment, preferring the same
/// original speaker and skipping already-claimed rows. Guards against float drift from
/// the transcript's JSON round-trip through the frontend.
fn nearest_row<'a>(
    rows: &'a [(String, f64, String)],
    start: f64,
    orig_speaker: &str,
    used: &HashSet<String>,
) -> Option<&'a (String, f64, String)> {
    rows.iter()
        .filter(|(id, t, sp)| {
            !used.contains(id) && sp == orig_speaker && (t - start).abs() < 0.15
        })
        .min_by(|a, b| {
            (a.1 - start)
                .abs()
                .partial_cmp(&(b.1 - start).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

/// Cluster the recorded buffer, persist voice vectors, and refine transcript labels
/// for `meeting_id`. Returns how many transcript rows were relabelled. Best-effort:
/// callers log and ignore errors so a diarization hiccup never fails a save.
pub async fn finalize_meeting(pool: &SqlitePool, meeting_id: &str) -> anyhow::Result<usize> {
    let segments = drain_buffer();
    if segments.is_empty() {
        return Ok(0);
    }

    let labels = cluster::assign_labels(&segments);

    // Snapshot the meeting's transcript rows for nearest-time matching.
    // `audio_start_time` is nullable on legacy rows; filter them so the f64 decode
    // (and our nearest-time match) only ever sees real timestamps.
    let rows: Vec<(String, f64, String)> = sqlx::query_as::<_, (String, f64, String)>(
        "SELECT id, audio_start_time, speaker FROM transcripts
         WHERE meeting_id = ? AND audio_start_time IS NOT NULL",
    )
    .bind(meeting_id)
    .fetch_all(pool)
    .await?;

    let mut relabeled = 0usize;
    let mut used: HashSet<String> = HashSet::new();
    let mut tx = pool.begin().await?;

    for seg in &segments {
        let resolved = labels.get(&seg.audio_start_time.to_bits()).cloned();

        // Refine the transcript label if clustering changed it.
        if let Some(new_label) = &resolved {
            if let Some((id, _, _)) =
                nearest_row(&rows, seg.audio_start_time, &seg.speaker, &used)
            {
                sqlx::query("UPDATE transcripts SET speaker = ? WHERE id = ?")
                    .bind(new_label)
                    .bind(id)
                    .execute(&mut *tx)
                    .await?;
                used.insert(id.clone());
                relabeled += 1;
            }
        }

        // Persist the voice vector (durable store for cross-meeting identification).
        if !seg.embedding.is_empty() {
            let final_label = resolved.unwrap_or_else(|| seg.speaker.clone());
            let emb_json = serde_json::to_string(&seg.embedding).unwrap_or_else(|_| "[]".into());
            let id = format!("spkemb-{}", uuid::Uuid::new_v4());
            sqlx::query(
                "INSERT INTO speaker_embeddings
                    (id, meeting_id, audio_start_time, speaker_label, embedding, rms_energy)
                 VALUES (?, ?, ?, ?, ?, ?)",
            )
            .bind(id)
            .bind(meeting_id)
            .bind(seg.audio_start_time)
            .bind(&final_label)
            .bind(emb_json)
            .bind(seg.energy as f64)
            .execute(&mut *tx)
            .await?;
        }
    }

    tx.commit().await?;
    info!(
        "diarize: finalized meeting {} — {} segments buffered, {} relabelled",
        meeting_id,
        segments.len(),
        relabeled
    );
    if relabeled == 0 && !labels.is_empty() {
        warn!("diarize: labels computed but no transcript rows matched by time — check audio_start_time drift");
    }
    Ok(relabeled)
}
