// audio/diarize/cluster.rs
//
// Post-recording clustering of "them" voice embeddings into distinct speakers,
// with energy arbitration + an echo-of-me filter so a mislabelled echo turn (the mic
// picking up the other side of a speakerphone call) never spawns a phantom speaker.
//
// Pure function over the recorded buffer → a map from a segment's audio_start_time to
// its RESOLVED label ('them:S1', 'them:S2', … or 'me' when a "them" turn was actually
// an echo of the user). Only segments whose label CHANGES appear in the returned map;
// if there is a single "them" voice we return nothing and Wave B's flat "them" stands.

use pyannote_rs::EmbeddingManager;
use std::collections::HashMap;

/// One buffered speech segment with its voice vector + loudness.
#[derive(Debug, Clone)]
pub struct SegEmbed {
    pub audio_start_time: f64,
    pub speaker: String, // original Wave-B label: "me" | "them"
    pub embedding: Vec<f32>,
    pub energy: f32, // RMS loudness of the segment
}

/// Cosine below which two "them" turns are treated as different speakers.
const SPEAKER_MATCH_THRESHOLD: f32 = 0.5;
/// A "them" turn this similar to the user's own voice centroid is an echo of *me*.
const ECHO_TO_ME_THRESHOLD: f32 = 0.72;
/// Never split into more than this many distinct "them" voices.
const MAX_SPEAKERS: usize = 8;
/// Drop "them" turns quieter than this fraction of the meeting's median "them" energy
/// from clustering — they are silence bleed / faint echo, not a real turn.
const ENERGY_FLOOR_FRAC: f32 = 0.15;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

fn centroid(vectors: &[&Vec<f32>]) -> Option<Vec<f32>> {
    let first = vectors.first()?;
    let dim = first.len();
    if dim == 0 {
        return None;
    }
    let mut acc = vec![0.0f32; dim];
    let mut n = 0.0f32;
    for v in vectors {
        if v.len() == dim {
            for (a, x) in acc.iter_mut().zip(v.iter()) {
                *a += x;
            }
            n += 1.0;
        }
    }
    if n == 0.0 {
        return None;
    }
    for a in acc.iter_mut() {
        *a /= n;
    }
    Some(acc)
}

fn median(mut xs: Vec<f32>) -> f32 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    xs[xs.len() / 2]
}

/// Assign resolved speaker labels. Returns start_time → new_label for segments whose
/// label changed. Empty when there is nothing to refine (0/1 "them" voice, no model).
pub fn assign_labels(segments: &[SegEmbed]) -> HashMap<u64, String> {
    let mut out: HashMap<u64, String> = HashMap::new();

    // Only embedded segments participate. audio_start_time keyed by bit pattern for
    // exact-ish map identity (we still write back via nearest-match on the DB side).
    let embedded: Vec<&SegEmbed> = segments
        .iter()
        .filter(|s| !s.embedding.is_empty())
        .collect();
    if embedded.is_empty() {
        return out;
    }

    // Reference for the user's own voice, from the unambiguous mic ("me") stream.
    let me_vecs: Vec<&Vec<f32>> = embedded
        .iter()
        .filter(|s| s.speaker == "me")
        .map(|s| &s.embedding)
        .collect();
    let me_centroid = centroid(&me_vecs);

    // "them" turns, loud enough to be real, ordered in time.
    let them: Vec<&SegEmbed> = embedded
        .iter()
        .copied()
        .filter(|s| s.speaker == "them")
        .collect();
    if them.is_empty() {
        return out;
    }
    let energy_floor = median(them.iter().map(|s| s.energy).collect()) * ENERGY_FLOOR_FRAC;

    // Time order (buffer is roughly ordered already, but be explicit).
    let mut them_sorted = them.clone();
    them_sorted.sort_by(|a, b| {
        a.audio_start_time
            .partial_cmp(&b.audio_start_time)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut manager = EmbeddingManager::new(MAX_SPEAKERS);
    let mut tagged: Vec<(f64, Tag)> = Vec::new();

    for s in &them_sorted {
        // Echo-of-me: a "them" turn that matches the user's own voice and is not loud
        // is the mic bleeding into the system tap — reattribute to "me".
        if let Some(mc) = &me_centroid {
            if cosine(&s.embedding, mc) >= ECHO_TO_ME_THRESHOLD
                && s.energy <= energy_floor.max(f32::MIN_POSITIVE) * 3.0
            {
                tagged.push((s.audio_start_time, Tag::EchoMe));
                continue;
            }
        }
        // Silence/faint bleed → leave as flat "them" (don't pollute a cluster).
        if s.energy < energy_floor {
            tagged.push((s.audio_start_time, Tag::Flat));
            continue;
        }
        // search_speaker returns None only when MAX_SPEAKERS is exceeded → keep flat.
        match manager.search_speaker(s.embedding.clone(), SPEAKER_MATCH_THRESHOLD) {
            Some(cid) => tagged.push((s.audio_start_time, Tag::Speaker(cid))),
            None => tagged.push((s.audio_start_time, Tag::Flat)),
        }
    }

    // Map raw cluster id → them:Sn in first-appearance order.
    let mut order: HashMap<usize, usize> = HashMap::new();
    let mut next = 1usize;
    for (_, t) in &tagged {
        if let Tag::Speaker(cid) = t {
            order.entry(*cid).or_insert_with(|| {
                let n = next;
                next += 1;
                n
            });
        }
    }

    let multi_speaker = order.len() >= 2;

    for (start, t) in tagged {
        let new_label = match t {
            Tag::EchoMe => Some("me".to_string()), // reattributed echo of the user
            Tag::Speaker(cid) if multi_speaker => Some(format!("them:S{}", order[&cid])),
            _ => None, // single voice or kept-flat → leave as "them"
        };
        if let Some(label) = new_label {
            out.insert(start.to_bits(), label);
        }
    }

    out
}

/// Per-segment clustering outcome.
enum Tag {
    /// A distinct "them" voice cluster (raw id from the embedding manager).
    Speaker(usize),
    /// A "them" turn that is actually an echo of the user's own voice.
    EchoMe,
    /// Keep the flat "them" label (silence bleed or over the speaker cap).
    Flat,
}
