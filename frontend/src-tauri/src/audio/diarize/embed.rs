// audio/diarize/embed.rs
//
// Voice-embedding extraction for diarization (Wave C).
//
// One wespeaker CAM++ ONNX model (shared `ort` build) turns a speech segment's
// samples into a fixed-dim voice vector. We run it on each "them"/"me" VAD segment
// the transcription worker already produced — the pipeline does the segmentation, so
// we only need the EMBEDDING model, not pyannote's segmentation model.
//
// Everything here is best-effort and degrade-safe: if the model isn't downloaded
// yet, or inference fails, `compute_embedding` returns None and diarization silently
// falls back to a single "them" (Wave B behaviour). It never panics on the hot path.

use log::{info, warn};
use once_cell::sync::Lazy;
use pyannote_rs::EmbeddingExtractor;
use std::path::PathBuf;
use std::sync::Mutex;

/// wespeaker CAM++ voice-embedding model (≈29 MB), same release pyannote-rs ships.
const MODEL_URL: &str = "https://github.com/thewh1teagle/pyannote-rs/releases/download/v0.1.0/wespeaker_en_voxceleb_CAM%2B%2B.onnx";
const MODEL_FILENAME: &str = "wespeaker_campp.onnx";
/// Embedding model expects 16 kHz mono (same as the STT front-end).
const TARGET_SAMPLE_RATE: u32 = 16000;

/// Absolute path to the diarization models dir, set once at startup from app_data_dir.
static MODEL_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

/// Lazily-loaded extractor. `None` until the model file exists on disk and loads.
/// `compute()` needs `&mut self`, so the whole thing lives behind a Mutex; the
/// transcription worker is serial (NUM_WORKERS = 1) so there is no real contention.
static EXTRACTOR: Lazy<Mutex<Option<EmbeddingExtractor>>> = Lazy::new(|| Mutex::new(None));

/// Record the models directory (`<app_data>/models/diarize`). Called from `setup()`.
pub fn set_model_dir(dir: PathBuf) {
    if !dir.exists() {
        if let Err(e) = std::fs::create_dir_all(&dir) {
            warn!("diarize: could not create model dir {}: {}", dir.display(), e);
        }
    }
    *MODEL_DIR.lock().unwrap() = Some(dir);
}

fn model_path() -> Option<PathBuf> {
    MODEL_DIR
        .lock()
        .unwrap()
        .as_ref()
        .map(|d| d.join(MODEL_FILENAME))
}

/// Is the embedding model present and usable? (Cheap file check.)
pub fn model_available() -> bool {
    model_path().map(|p| p.exists()).unwrap_or(false)
}

/// Best-effort background download of the embedding model if it isn't on disk.
/// Called once at startup; a failure just means the first recording gets a single
/// "them" and diarization kicks in on the next one.
pub async fn ensure_model_downloaded() {
    let path = match model_path() {
        Some(p) => p,
        None => {
            warn!("diarize: model dir not set; skipping model download");
            return;
        }
    };
    if path.exists() {
        info!("diarize: embedding model present at {}", path.display());
        return;
    }

    info!("diarize: downloading voice-embedding model (~29 MB) → {}", path.display());
    let tmp = path.with_extension("onnx.part");
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(600))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            warn!("diarize: http client build failed: {}", e);
            return;
        }
    };
    let bytes = match client.get(MODEL_URL).send().await {
        Ok(resp) if resp.status().is_success() => match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                warn!("diarize: model body read failed: {}", e);
                return;
            }
        },
        Ok(resp) => {
            warn!("diarize: model download HTTP {}", resp.status());
            return;
        }
        Err(e) => {
            warn!("diarize: model download failed: {}", e);
            return;
        }
    };
    if let Err(e) = std::fs::write(&tmp, &bytes) {
        warn!("diarize: writing model temp file failed: {}", e);
        return;
    }
    if let Err(e) = std::fs::rename(&tmp, &path) {
        warn!("diarize: finalizing model file failed: {}", e);
        return;
    }
    info!("diarize: embedding model ready ({} bytes)", bytes.len());
}

/// Compute a voice embedding for one speech segment.
///
/// `samples` are f32 in [-1, 1] at `sample_rate`. Returns None if the model is
/// unavailable, the segment is too short, or inference fails (always degrade-safe).
pub fn compute_embedding(samples: &[f32], sample_rate: u32) -> Option<Vec<f32>> {
    if !model_available() {
        return None;
    }
    // wespeaker needs enough signal to be meaningful — skip sub-0.3s blips.
    if (samples.len() as f32 / sample_rate.max(1) as f32) < 0.3 {
        return None;
    }

    // Resample to 16 kHz mono, then to the i16 PCM the extractor expects.
    let resampled = if sample_rate != TARGET_SAMPLE_RATE {
        crate::audio::audio_processing::resample_audio(samples, sample_rate, TARGET_SAMPLE_RATE)
    } else {
        samples.to_vec()
    };
    let pcm: Vec<i16> = resampled
        .iter()
        .map(|&x| (x.clamp(-1.0, 1.0) * 32767.0) as i16)
        .collect();

    let mut guard = EXTRACTOR.lock().ok()?;
    if guard.is_none() {
        let path = model_path()?;
        match EmbeddingExtractor::new(&path) {
            Ok(ex) => {
                info!("diarize: embedding model loaded");
                *guard = Some(ex);
            }
            Err(e) => {
                warn!("diarize: failed to load embedding model: {}", e);
                return None;
            }
        }
    }
    let extractor = guard.as_mut()?;
    match extractor.compute(&pcm) {
        Ok(iter) => {
            let v: Vec<f32> = iter.collect();
            if v.is_empty() || v.iter().any(|x| !x.is_finite()) {
                None
            } else {
                Some(v)
            }
        }
        Err(e) => {
            warn!("diarize: embedding compute failed: {}", e);
            None
        }
    }
}
