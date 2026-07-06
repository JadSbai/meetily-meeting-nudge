-- Wave C: speaker diarization — durable per-segment voice embeddings.
--
-- Written ONLY by the post-recording `diarize::finalize_meeting` pass (never on the
-- recording-critical transcripts insert path). Each row is one "them"/"me" speech
-- segment's voice vector, joined back to `transcripts` by (meeting_id, audio_start_time).
-- This table is the durable voice store the identification stage (Wave D) matches
-- against to bind a `them:Sn` label to a real person across meetings.
--
-- `embedding` is a JSON array of f32 (the wespeaker CAM++ voice vector). `speaker_label`
-- is the RESOLVED label after clustering ('them:S1', 'them:S2', … or 'me'). All columns
-- NOT NULL with defaults so a pre-existing row can never NULL-500 a reader.

CREATE TABLE IF NOT EXISTS speaker_embeddings (
    id               TEXT PRIMARY KEY,
    meeting_id       TEXT NOT NULL,
    audio_start_time REAL NOT NULL,            -- join key back to transcripts.audio_start_time
    speaker_label    TEXT NOT NULL DEFAULT '', -- resolved: 'them:S1' | 'me' | 'them'
    embedding        TEXT NOT NULL DEFAULT '[]',-- JSON array of f32 (voice vector)
    rms_energy       REAL NOT NULL DEFAULT 0,   -- per-segment loudness (echo/silence arbitration)
    created_at       TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_speaker_embeddings_meeting
    ON speaker_embeddings (meeting_id);
