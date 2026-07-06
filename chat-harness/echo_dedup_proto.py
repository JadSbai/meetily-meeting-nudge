#!/usr/bin/env python3
"""Prototype + validate echo-dedup on the REAL cofounder recording before porting to Rust.

Rule: two segments from DIFFERENT speakers that overlap in time AND have near-identical
text = an echo pair (mic picked up the phone/speaker). Keep the LONGER (cleaner)
transcription; drop the degraded echo. Genuine overlaps (different text) are untouched.
"""
import os, re, sqlite3
from difflib import SequenceMatcher

DB = os.path.expanduser("~/Library/Application Support/com.meetily.ai/meeting_minutes.sqlite")
MEETING = "meeting-07695b0a-b8bb-40f7-acae-fe60b79d685c"

# Tunables (validated below)
TIME_WINDOW = 2.0     # secs: echo lag is small
SIM_THRESHOLD = 0.55  # normalized text similarity to call it an echo


def norm(t: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()


def sim(a: str, b: str) -> float:
    # Word-set Jaccard — trivially portable to Rust (vs Python's difflib).
    wa, wb = set(norm(a).split()), set(norm(b).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def dedup(segs):
    """segs: list of dict(start, end, speaker, text). Returns (kept, dropped_pairs)."""
    drop = set()
    pairs = []
    for i in range(len(segs)):
        if i in drop:
            continue
        for j in range(i + 1, len(segs)):
            if j in drop:
                continue
            a, b = segs[i], segs[j]
            if b["start"] - a["start"] > TIME_WINDOW:
                break  # sorted by start; no further candidates
            if a["speaker"] == b["speaker"]:
                continue
            if sim(a["text"], b["text"]) >= SIM_THRESHOLD:
                # echo pair → drop the shorter (degraded echo)
                loser = j if len(a["text"]) >= len(b["text"]) else i
                keeper = i if loser == j else j
                drop.add(loser)
                pairs.append((segs[keeper], segs[loser]))
                if loser == i:
                    break
    kept = [s for k, s in enumerate(segs) if k not in drop]
    return kept, pairs


def main():
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    rows = con.execute(
        "SELECT audio_start_time, audio_end_time, speaker, transcript FROM transcripts "
        "WHERE meeting_id=? ORDER BY audio_start_time", (MEETING,)).fetchall()
    segs = [{"start": r[0] or 0.0, "end": r[1] or 0.0, "speaker": r[2] or "", "text": (r[3] or "").strip()}
            for r in rows if (r[3] or "").strip()]
    kept, pairs = dedup(segs)

    before = {"me": sum(s["speaker"] == "me" for s in segs), "them": sum(s["speaker"] == "them" for s in segs)}
    after = {"me": sum(s["speaker"] == "me" for s in kept), "them": sum(s["speaker"] == "them" for s in kept)}
    print(f"BEFORE: {len(segs)} segs  me={before['me']} them={before['them']}")
    print(f"AFTER : {len(kept)} segs  me={after['me']} them={after['them']}  (dropped {len(segs)-len(kept)} echoes)\n")
    print("=== echo pairs removed (KEPT ⟵ dropped) ===")
    for keep, lose in pairs:
        print(f"  {keep['start']:6.2f} {keep['speaker']:4} «{keep['text'][:44]}»")
        print(f"        ✗ drop {lose['start']:6.2f} {lose['speaker']:4} «{lose['text'][:44]}»")
    print("\n=== genuine overlaps PRESERVED (different text, same time) — spot check ===")
    for i in range(len(kept) - 1):
        a, b = kept[i], kept[i + 1]
        if abs(a["start"] - b["start"]) < TIME_WINDOW and a["speaker"] != b["speaker"]:
            print(f"  {a['start']:6.2f} {a['speaker']:4} «{a['text'][:38]}»  ||  {b['speaker']} «{b['text'][:38]}»")


if __name__ == "__main__":
    main()
