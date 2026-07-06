#!/usr/bin/env python3
"""
Synthetic multi-topic meeting corpus for testing CROSS-APP chat retrieval.

Writes demo.sqlite with the same schema subset the harness reads (meetings,
transcripts, summary_processes). Every meeting has planted, checkable facts so
this doubles as a recall eval set. Fully synthetic — no real people/PHI.

Try questions like:
  - "What did we decide about the Q3 budget?"                 -> Budget Review
  - "What was the root cause of the security incident?"        -> Security Postmortem
  - "When do we launch Meetily Pro and what's the coupon?"     -> Marketing Launch
  - "Did we pick embeddings or FTS for search, and why?"       -> Architecture Review
  - "Summarize every hiring decision across all meetings."     -> Hiring + Budget
"""
from __future__ import annotations

import json
import os
import sqlite3

# (title, date, [(speaker, seconds, text)...], summary_lines[])
MEETINGS = [
    ("Q3 Budget Review", "2026-06-02", [
        ("Jad", 0, "Let's lock the Q3 budget today. Marketing is running hot."),
        ("Priya", 18, "Proposal is to cut marketing spend by fifteen percent this quarter."),
        ("Jad", 41, "Agreed. Reallocate that fifteen percent — put fifty thousand dollars into hiring."),
        ("Sam", 70, "And we should freeze all non-essential travel until Q4."),
        ("Jad", 95, "Yes, travel freeze approved. So: minus 15% marketing, +$50k hiring, travel frozen."),
        ("Priya", 130, "I'll update the finance sheet and send it around by Friday."),
    ], ["Q3 budget decisions", "Cut marketing spend by 15%",
        "Reallocate $50,000 to hiring", "Freeze non-essential travel until Q4",
        "Priya to update finance sheet by Friday"]),

    ("Hiring Sync", "2026-06-05", [
        ("Sam", 0, "We have headcount for three roles this quarter."),
        ("Jad", 22, "Approved: two backend engineers and one product designer."),
        ("Sam", 55, "For the pipeline, I want to move us onto the Ashby applicant tracking system."),
        ("Jad", 80, "Fine, adopt Ashby. Target start dates in September."),
        ("Mouad", 110, "I'll draft the backend job descriptions this week."),
    ], ["Hiring plan", "Approved 2 backend engineers and 1 product designer",
        "Adopt Ashby ATS for the pipeline", "Target start dates in September",
        "Mouad to draft backend JDs"]),

    ("Security Incident Postmortem", "2026-06-09", [
        ("Mouad", 0, "Recapping the incident. An API key leaked through a compromised dependency."),
        ("Jad", 20, "It came in via the LiteLLM package — a supply-chain style issue."),
        ("Mouad", 48, "Root cause: the leaked key was committed transitively; we had no dependency audit in CI."),
        ("Jad", 79, "Remediation: we rotated all keys and added pip-audit to the pipeline."),
        ("Sam", 105, "Let's keep this blameless. Action items: pin hashes, audit .pth files."),
        ("Jad", 132, "Agreed, blameless. Hash-pinning and pip-audit are now mandatory."),
    ], ["Security incident postmortem", "API key leaked via the LiteLLM dependency (supply chain)",
        "Root cause: no dependency audit in CI, key committed transitively",
        "Rotated all keys, added pip-audit, mandatory hash-pinning", "Blameless culture"]),

    ("Product Roadmap Q3", "2026-06-12", [
        ("Jad", 0, "Roadmap priorities for Q3. Top of the list?"),
        ("Priya", 15, "Mobile app is the number one priority for the quarter."),
        ("Jad", 38, "Agreed, mobile first. We defer the analytics dashboard to Q4."),
        ("Mouad", 66, "I'd like the meeting-chat feature — chatting with recordings — shipped by August."),
        ("Jad", 92, "Yes, ship meeting-chat by August. Mobile, then chat, dashboard deferred."),
    ], ["Q3 roadmap", "Priority 1: mobile app", "Defer analytics dashboard to Q4",
        "Ship meeting-chat (chat with recordings) by August"]),

    ("Customer Bug Triage — ACME", "2026-06-16", [
        ("Sam", 0, "ACME Corp reported a crash when exporting data."),
        ("Mouad", 19, "It only happens above ten thousand rows — the data export crashes."),
        ("Mouad", 47, "Root cause is a pagination bug in the export path."),
        ("Jad", 72, "Ship a hotfix. Tag it version 2.3.1 and notify ACME today."),
        ("Sam", 99, "Hotfix 2.3.1 is out. ACME confirmed the export works now."),
    ], ["ACME data-export crash", "Crash triggered above 10,000 rows",
        "Root cause: pagination bug in export path", "Hotfix shipped as v2.3.1", "ACME confirmed fixed"]),

    ("Marketing Launch Plan — Meetily Pro", "2026-06-19", [
        ("Priya", 0, "Launch plan for Meetily Pro. Proposed date is August 15th."),
        ("Jad", 20, "August 15 works. Lead the messaging with the privacy angle — local, no cloud."),
        ("Priya", 52, "We'll run a launch coupon, code LAUNCH20, for twenty percent off."),
        ("Sam", 80, "I'll book a podcast tour focused on data sovereignty."),
        ("Jad", 104, "Great. Meetily Pro launches Aug 15, coupon LAUNCH20, privacy-first narrative."),
    ], ["Meetily Pro launch", "Launch date August 15", "Coupon code LAUNCH20 (20% off)",
        "Privacy-first / data-sovereignty messaging", "Podcast tour booked"]),

    ("Architecture Review — Search", "2026-06-23", [
        ("Mouad", 0, "For meeting search, should we use a vector database and embeddings?"),
        ("Jad", 22, "Transcripts are lexical and small. I don't want to maintain an embedding index."),
        ("Jad", 50, "Decision: use SQLite FTS5 full-text search, not a vector DB. Drop the embeddings idea."),
        ("Sam", 78, "Separately, for live status we should move from polling to SSE."),
        ("Jad", 100, "Yes — adopt SSE for status updates, retire the polling endpoints."),
    ], ["Search architecture", "Decision: use SQLite FTS5, NOT a vector database / embeddings",
        "Reason: transcripts are lexical and small; avoid index maintenance",
        "Move status updates from polling to SSE"]),

    ("Weekly 1:1 — Jad & Mouad", "2026-06-26", [
        ("Jad", 0, "Ownership check-in. You take the ontology and SQL agents."),
        ("Mouad", 16, "Got it, I own ontology plus the SQL agents going forward."),
        ("Jad", 40, "I'll focus on the compiler work. Our gate is August 1st."),
        ("Mouad", 63, "Understood, Aug 1 gate. I'll have the ontology tools wired by then."),
    ], ["1:1 ownership", "Mouad owns ontology + SQL agents", "Jad focuses on the compiler",
        "Shared gate: August 1"]),
]


def seed_demo(path: str | None = None) -> str:
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo.sqlite")
    if os.path.exists(path):
        os.remove(path)
    con = sqlite3.connect(path)
    con.executescript(
        """
        CREATE TABLE meetings (id TEXT PRIMARY KEY, title TEXT, created_at TEXT, updated_at TEXT);
        CREATE TABLE transcripts (
            id TEXT PRIMARY KEY, meeting_id TEXT, transcript TEXT, timestamp TEXT,
            audio_start_time REAL, speaker TEXT);
        CREATE TABLE summary_processes (meeting_id TEXT PRIMARY KEY, status TEXT, result TEXT);
        """
    )
    for mi, (title, date, segs, summary_lines) in enumerate(MEETINGS):
        mid = f"demo-{mi:02d}"
        created = f"{date}T09:00:00+00:00"
        con.execute("INSERT INTO meetings VALUES (?,?,?,?)", (mid, title, created, created))
        for si, (speaker, secs, text) in enumerate(segs):
            con.execute(
                "INSERT INTO transcripts VALUES (?,?,?,?,?,?)",
                (f"{mid}-{si:03d}", mid, text, created, float(secs), speaker),
            )
        result = json.dumps({"sections": [{"title": "Key Points", "blocks": summary_lines}]})
        con.execute("INSERT INTO summary_processes VALUES (?,?,?)", (mid, "completed", result))
    con.commit()
    con.close()
    return path


if __name__ == "__main__":
    print(seed_demo())
