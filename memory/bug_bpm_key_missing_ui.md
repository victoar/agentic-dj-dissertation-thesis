---
name: BPM source — Deezer API (resolved)
description: BPM was missing from UI; now sourced via Deezer public API (two-step: search + track detail). Key remains None (documented limitation).
type: project
---

**Resolved.** BPM is now fetched via Deezer's public API (no auth required) in `src/agentic_dj/music/deezer_client.py`.

**Implementation:**
- `fetch_bpm(artist, title)` does a two-step fetch: search for track ID, then `GET /track/{id}` for the BPM field (BPM is absent from search results, present in track detail).
- Disk cache: `.cache_deezer/` SHA1-keyed JSON — same pattern as `.cache_lastfm/`.
- `_spotify_to_candidate()` in `tools.py` calls `fetch_bpm()` after Last.fm enrichment and sets `"bpm": result["bpm"]`, `"key": None`.
- `bridge.py` `adapt_now_playing()` and `adapt_queue()` now read `bpm` from history entries instead of hardcoding `None`.

**Remaining known limitation:** Harmonic key is unavailable from any streaming-accessible API without audio analysis. Key badge in the UI shows `—`. This is documented in `project_known_limitations.md`.

**Why:** Spotify's `audio-features` endpoint (which had `tempo` and `key`) was deprecated Nov 2024. GetSongBPM was evaluated but requires a live website backlink. Deezer was chosen as the simplest free/no-auth alternative.
