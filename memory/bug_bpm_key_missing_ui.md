---
name: Bug — BPM and key missing in Streamlit UI
description: BPM and harmonic key are not surfaced in the Queue or Now Playing tabs despite being central to the music theory logic
type: project
---

**Bug:** The Queue and Now Playing tabs display `None` for BPM and musical key. These fields are silently omitted from the metadata line in the UI.

**Why this matters:** BPM compatibility and Camelot Wheel key compatibility are the two hard music-theory constraints that define the agent's core selection logic. Not showing them in the UI undermines the thesis's central argument and makes the interface misleading — it looks like a generic recommender rather than a theory-aware agent.

**Root cause:** `_spotify_to_candidate()` in `src/agentic_dj/agent/tools.py` does not populate `bpm` or a `key`/`camelot` field in the candidate dict. Spotify's `audio-features` endpoint (which provided both `tempo` and `key`/`mode`) was deprecated in Nov 2024. The agent already has the Camelot Wheel logic (`music/camelot.py`) and BPM compatibility logic (`estimate_bpm_compatibility()`), but there is no current data source feeding actual BPM or key values into track objects.

**What needs to be resolved:**
1. Source BPM and key data from an alternative provider (Last.fm does not reliably carry BPM; candidates include AcousticBrainz, Essentia, or the `bpmdetect` approach on audio previews).
2. Once sourced, populate `bpm` and `camelot_position` fields in the candidate dict inside `_spotify_to_candidate()`.
3. Surface these fields in:
   - `app/components/now_playing.py` — badge pills ("BPM 105", "Key A♭ maj")
   - `app/components/queue.py` — metadata line per track row

**How to apply:** Do not close this bug by hardcoding values or showing "—". The fix requires a real data source for BPM and key. Prioritise after Streamlit integration (Steps A–D) is complete.
