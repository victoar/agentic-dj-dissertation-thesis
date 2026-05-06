---
name: AgDJ — Project overview and architecture
description: Core context for the Agentic DJ dissertation: what it does, architecture, tech stack, and implementation state
type: project
---

**Project:** Agentic DJ (AgDJ) — LLM-powered agent for real-time adaptive music curation.

**Central idea:** Traditional recommenders use static preference vectors from history. AgDJ models the listener as a dynamic system with a 6-dimensional state vector (energy, valence, focus, openness, social, arc_phase) that updates after every track based on feedback signals (skip, replay, thumbs up/down, full listen). An LLM (Gemini Flash) selects the next track with music-theoretic constraints.

**Why:** This is an original research contribution — the direction-aware state update rules (e.g., skipping a high-energy track lowers energy; skipping a low-energy track raises it) are the thesis's novel claim.

**Architecture:** Single-shot ReAct pattern. Full multi-step ReAct was prototyped but abandoned due to Gemini free-tier rate limits (5 req/min → max 2 API calls per cycle).

**Tech stack:**
- Python 3.11, src layout (`src/agentic_dj/`), Hatchling build, pytest
- LLM: Google Gemini Flash via `google-genai` SDK (NOT deprecated `google-generativeai`)
- Spotify: Spotipy with SpotifyPKCE auth. Audio features + recommendations endpoints are deprecated (post Nov 2024) — NOT used
- Last.fm: pylast, disk-cached (`.cache/lastfm/` SHA1-keyed JSON), replaces Spotify audio features
- Music theory: music21 + custom Camelot Wheel (`music/camelot.py`)
- Tag features: 110+ tag lexicon, semantic fallback via sentence-transformers `all-MiniLM-L6-v2`
- UI (not yet built): Streamlit + Plotly

**Key modules:**
- `agent/state.py` — ListenerState, FeedbackEvent, update_state, advance_track, init_state
- `agent/tools.py` — 12 tool functions in 4 groups; session uniqueness guard (_queued_ids, _queued_names); reset_session()
- `agent/loop.py` — Single-shot ReAct cycle, Gemini integration, exponential backoff (12/24/48/96s on 429), fallback scoring, duplicate retry
- `music/camelot.py` — Camelot Wheel, CamelotKey, compatibility_strength(), compatible_positions()
- `music/tags.py` — TAG_LEXICON, estimate_features, estimate_features_with_fallback
- `music/lastfm_client.py` — enrich_track(), disk cache, graceful unknown-track handling
- `spotify/client.py` — SpotifyClient, SpotifyTrack, PlaybackState

**Music-theory hard constraints (pre-LLM filter):**
- Camelot Wheel: same/adjacent/relative major-minor positions only
- BPM jump limits per arc phase: warmup=10, build=15, peak=20, cooldown=10 BPM (falls back to bpm_ok=True when BPM unknown — Spotify audio features deprecated)
- Energy arc alignment: candidate energy within 0.15 of target range

**Session arc:** warmup → build → peak → cooldown (auto-advances by track count).

**What's done:** State model, Camelot Wheel, tag translator, Last.fm client, Spotify client, agent tools, agent loop.

**What remains:** Streamlit UI, weight calibration experiment (20 scripted scenarios), user study infrastructure (15-20 participants, AgDJ vs Spotify radio), evaluation notebooks (4), README for examiner.

**Why:** Master's dissertation deliverable — academic year 2025-2026. Evaluation includes blind transition ratings (100 pairs), user study (counterbalanced design), failure taxonomy (state lag, novelty miscalibration, arc rigidity, key tunnel vision).
