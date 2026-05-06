# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable install from src layout)
pip install -e .

# Run all tests
pytest

# Run a single test file
pytest tests/test_state.py -v

# Run a single test by name
pytest tests/test_camelot.py::test_compatible_positions -v

# Run with coverage
pytest --cov=agentic_dj

# Run the agent loop manually (requires Spotify open + active device)
python -c "from agentic_dj.agent.loop import run_agent_cycle; run_agent_cycle(verbose=True)"
```

Tests that hit live APIs (`test_loop.py`, `test_spotify_client.py`, `test_lastfm_client.py`) require `.env` to be populated and a Spotify device to be active. Unit tests (`test_state.py`, `test_camelot.py`, `test_tags.py`, `test_tags_fallback.py`) have no external dependencies.

## Architecture

The system has three conceptual layers:

**1. State model** (`agent/state.py`)
`ListenerState` is a 6-dimensional float vector (energy, valence, focus, openness, social, arc_phase). `update_state()` applies direction-aware delta rules: the sign of the energy/valence update depends on the *track's own profile* — skipping a high-energy track pushes energy down, skipping a low-energy track pushes it up. `advance_track()` increments the play counter and fires arc-phase transitions (warmup → build → peak → cooldown) at fixed track count thresholds.

**2. Tool catalogue** (`agent/tools.py`)
12 functions grouped into four sets: state read/write, music theory (Camelot + BPM), Spotify+Last.fm search/enrichment, and queue management. All session state (listener state vector, queue, history, played-track sets) lives as module-level globals. `reset_session()` clears everything. The duplicate guard uses both `_queued_ids` (Spotify track ID set) and `_queued_names` (lowercase name set) to prevent any repeat across a session.

**3. Agent loop** (`agent/loop.py`)
Single-shot ReAct cycle — all context is collected in Python (no LLM tool calls for observation), then one Gemini call selects the track. Steps:
1. Apply feedback (if any) → update state
2. Collect state, arc, playback, history, queue
3. Two Spotify+Last.fm searches → deduplicated candidate list
4. BPM compatibility scored per candidate (falls back to `bpm_ok=True` when BPM unknown)
5. Single Gemini call with a structured JSON prompt → parse selection
6. Fallback to `_best_fallback()` (energy+valence distance scorer) if Gemini parse fails
7. Duplicate retry: if the selected track was already played, sort remaining candidates by state distance and try each in order

**Music intelligence** (`music/`)
- `camelot.py`: `CamelotKey`, `compatible_positions()` (returns 4 positions), `compatibility_strength()` (0.0–1.0 score). Wheel wrap-around at 12→1 handled via `_wheel_distance()`.
- `tags.py`: `TAG_LEXICON` maps Last.fm tags to `(energy_contribution, valence_contribution)` on −1..+1. `estimate_features_with_fallback()` uses sentence-transformer `all-MiniLM-L6-v2` for unknown tags (cosine threshold 0.45, contribution attenuated 0.6×, lazy-loaded).
- `lastfm_client.py`: `enrich_track()` / `fetch_enrichment()` — disk-cached (`.cache/lastfm/` SHA1-keyed JSON).

**Spotify client** (`spotify/client.py`)
SpotifyPKCE auth via Spotipy. `is_saved()` checks the user's library (familiarity signal). Audio features and recommendations endpoints are **not used** — deprecated for apps registered after Nov 2024.

## Critical constraints

- **LLM**: Use `google-genai` SDK (`from google import genai`). The model ID is `gemini-3-flash-preview`. **Do not** use `google-generativeai` (deprecated). Rate limit is 5 req/min; the loop uses exponential backoff (12s, 24s, 48s, 96s on 429).
- **Spotify**: `audio-features` and `recommendations` endpoints return 403/404 — never call them. The popularity field is always 0 in Development Mode — use Last.fm listener counts instead. Playback control requires Spotify Premium.
- **BPM**: Spotify audio-features (which included tempo) is gone. Track BPM is currently not populated; BPM checks fall back to `bpm_ok=True`.
- **State is module-level**: `tools.py` holds all session state as module globals. Not thread-safe; single session at a time.
- **Python 3.11**: Required (not 3.13) for LangGraph compatibility.
