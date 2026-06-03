# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable install from src layout)
pip install -e .

# Run the Streamlit app (requires .env populated)
streamlit run app/app.py

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

**2. Tool catalogue** (`agent/tools.py` + `agent/lc_tools.py`)
`tools.py` holds 15 plain functions grouped into four sets: state read/write, music theory (Camelot + BPM), Spotify+Last.fm search/enrichment, and queue management. These stay plain callables (the Streamlit `bridge.py` and the loop's Python pre-step call them directly). `agent/lc_tools.py` wraps each as a LangChain `@tool` object — the single source of truth the LLM sees. It exports `ALL_TOOLS`, `CYCLE_TOOLS` (no search tools, for mid-session), `OPENER_TOOLS` (session start), and the synthetic `select_opening_track` terminal. All session state (listener state vector, queue, history, played-track sets) lives as module-level globals in `tools.py`. `reset_session()` clears everything. The duplicate guard uses both `_queued_ids` (Spotify track ID set) and `_queued_names` (lowercase name set) to prevent any repeat across a session.

**3. Agent graphs** (`agent/graph.py` + `agent/loop.py`)
The agent is a compiled **LangGraph** ReAct graph (`agent/graph.py`), driven from `agent/loop.py`. Two graphs share `agent`/`tools`/`explain` nodes and routers via `build_dj_graph` (mid-session, stops on `add_track_to_queue`) and `build_opener_graph` (session start, stops on `select_opening_track`). Control flow: `agent → tools → agent`, routing to a tool-less `explain` turn once the stop tool succeeds. The model's reasoning is read from `additional_kwargs["reasoning_content"]` (gpt-oss puts it on a separate channel; `.content` is empty on tool-call turns). `run_agent_cycle` steps:
1. Apply feedback (if any) → update state in Python immediately (step-0 trace entry)
2. Python pre-fetch: Last.fm tag search → scored shortlist injected into the user message
3. Run the cycle graph (`CYCLE_TOOLS`); the graph reasons, verifies harmonic fit, and commits
4. Graph ends when `add_track_to_queue` succeeds; the `explain` node returns the listener explanation
5. Fallback to `_best_fallback()` (energy+valence distance scorer) if the graph exits without queuing

`start_session` runs the opener graph; its `select_opening_track` anti-hallucination guard (only commit a track actually seen in a search result) lives in the opener graph node, validating against `seen_tracks` in graph state. A single ChatGroq call extracts the vibe vector first.

**Music intelligence** (`music/`)
- `camelot.py`: `CamelotKey`, `compatible_positions()` (returns 4 positions), `compatibility_strength()` (0.0–1.0 score). Wheel wrap-around at 12→1 handled via `_wheel_distance()`.
- `tags.py`: `TAG_LEXICON` maps Last.fm tags to `(energy_contribution, valence_contribution)` on −1..+1. `estimate_features_with_fallback()` uses sentence-transformer `all-MiniLM-L6-v2` for unknown tags (cosine threshold 0.45, contribution attenuated 0.6×, lazy-loaded).
- `lastfm_client.py`: `enrich_track()` / `fetch_enrichment()` — disk-cached (`.cache/lastfm/` SHA1-keyed JSON).

**Spotify client** (`spotify/client.py`)
SpotifyPKCE auth via Spotipy. `is_saved()` checks the user's library (familiarity signal). Audio features and recommendations endpoints are **not used** — deprecated for apps registered after Nov 2024.

## Critical constraints

- **LLM**: Use `ChatGroq` from `langchain-groq` (the agent runs on LangGraph). Model: `openai/gpt-oss-120b`. API key: `GROQ_API_KEY`. Rate-limit backoff is delegated to `ChatGroq(max_retries=4)` — there is no custom retry layer. Reasoning text is on `additional_kwargs["reasoning_content"]`, not `.content`. Do not reintroduce the raw `groq` SDK loop or `google-genai` (both removed).
- **Spotify**: `audio-features` and `recommendations` endpoints return 403/404 — never call them. The popularity field is always 0 in Development Mode — use Last.fm listener counts instead. Playback control requires Spotify Premium.
- **BPM**: Spotify audio-features (which included tempo) is gone. Track BPM is currently not populated; BPM checks fall back to `bpm_ok=True`.
- **State is module-level**: `tools.py` holds all session state as module globals. Not thread-safe; single session at a time.
- **Python 3.11**: Required (not 3.13) for LangGraph compatibility.
