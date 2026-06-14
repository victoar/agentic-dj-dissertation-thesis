# Agentic DJ — System Flow Documentation

*Functional analysis of the current implementation, prepared as source material for the dissertation. Generated 2026-06-09 from a read of the `src/agentic_dj` package and the Streamlit application layer.*

---

## 1. Purpose and scope

This document describes **how the system actually works today**, at the functional level: how the agent is invoked, the tools it has at its disposal, the music-theory rules used to find a compatible next track, and how the listener-state model evolves over a session. It is intended as raw material for the written dissertation, so it favours completeness and faithfulness to the code over brevity.

The system is an **Agentic DJ**: an autonomous music-selection agent that continuously chooses and queues the next track on a live Spotify session, reasoning over a model of the listener's evolving state and the rules of harmonic mixing. The agent is implemented as a **LangGraph ReAct graph** driven by an LLM (`openai/gpt-oss-120b`, served through Groq), surrounded by a deterministic Python layer that handles music discovery, harmonic scoring, and Spotify playback mechanics.

---

## 2. High-level architecture

The system separates **reasoning** (what the LLM decides) from **mechanics** (deterministic Python that gathers candidates, scores harmony, and controls playback). There are three conceptual layers plus the music-intelligence and integration modules.

```
┌──────────────────────────────────────────────────────────────┐
│  Streamlit UI  (app/app.py + app/components/*)                │
│  Now Playing · Listener State · Queue · Agent Trace tabs      │
└──────────────┬───────────────────────────────────────────────┘
               │  calls
┌──────────────▼───────────────────────────────────────────────┐
│  Bridge  (app/bridge.py)                                      │
│  session_state init · feedback dispatch · buffer management   │
└──────────────┬───────────────────────────────────────────────┘
               │  calls
┌──────────────▼───────────────────────────────────────────────┐
│  Agent orchestration  (agent/loop.py)                         │
│  run_agent_cycle · start_session                              │
│        │                                                      │
│        ▼ drives                                               │
│  LangGraph ReAct graphs  (agent/graph.py)                     │
│  build_dj_graph (mid-session) · build_opener_graph (start)    │
│        │                                                      │
│        ▼ the LLM calls                                        │
│  LangChain tool objects  (agent/lc_tools.py)  ── wrap ──▶     │
│  Plain tool functions  (agent/tools.py)  ── hold session ──   │
│  state as module globals; call music + spotify layers         │
└──────────────┬───────────────────────────────────────────────┘
               │  uses
┌──────────────▼───────────────────────────────────────────────┐
│  Music intelligence (music/)     │  Integrations             │
│  camelot.py · tags.py            │  spotify/client.py         │
│  lastfm_client.py                │  (Spotipy, PKCE auth)      │
│  soundcharts_client.py           │  Last.fm, Soundcharts,     │
│  state.py (listener model)       │  RapidAPI (playlist reads) │
└──────────────────────────────────────────────────────────────┘
```

Two design decisions dominate the architecture:

1. **The LangChain tool wrappers (`lc_tools.py`) are the single source of truth for what the LLM can see.** Each is a thin `@tool`-decorated wrapper around a plain function in `tools.py`. The business logic stays plain and directly callable, so the Streamlit bridge and the Python fallback paths can call `tool_module.<fn>()` without going through the LLM.

2. **All session state lives as module-level globals in `tools.py`** (the listener-state vector, the queue, play history, the played-track guard sets, and the resolve/enrichment caches). This is simple and keeps state inspectable, but it is **not thread-safe and supports a single session at a time**. `reset_session()` clears everything except the played-track guard.

---

## 3. The listener-state model (`agent/state.py`)

The core abstraction is `ListenerState`, a **six-dimensional vector** of floats in `[0.0, 1.0]` plus a categorical arc phase. It is the agent's working model of who it is playing for *right now*.

| Dimension   | Meaning                                  | Low end ↔ high end                       | Default |
|-------------|------------------------------------------|------------------------------------------|---------|
| `energy`    | Physical arousal                         | calm ↔ fast / intense                    | 0.5     |
| `valence`   | Emotional tone                           | melancholic ↔ happy                      | 0.5     |
| `focus`     | Engagement mode                          | active listening ↔ background            | 0.5     |
| `openness`  | Willingness to hear novelty              | familiar only ↔ try new things           | 0.7     |
| `social`    | Listening context                        | solo ↔ group / party                     | 0.3     |
| `arc_phase` | Session narrative stage (categorical)    | warmup → build → peak → cooldown         | warmup  |

Internal counters (`_tracks_played`, `_session_minutes`, `_history`) are tracked but not exposed directly to the agent. `to_dict()` rounds the five floats, the arc phase string, and the play count into the dict the LLM sees.

### 3.1 Feedback-driven updates — `update_state()`

State is moved by **feedback events**, each carrying a fixed weight vector (`WEIGHTS` table in `state.py`). The seven events are: `early_skip` (skipped before 30 %), `skip`, `partial_listen` (30–85 %), `full_listen` (> 85 %), `replay`, `thumbs_up`, and `thumbs_down`.

The conceptually important rule is **direction-aware updating**: the *sign* of an energy or valence update depends on the **track's own profile**, not just the event.

```python
# Energy update — sign depends on whether the track was high or low energy
if "energy" in weights:
    direction = 1 if track.energy_est >= 0.5 else -1
    state.energy = _clamp(state.energy + weights["energy"] * direction)
```

So skipping a *high-energy* track pushes the listener's modelled energy **down** (they did not want that intensity); skipping a *low-energy* track pushes it **up**. The same logic applies to valence using `track.valence_est`. `openness` and `focus` updates are direction-independent. All results are clamped to `[0, 1]`. Each update also appends a record to `_history`.

Representative weights (per event): an `early_skip` applies `energy −0.15, valence −0.08, openness −0.10, focus +0.05` (skipping is read as *active* engagement, hence focus rises); a `full_listen` applies `energy +0.06, valence +0.04, openness +0.04, focus −0.04`; a `thumbs_up` is the strongest positive signal (`valence +0.10, energy +0.08`).

### 3.2 Session arc — `advance_track()`

Each newly *played* track increments `_tracks_played` and may trigger an **arc-phase transition** at fixed track-count thresholds (`ARC_TRANSITIONS`):

- **warmup** → **build** after 4 tracks
- **build** → **peak** after 8 more (12 total)
- **peak** → **cooldown** after 4 more (16 total)
- **cooldown** stays until the session ends

The arc phase is a deliberate dramaturgical model of a DJ set (ease in → build energy → climax → wind down). It feeds two downstream mechanisms: the BPM-tolerance window and the candidate ranker's target energy band (see §6).

### 3.3 Initialisation

State can be seeded two ways:

- `init_state(context)` — named presets (`study`, `workout`, `party`, `focus`, `chill`, `general`), each a hand-tuned starting vector. E.g. `party = (energy 0.8, valence 0.8, focus 0.2, openness 0.8, social 0.9)`.
- `init_state_from_values(...)` — explicit floats, used when the opener LLM call infers a vibe vector from a free-text session description (see §5).

---

## 4. How the agent is invoked

There are **two public entry points** in `agent/loop.py`, each backed by its own compiled LangGraph graph. Both follow the ReAct pattern: the model is given a goal and a toolbox and decides which tools to call, in what order, and when it has gathered enough to commit. There are no procedural selection rules in the prompt — the constraints are enforced by the tools themselves.

### 4.1 The graph machinery (`agent/graph.py`)

Both entry points compile a `StateGraph` over a shared control flow:

```
START → agent
agent → tool_calls?   yes → tools        no → END (final text = explanation)
tools → stop tool fired & succeeded?   yes → END     no → agent
```

- **`agent` node** invokes the tool-bound LLM, extracts reasoning, and emits trace entries. gpt-oss on Groq puts its chain-of-thought on a **separate channel** — `additional_kwargs["reasoning_content"]` — and leaves `.content` empty on tool-call turns, so reasoning is read from there (`_extract_reasoning`).
- **`tools` node** executes each requested tool against a name→object registry, records an `act` trace entry per call, and watches for the **stop tool**. When the stop tool returns `{"success": True}`, its result becomes `terminal` and the graph ends.
- **Routers** (`_route_after_agent`, `_route_after_tools`) decide whether to loop back or terminate.

The graph carries a typed `DJState` (`messages`, `trace`, `step`, `terminal`, `explanation`, `seen_tracks`, `choice`). `messages` uses the `add_messages` reducer (append semantics); the rest are overwrite-on-return.

**Robustness handling baked into the graph:**

- *Harmony-token leakage.* gpt-oss intermittently leaks a channel token into the tool name (e.g. `search_tracks<|channel|>commentary`) or a `functions.` prefix. `_clean_tool_name` / `_sanitize_tool_calls` strip these in place on both the parsed and raw tool-call structures so execution, the trace, and the next-turn message history all see the real name. The LLM is built with `disable_tool_validation: True` so Groq returns the malformed call instead of 400-ing mid-generation.
- *Malformed tool calls.* `_invoke_with_tool_retry` re-samples up to 3 times on Groq's `tool_use_failed` 400; if all fail it returns an empty `AIMessage` so the graph ends gracefully and the Python fallback takes over rather than crashing.
- *Rate limiting.* Backoff is delegated entirely to `ChatGroq(max_retries=4)`; there is no custom retry layer.
- *Recursion cap.* `run_graph` sets `recursion_limit = max_iterations * 2 + 2` (15 iterations → cap of 32 graph steps).

### 4.2 Mid-session selection — `run_agent_cycle()`

This is the per-track workhorse, called whenever a track is needed (after feedback, or to top up the buffer). Steps:

1. **Apply feedback in Python first (step 0).** If a feedback event is passed, `update_listener_state()` is applied *before* the graph runs, so when the LLM reads `get_listener_state()` it already sees the post-feedback vector. The feedback is logged as a step-0 `act` trace entry.
2. **Build the prompt.** The system prompt (`CYCLE_SYSTEM_PROMPT`) tells the model to call `get_ranked_candidates()`, optionally verify a borderline pick with `check_transition` / `get_compatible_keys`, then commit with `add_track_to_queue(...)`. A `context_note` is appended to the human message if feedback occurred.
3. **Run the cycle graph** built from `CYCLE_TOOLS` (the full toolset *minus* the four granular search tools, *plus* `get_ranked_candidates`), with `reasoning_effort = "medium"` and `add_track_to_queue` as the stop tool.
4. **Commit.** The graph ends when `add_track_to_queue` succeeds. The one-sentence `reason` the model passes at commit time *is* the listener-facing explanation (no separate explain turn is needed in the happy path).
5. **Fallback if no commit.** If the graph exits without queuing, Python calls `get_ranked_candidates(limit=1)` and queues the top-ranked candidate directly, synthesising a generic explanation. A final fallback surfaces the model's last `think` entry if the explanation is still empty.

The return shape is `{explanation, queued_track, trace, success}`.

Crucially, in the mid-session cycle **discovery and ranking happen inside the single `get_ranked_candidates` tool** (deterministic Python). The LLM's job is narrowed to *judgement*: weighing the pre-screened, pre-ranked shortlist against the arc and state, occasionally overriding rank #1, and articulating why.

### 4.3 Session bootstrap — `start_session(description)`

Starting a session from a free-text description (e.g. *"2016 clubbing vibes"*) is a **two-phase** design:

**Phase 1 — structured vibe extraction.** A single non-tool `ChatGroq` call (`_interpret_description`) parses the description into a JSON vibe object: the five state floats, `search_queries` (2–4 short playlist-style strings), `session_label`, and a `confident` flag. The parser is defensive — it strips markdown fences, reads from `reasoning_content` when `.content` is empty, and extracts the outermost `{...}` from any surrounding prose. The parsed vibe seeds the state vector via `init_state_from_values`.

**Phase 2 — ReAct opener sub-loop.** `_react_pick_opener` runs the **opener graph** (`OPENER_TOOLS`: `search_tracks_by_playlist`, `search_tracks`, `get_track_details`, and the synthetic terminal `select_opening_track`). The model searches, inspects candidates, and commits one opener.

The opener graph carries a notable **anti-hallucination guard**. `select_opening_track`'s body is never executed; instead the opener tools node validates the chosen `(name, artist)` against a `seen_tracks` set built from every candidate the model has actually seen in prior search results. A track that was never returned by a search is rejected with an error listing the available candidates. A partial-name match path tolerates minor punctuation differences. This guarantees the committed opener is a real, resolvable track.

Around the loop, Python handles the mechanics: `reset_session`, resolving the chosen track to a real `SpotifyTrack`, starting playback via `_spotify.play()`, recording the track in session state, and pre-warming the sentence-transformer model in a background daemon thread so it is ready before the first mid-session cycle. If interpretation or the opener fails to yield a playable track, it **falls back to a random pick from the user's Spotify top tracks** (relabelling the session "Your recent favourites" when not confident). Failure modes return `{"success": False, "error": "no_device" | "no_candidates"}`.

---

## 5. The tool catalogue

The LLM-callable tools are defined in `lc_tools.py` (the descriptions the model sees) and implemented in `tools.py`. They fall into five groups. `ALL_TOOLS` lists every tool; `CYCLE_TOOLS` is `ALL_TOOLS` minus the four search tools plus `get_ranked_candidates`; `OPENER_TOOLS` is the focused opener subset.

### 5.1 State tools

| Tool | Function |
|------|----------|
| `get_listener_state()` | Returns the six-dimensional vector + tracks played. Called to orient each cycle. |
| `update_listener_state(event, track_name, artist)` | Applies a feedback event (looks the track up in history for its energy/valence profile). |
| `get_session_arc()` | Returns the current arc phase plus a plain-language instruction (e.g. peak = "choose the most powerful, high-energy tracks now"). |

### 5.2 Music-theory tools

| Tool | Function |
|------|----------|
| `get_compatible_keys(camelot_position)` | Returns the four Camelot positions that form smooth transitions from the given key. |
| `check_transition(from_camelot, to_camelot)` | Returns a smoothness score 0.0–1.0 + verdict (≥ 0.75 smooth, 0.4–0.74 acceptable, < 0.4 avoid). |
| `estimate_bpm_compatibility(current_bpm, candidate_bpm, arc_phase)` | Checks whether a tempo jump is acceptable for the current phase (±8 BPM steady, ±25 BPM transitional). |

### 5.3 Discovery / search tools

| Tool | Source | Use |
|------|--------|-----|
| `search_tracks_by_tag(tag, limit)` | Last.fm tag index → resolved to Spotify | Mood/genre queries (*energetic*, *chill*, *dark*). |
| `search_artist_tracks(artist, limit)` | Last.fm artist top tracks → Spotify | An artist's catalogue, ranked by listener count (avoids covers/tributes). |
| `search_tracks(query, limit)` | Spotify text search + Last.fm enrichment | Precise track/artist lookups. |
| `search_tracks_by_playlist(query, sample_size)` | Spotify playlist search → RapidAPI track reads | Vibe/era queries (*2016 pop hits*); raw metadata only. |
| `get_track_details(track_name, artist)` | Spotify + full enrichment | Complete feature profile for one known track. |
| `get_current_playback()` | Spotify | What is actually playing — grounds reasoning. |

### 5.4 Queue tools

| Tool | Function |
|------|----------|
| `get_queue_state()` | The planned upcoming queue (last 5 queued). |
| `get_session_history()` | Tracks played so far, most recent first (last 8). |
| `add_track_to_queue(track_name, artist, reason)` | **Terminal action** of the mid-session cycle. Resolves, dedupes, queues on Spotify, records in history, advances the arc, and returns `reason` as the explanation. |

### 5.5 The composite selection tool — `get_ranked_candidates(limit=5)`

This single tool subsumes discovery + filtering + ranking for the mid-session cycle and is detailed in §6. It is what makes the mid-session LLM call cheap and constrained.

### 5.6 The duplicate guard

Two module-level sets, `_queued_ids` (Spotify track IDs) and `_queued_names` (lowercase names), prevent any track repeating within a session. They are checked at every search, queue, and bucket-fill step, and are **deliberately preserved across `reset_session()`** so a context switch cannot replay a track. Session-scoped caches (`_resolve_cache` by query, `_candidate_cache` by Spotify ID) prevent the same track being re-searched and re-enriched at each stage of a commit.

---

## 6. Music theory: finding the compatible candidate

This is the analytical heart of the system. Two bodies of theory are applied — **harmonic mixing (the Camelot Wheel)** and **tempo matching (BPM)** — and combined with **listener-state fit** and **session-arc alignment** into a single composite ranking.

### 6.1 Harmonic compatibility — the Camelot Wheel (`music/camelot.py`)

The Camelot Wheel maps all 24 musical keys onto a 12-position clock face with a letter suffix (`A` = minor, `B` = major). The module provides a full bidirectional mapping between Spotify's `(key 0–11, mode 0/1)` encoding and Camelot positions (`CAMELOT_FROM_SPOTIFY`, `CAMELOT_TO_NAME`).

A transition is **harmonically compatible** under three rules (`compatible()`):

1. **Same position** (identical key) — perfect.
2. **Adjacent position, same letter** (e.g. 8B → 9B or 8B → 7B) — smooth energy step.
3. **Same number, different letter** (e.g. 8B → 8A) — relative major/minor mode switch.

`compatibility_strength()` turns this into a continuous 0.0–1.0 score used for ranking:

| Relationship | Strength |
|--------------|----------|
| Identical key | 1.00 |
| Adjacent, same letter (wheel distance 1) | 0.85 |
| Relative major/minor (same number) | 0.75 |
| Two steps, same letter | 0.55 |
| Adjacent relative (diff letter, distance 1) | 0.40 |
| Three steps, same letter | 0.30 |
| Far apart | 0.05–0.10 |

Wheel wrap-around (12 → 1) is handled by `_wheel_distance`, which takes the shorter of the clockwise/anticlockwise distances. `compatible_positions()` returns the four "green" positions used to filter candidates. The **acceptability threshold used throughout the pipeline is 0.4** (`_HARMONIC_MIN`), matching `check_transition`'s "acceptable" verdict.

### 6.2 Tempo compatibility — BPM windows

BPM tolerance is **arc-phase-dependent** (`estimate_bpm_compatibility` and `_BPM_THRESHOLDS`):

- **Steady phases** (warmup, peak): tight **±8 BPM** — the session is holding a level.
- **Transition phases** (build, cooldown): loose **±25 BPM** — energy is intentionally shifting, so larger tempo steps are fine.

### 6.3 Where the key/BPM data comes from — Soundcharts

Spotify's `audio-features` endpoint (which historically supplied key, mode, and tempo) returns 403/404 for apps registered after Nov 2024 and is **never called**. Instead, key/mode/BPM are sourced from **Soundcharts** (`music/soundcharts_client.py`). `fetch_track_info(spotify_id)` queries Soundcharts by Spotify ID (falling back to ISRC, a universal identifier, on a 404), reads `audio.tempo`, `audio.key`, and `audio.mode`, and converts `(key, mode)` to a Camelot position via the same `camelot.from_spotify` mapping. Results are disk-cached (`.cache_soundcharts/`). When a track has neither BPM nor key, the harmonic/tempo checks **degrade gracefully** — unknown values pass the filter and contribute a neutral 0.5 to ranking, so missing data never empties the candidate pool.

> *Note for the writeup:* the `CLAUDE.md` constraint note states BPM "is currently not populated" and falls back to `bpm_ok=True`. The Soundcharts integration (`soundcharts_client.py`, wired into `_spotify_to_candidate`) is the live mechanism that now supplies key and BPM, so the system does perform real harmonic and tempo scoring when Soundcharts has the track indexed. This is a point worth confirming empirically with logged sessions.

### 6.4 Energy / valence estimation from tags (`music/tags.py`)

Tracks have no native energy/valence numbers, so they are **estimated from Last.fm tags**. `TAG_LEXICON` is a curated dictionary mapping ~140 tags to an `(energy, valence)` contribution on a −1..+1 scale — tempo/intensity words push energy (*aggressive* = `+0.8, −0.3`), mood words push valence (*melancholic* = `−0.4, −0.7`), genres carry weak signals (*pop* = `+0.3, +0.4`).

`estimate_features()` takes a popularity-weighted average of matched tag contributions and shifts the result from −1..+1 onto the 0..1 state scale (neutral = 0.5). Confidence scales with the number of matched tags (capped at 5).

`estimate_features_with_fallback()` adds a **semantic fallback** for tags absent from the lexicon: a lazily loaded sentence-transformer (`all-MiniLM-L6-v2`) embeds the unknown tag and the lexicon keys, and the nearest neighbours above cosine similarity 0.45 lend their values, attenuated by 0.6× because inferred values are less reliable. The model is pre-warmed in a background thread at session start to avoid a cold-start spike mid-cycle.

### 6.5 The composite ranking pipeline — `get_ranked_candidates`

This is the deterministic engine the mid-session agent leans on. Given the current listener state and the currently playing track's profile, it runs **gather → hard-filter → rank → format**:

**Gather (`_gather_bucket`).** Fill a de-duplicated bucket of up to 12 enriched candidates, in priority order:
1. **Similar tracks** to the current track (Last.fm `track.getSimilar`) — the strongest continuity signal (up to 12).
2. **Similar artists'** top tracks (up to 4 artists, one track each).
3. **Genre/mood tags** — seeded from the current track's own top tags, falling back to tags derived from the state vector (`_state_to_tags`: e.g. peak→*anthemic*, energy>0.65→*energetic*, valence<0.35→*melancholic*).

Each candidate is resolved to a Spotify track and enriched (tags → energy/valence, Soundcharts → BPM/key, `is_saved` → familiarity). Already-played and currently-playing tracks are skipped.

**Hard filter (`_compatible_pool`).** Keep only candidates whose Camelot key is compatible with the current track (strength ≥ 0.4) **and** whose BPM is within the arc's tolerance window. Unknown key/BPM values pass. If nothing passes, the filter soft-degrades and returns the unfiltered bucket so the agent never receives an empty list.

**Rank (`_rank_candidates`).** Sort best-first by a weighted composite (`RANK_WEIGHTS`):

| Component | Weight | What it measures |
|-----------|--------|------------------|
| `ev` | 0.35 | Energy/valence fit vs the listener-state targets |
| `harmonic` | 0.25 | Camelot compatibility strength vs the current track |
| `bpm` | 0.15 | Tempo closeness vs the current track (normalised by phase threshold) |
| `arc` | 0.15 | Alignment with the arc phase's target energy band |
| `familiarity` | 0.10 | Openness-aware: when `openness ≥ 0.5` novelty scores higher, else familiarity does |

The arc-phase target energy bands (`_ARC_TARGET`) are warmup (0.3–0.5), build (0.5–0.7), peak (0.8–1.0), cooldown (0.2–0.4). Unknown key/BPM contribute a neutral 0.5 so missing data does not distort the order.

**Format.** Return the top `limit` candidates, each with `name, artist, key, bpm, energy, valence, harmonic_score, tags`, plus a `from` descriptor of the current track. The same `_rank_candidates` function backs the loop's safety-net fallback, so the autonomous and fallback paths select consistently.

---

## 7. End-to-end runtime flows

### 7.1 Session start

```
User types a description in the Streamlit start panel
  → bridge.start_session_from_description()
     → loop.start_session(description)
        1. _interpret_description()  — 1 Groq call → vibe JSON (state floats, queries, label)
        2. reset_session() + init_state_from_values(vibe)
        3. _react_pick_opener()  — opener graph: search → inspect → select_opening_track
             (anti-hallucination guard validates against seen_tracks)
        4. resolve choice → SpotifyTrack;  _spotify.play()
        5. background-thread pre-warm the sentence-transformer
        6. _record_played_track()  → updates history, guard sets, advances arc
     → bridge stores label/explanation/trace, refresh(), ensure_buffer(2)
```

If the opener cannot commit a real track, the flow falls back to a random pick from the user's Spotify top tracks.

### 7.2 Mid-session cycle (feedback or buffer top-up)

```
User clicks Skip / Thumbs-up / etc.  (or buffer runs low)
  → bridge.handle_feedback(event)  /  bridge.ensure_buffer()
     - immediate Spotify action (skip / seek-to-start / save-track) if applicable
     → loop.run_agent_cycle(feedback_event, track, artist)
        0. update_listener_state()  in Python  (step-0 trace entry)
        1. cycle graph runs:
             agent → get_ranked_candidates()  [gather → filter → rank]
                   → (optional) check_transition / get_compatible_keys
                   → add_track_to_queue(name, artist, reason)   [TERMINAL]
        2. on commit: reason → explanation;  track queued on Spotify,
           recorded in history, arc advanced
        3. fallback: if no commit, Python queues the #1 ranked candidate
     → bridge refresh(); store trace + explanation; st.rerun()
```

### 7.3 Buffer management and track-change detection

The bridge keeps **2 tracks buffered ahead** in Spotify's real queue (`ensure_buffer`). A `_lookahead` list of queued-but-not-yet-playing track IDs is reconciled against actual playback: `detect_and_handle_track_change()` polls current playback, and when the track ID changes it calls `consume_lookahead_up_to(new_id)`, which also handles out-of-band skips (the user skipping directly in Spotify, possibly past several buffered tracks).

---

## 8. The agent trace — the dissertation's evidence substrate

Every graph turn logs structured trace entries of four kinds:

- **`think`** — the model's reasoning, read from `reasoning_content`.
- **`act`** — one entry per tool call, with tool name, arguments, and result.
- **`observe`** — (reserved) tool observations.
- **`explain`** — the final natural-language explanation shown to the listener.

This makes the trace a faithful, replayable record of the agent's behaviour. The loop's module docstring explicitly frames it as **"the substrate the dissertation's failure taxonomy (state lag, novelty miscalibration, arc rigidity, key tunnel vision) is built on."** The Streamlit Agent Trace tab renders these entries (collapsing `explain` into `think` for display via `adapt_trace`).

---

## 9. External integrations and their constraints

| Service | Role | Key constraint |
|---------|------|----------------|
| **Spotify** (Spotipy, PKCE) | Search, playback control, library/familiarity, top tracks, playlist search | `audio-features` & `recommendations` deprecated (403/404) — never called. `popularity` always 0 in Development Mode. Playback control needs Premium. Playlist *track reads* go via RapidAPI because the official endpoint 403s on non-owned playlists. |
| **Last.fm** (pylast) | Tags (→ energy/valence), similar tracks, similar artists, artist/tag top tracks | Disk-cached in `.cache_lastfm/` (SHA1-keyed JSON). |
| **Soundcharts** | BPM, key, mode → Camelot position | Disk-cached in `.cache_soundcharts/`. Falls back Spotify-ID → ISRC lookup. |
| **Groq** (`ChatGroq`, gpt-oss-120b) | The reasoning LLM | Reasoning on `reasoning_content`; harmony-token leakage handled; backoff via `max_retries=4`. |

---

## 10. Known constraints and limitations (for honest writeup)

- **Single session, not thread-safe.** All session state is module-global in `tools.py`.
- **Python 3.11 required** (not 3.13) for LangGraph compatibility.
- **No native audio features.** Energy/valence are *estimated* from crowdsourced tags; key/BPM depend on Soundcharts coverage. Where data is missing, the pipeline degrades to neutral defaults — which can mask the harmonic/tempo reasoning the system is meant to demonstrate.
- **Popularity signal unavailable** from Spotify (always 0); Last.fm listener counts substitute as a familiarity/popularity proxy.
- **LLM brittleness on Groq** (malformed tool calls, channel-token leakage) is handled defensively but is a real source of cycle-level variance worth quantifying.

---

## 11. Quick file map

| File | Responsibility |
|------|----------------|
| `agent/state.py` | `ListenerState`, feedback weights, `update_state`, `advance_track`, arc transitions, presets. |
| `agent/tools.py` | 15 plain tool functions + the `get_ranked_candidates` selection pipeline; holds all session state as globals. |
| `agent/lc_tools.py` | LangChain `@tool` wrappers — the single source of truth the LLM sees; `ALL_TOOLS` / `CYCLE_TOOLS` / `OPENER_TOOLS`. |
| `agent/graph.py` | LangGraph ReAct graphs (`build_dj_graph`, `build_opener_graph`), nodes, routers, robustness handling, `run_graph`. |
| `agent/loop.py` | `run_agent_cycle`, `start_session`, system prompts, vibe extraction, fallbacks. |
| `music/camelot.py` | Camelot Wheel: mappings, compatibility rules, strength scoring. |
| `music/tags.py` | Tag lexicon, feature estimation, semantic fallback. |
| `music/lastfm_client.py` | Last.fm enrichment, similar tracks/artists, tag/artist top tracks (cached). |
| `music/soundcharts_client.py` | BPM / key / mode lookup → Camelot (cached). |
| `spotify/client.py` | Spotipy wrapper: search, playback, queue, library, playlists. |
| `app/app.py`, `app/bridge.py`, `app/components/*` | Streamlit UI, session-state plumbing, feedback dispatch, buffer management. |
