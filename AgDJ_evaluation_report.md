# AgDJ — Tutor Evaluation Report

*Cross-reference of `plan.md` and `plan_natural_language_start.md` against the actual codebase.*

---

## 0. Scope note — what was compared

The dissertation (`agentic_dj_dissertation.docx`, "Master Dissertation — Documentation", dated March 2026) has now been read in full and is treated here as the high-level reference. It is explicitly a **"living specification"**: its own closing note says Sections 1–2 are stable, Section 3 (Proposed Solution) "will be updated as implementation decisions are finalised", and Section 4 (Next Steps) is updated weekly. So some drift is expected — but the gap between the **written architecture** and the **built system** is now large enough that it will cost marks if the prose is submitted as-is. **Section 4 below is the core of this report.**

Also note: the dissertation predates the four `plan.md` features and the NL-start plan. None of Playback Actions, Proactive Queue Buffer, Track Change Detection, Out-of-Band Skip Detection, or Natural Language Start appear anywhere in the dissertation text. They are real in the code (sections 2–3 above/below) but invisible in the thesis (section 5).

---

## 1. Verdict summary

| # | Planned feature | Verdict | Evidence |
|---|-----------------|---------|----------|
| 1 | Playback Actions on button press | **Fully implemented** (one semantic mismatch) | `bridge.handle_feedback`, `SpotifyClient.skip/seek_to_beginning/save_track` |
| 2 | Proactive Queue Buffer (2-track lookahead) | **Mostly implemented** | `bridge.ensure_buffer`, `tools._lookahead`, `add_track_to_queue` |
| 3 | Out-of-Band Skip Detection (queue reconciliation) | **Partially implemented — wrong mechanism** | `get_spotify_queue()` exists but is **never called**; substituted by track-change detection |
| 4 | Track Change Detection | **Implemented (with caveats)** | `app.py` `@st.fragment(run_every=5)`, `bridge.detect_and_handle_track_change` |
| NL | Natural Language Session Start | **Substantially implemented** (UI feedback gaps) | `loop.start_session`, `components/start_session.py`, `bridge.start_session_from_description` |

---

## 2. Feature-by-feature analysis (`plan.md`)

### Feature 1 — Playback Actions ✅ (with one mismatch)

Each Now Playing button dispatches a real Spotify action plus the state/agent logic, exactly as the plan requires. In `app/components/now_playing.py` the four buttons emit `skip`, `replay`, `thumbs_up`, `full_listen`, and `bridge.handle_feedback` maps them:

- **Skip** → `_spotify.skip()` then `run_agent_cycle(feedback_event="skip")` → refills buffer. ✅
- **Replay** → `_spotify.seek_to_beginning()`, state update only, **no** agent cycle and **no** forward skip. ✅ (matches "Do not skip forward").
- **Love** → `_spotify.save_track(track_id)` then agent cycle. ✅
- **Next** → `_spotify.skip()` then agent cycle. ✅ playback-wise.

**Mismatch to fix:** the plan specifies **Next = a *neutral* signal**. In code the "Next →" button emits the event `full_listen`, which `state.update_state` treats as a *positive* completion signal (listened to >85%), not neutral. So pressing Next nudges the listener model in the wrong direction relative to the approved design. Either add a `neutral`/`partial_listen` mapping for Next, or justify in the dissertation why "Next" is modelled as a full listen. Right now there is no `neutral` `FeedbackEvent` at all (`state.py` enum: skip, early_skip, replay, full_listen, partial_listen, thumbs_up, thumbs_down).

Minor: `now_playing.py` still ships hard-coded `MOCK` data and `app.py` contains a leftover `print(st.session_state)` and a `# TODO: remove print` — clean these up before submission.

### Feature 2 — Proactive Queue Buffer ✅ (mostly)

- A local record is kept (`tools._lookahead: list[str]` of queued-ahead IDs, plus `_queue`), satisfying "don't fully trust Spotify's queue." ✅
- `add_track_to_queue` appends to `_lookahead` on success. ✅
- `bridge.ensure_buffer(2)` runs cycles until `get_lookahead_depth() >= 2`. ✅
- Called at session start (`start_session_from_description` → `ensure_buffer(2)`) and after a detected track change (`now_playing_fragment` → `ensure_buffer(2)`). ✅

**Caveats:**
- `ensure_buffer` caps iterations with `attempts < target` (i.e. ≤ 2 cycles). If a cycle *succeeds at the duplicate guard but fails to queue*, it `break`s and the buffer can be left at depth 1. There is no retry-with-different-candidate beyond the fallback scorer. Acceptable, but worth a sentence in the write-up.
- The buffer is only ever refilled to 2 from two trigger points (start, and track-change in the polling fragment). There is no refill when the *Queue* or *other* tabs are open and the Now Playing fragment is not running (see Feature 4 caveat).

### Feature 3 — Out-of-Band Skip Detection ⚠️ implemented by a *different* mechanism

This is the biggest divergence from the approved plan. The plan is explicit:

> "The application should **periodically read the actual Spotify queue** using the available API endpoint … compare what Spotify reports as queued against the application's internal record … if a discrepancy is detected … the internal record should be corrected."

The code has the building block — `SpotifyClient.get_spotify_queue()` (calls `sp.queue()`) — **but it is never called anywhere in `app/` or `src/`** (verified by grep; the only reference is its own definition). There is no queue-diffing/reconciliation logic.

Instead, out-of-band skips are handled *incidentally* by **Feature 4's** track-change detector: `detect_and_handle_track_change()` compares the current track ID to the previous one and calls `consume_lookahead_up_to(new_id)`, which the docstring claims "handles out-of-band skips too." This covers the common case (user skips forward on their phone → current track changes → lookahead is consumed correctly, even multi-step). 

**But it does NOT cover the cases the plan's queue-reconciliation was designed for:**
- The user **re-orders or removes** a queued track without the *currently playing* track changing → internal `_lookahead` is now wrong and nothing detects it.
- The user skips to a track **not in** `_lookahead` → `consume_lookahead_up_to` returns 0 and frees nothing, so `_lookahead` keeps stale IDs while the buffer count is misreported.
- Spotify silently drops/finishes a queued item.

**Verdict: partially implemented.** Either (a) implement the planned `get_spotify_queue()` reconciliation and call it on the polling interval, or (b) explicitly re-scope the dissertation to say out-of-band handling is done via current-track diffing and document the un-handled cases as limitations. Option (a) is closer to the approved plan and is a small amount of work since the API wrapper already exists.

### Feature 4 — Track Change Detection ✅ (caveats)

Implemented as a Streamlit polling fragment, not a true background thread:

```python
@st.fragment(run_every=5)
def now_playing_fragment():
    changed = bridge.detect_and_handle_track_change()
    if changed:
        bridge.ensure_buffer(2)
```

`detect_and_handle_track_change` refreshes playback, compares track IDs, consumes the lookahead slot, returns `True`, and the buffer is refilled. This meets "detect natural advance, update internal state, trigger a replacement," and runs without listener input. ✅

**Caveats to disclose in the write-up:**
- It is **not a background process** in the OS sense. The fragment only fires while the app is open **and the "Now playing" tab is the active tab**. If the user sits on the Queue/State/Trace tab, no detection or refill happens. The plan's language ("a background process should continuously monitor") implies tab-independence.
- 5-second poll granularity means up to ~5 s of latency on a natural advance; with a 2-track buffer this is fine, but state it.
- The finished track is removed from the lookahead record via `consume_lookahead_up_to`, but it is **not** removed from the `_queue` list (which only ever grows / is trimmed to last 5 in `get_queue_state`). The "Queue" tab therefore shows recently-queued tracks, not a true forward queue. Cosmetic but worth knowing.

---

## 3. Natural Language Session Start (`plan_natural_language_start.md`)

**Substantially implemented** and genuinely well-built. Coverage:

- **Idle screen** — `components/start_session.py` renders the centred input, the inviting label ("What do you want to listen to?"), the example placeholder, and a primary submit button disabled until text is entered. ✅
- **Loading state** — handled in `app.py` via `st.spinner("Finding the right track for you…")` wrapping `start_session_from_description`. ✅ (matches the plan's note that the spinner covers the whole interaction).
- **Two-phase agent** — `loop.start_session`: (1) `_interpret_description` does the structured-JSON vibe extraction (energy/valence/focus/openness/social/search_queries/session_label/confident); (2) `_react_pick_opener` runs a focused ReAct sub-loop to commit an opener. ✅ This matches "extract signals → find a strong, recognisable opening track."
- **State seeding from the description** — `init_state_from_values(...)` seeds the 6-D vector from the inferred vibe rather than a neutral default. ✅ (the plan's "clubbing → high energy/valence" requirement).
- **Start playback (not just queue)** — `_spotify.play(sp_track)` actually starts playback, then `ensure_buffer(2)` fills the lookahead. ✅
- **Anti-hallucination guard** — `select_opening_track` rejects any track not seen in a real search result. Nice touch; worth highlighting in the dissertation as a reliability measure.
- **Session label on Now Playing** — `adapt_now_playing` passes `session_label`, and `now_playing.py` renders a "Session: …" badge. ✅

### Gaps / un-met requirements in NL start

1. **Buffer is NOT seeded from the opening track as the plan demands.** The plan is very specific: the first two buffer selections "must be seeded using the opening track as its musical anchor … using its BPM, key, energy and tags." In reality `ensure_buffer(2)` just calls `run_agent_cycle`, whose Python pre-step builds candidates from **`_state_to_tags(listener_state)`** — i.e. from the seeded *state vector*, not from the opening track's audio profile. The opening track *is* recorded in history (`_record_played_track`) and advances state, but the cycle never anchors on the opener's key/BPM specifically. **This is a real deviation from the approved plan** and should either be implemented or explicitly re-scoped.

2. **The "ambiguous description" fallback does not inform the user, contrary to the plan.** The plan says: if the description is meaningless, fall back to `me/top/tracks?time_range=short_term` *and* "the user should be informed that their description was unclear and that the session is starting from their personal listening history instead." The code does the fallback correctly (`get_top_tracks(limit=20)` defaults to `time_range="short_term"`, random choice, label set to "Your recent favourites"), **but no message is surfaced to the UI.** `start_session_from_description` only inspects `result["error"]`; it ignores `result["fallback_used"]`. So the user silently gets a random favourite with no explanation. Add a UI notice keyed off `fallback_used`.

3. **`fallback_used` is conflated between two very different cases.** It is set both when (a) the description was vague (`confident == False`) and (b) the agent was *confident* but the ReAct loop failed to commit an opener. In case (b) the label is kept as the vibe label and the user is told nothing. Worth separating these, and at minimum logging case (b) as an agent failure for your failure taxonomy.

### NL start edge cases — coverage check

| Edge case (from plan) | Handled? | Notes |
|---|---|---|
| No active Spotify device | ✅ partially | `play()` returns `False` on `NO_ACTIVE_DEVICE` → `start_session` returns `error: "no_device"` → bridge shows "Open Spotify on any device and try again." Good. **But** `play()` re-raises any *other* SpotifyException, and the no-device check happens *after* interpretation + a full ReAct search — wasteful, and a non-NO_ACTIVE_DEVICE error (e.g. expired token) will crash the fragment. |
| Ambiguous / meaningless / very short input | ⚠️ logic yes, UX no | Fallback to short-term top tracks works; user is **not** told (gap #2 above). Also "ambiguity" is decided solely by the LLM's `confident` boolean — no guard for empty/garbage strings before the Groq call (the submit button's `disabled` guard only blocks empty, not "asdf"). |
| Non-English description | ❌ not handled | Plan says treat as out-of-scope and **document as a known limitation**. There is no handling and (pending the missing docx) likely no written limitation. Add the sentence. |

---

## 4. Dissertation prose vs actual code — the mismatches that matter

The dissertation's Chapters 1–2 (problem, related work, research questions) are sound and need no code support. The problem is **Chapter 3 (Proposed Solution)** and parts of **Chapter 4**: they describe an architecture, a technology stack, and a tool catalogue that the implemented system does **not** use. Each row below is verified against source.

### 4.1 Claims written as fact but contradicted by the code

| Dissertation says (§) | Code actually does | Severity |
|---|---|---|
| **"orchestrated via LangGraph"**, "implemented using LangGraph, with each reasoning step structured as a node in a directed graph" (§3.1, §3.4, §3.7, Week 4) | ~~No LangGraph; hand-written `_run_react_loop`.~~ **RESOLVED** on branch `feature/ADJ-16-add-lang-graph`: both entry points now run compiled LangGraph `StateGraph`s (`agent/graph.py`), the old `_run_react_loop`/`_groq_call` are deleted, and tools are LangChain `@tool` objects (`agent/lc_tools.py`). The dissertation's LangGraph claim is now accurate. | ~~High~~ **Resolved** |
| **"GPT-4o (primary), Mistral-7B (comparison)"** (§3.7); "GPT-4o API calls may introduce latency" (Risk 3) | `MODEL = "openai/gpt-oss-120b"` via the **Groq** SDK (`loop.py:48`). Not GPT-4o, not Mistral. (Your own `CLAUDE.md` says Llama-3.3-70B and a memory note says Gemini — three different stories.) | **High** — examiner will check the model. |
| **"SQLite (session and trace logging)"** (§3.7) | No SQLite anywhere. Session state is **module-level globals** in `tools.py`; traces are in-memory Python lists discarded at session end. | **High** — affects the evaluation plan, which assumes logged traces exist for the failure taxonomy (§4.2 Part D). |
| Music-theory layer is built on **Spotify audio features**: "Every track … has a key (0-11) and a mode (0/1)"; tools `get_audio_features(track_id)`, `get_recommendations(...)` (§3.3, §3.5); "Spotify's audio features (energy, valence, danceability)" (§3.2) | Spotify `audio-features` and `recommendations` are **deprecated/403 for this app and explicitly never called** (your own `CLAUDE.md` constraint). Key + BPM come from **Soundcharts**; energy/valence are **estimated from Last.fm tags**. So the entire data-sourcing basis of the constraint layer described in Ch 3 is not how the system works. | **High** — the constraint layer is real, but sourced completely differently from what's written. |
| **Tool catalogue** (§3.5): `key_compatible(key, mode)`, `tempo_range(...)`, `energy_target(arc_phase)`, `search_tracks(energy, valence, tempo, key_filter, limit)`, `get_audio_features`, `get_recommendations`, `add_to_queue(track_id, position)`, `reorder_queue(new_order)`, `update_state(dimension, delta)` | Almost none of these exist by name or signature. Actual tools: `get_compatible_keys`, `check_transition`, `estimate_bpm_compatibility`, `search_tracks(query, limit, enrich)`, `search_tracks_by_tag`, `search_artist_tracks`, `search_tracks_by_playlist`, `get_track_details`, `get_current_playback`, `get_session_history`, `add_track_to_queue`, `update_listener_state(event, …)`. No `reorder_queue`, no `get_recommendations`, no `get_audio_features`, no `energy_target`, no positional queue insert. `search_tracks` takes a **text query**, not a feature vector. | **High** — §3.5 must be rewritten from the real `tools.py`. |
| **Hard pre-filters** "applied to filter the search space … before LLM evaluation" (§3.3) | Harmonic/tempo checks are **LLM-callable soft tools** (`check_transition`, `estimate_bpm_compatibility`), not deterministic pre-filters. The only true pre-filter is `search_tracks` dropping candidates missing *both* BPM and key. Energy-arc range filtering (below) isn't implemented at all. | **Medium** — reframe as soft, agent-invoked checks. |
| **Energy arc alignment**: target bands warmup 0.3–0.5, build 0.5–0.7, peak 0.8–1.0, cooldown 0.2–0.4; deprioritise candidates >0.15 outside band (§3.3) | **Not implemented.** There is no `energy_target` function and no arc-band filter. The mid-session pre-fetch scores candidates by distance to the *current state* energy/valence (`_best_fallback_top_n`), not to an arc target band. | **Medium** — written as a constraint that doesn't run. |
| **Arc** "updated by elapsed time and cumulative energy trajectory" (§3.2, §3.3) | Arc advances purely by **track count** (`ARC_TRANSITIONS` = 4 warmup → 8 build → 4 peak → cooldown). No time or energy-trajectory input. | **Medium.** |
| **SOCIAL** dimension "updated slowly from session duration and time of day signals" (§3.2) | SOCIAL is set at init and **never updated** — there is no `social` weight in `WEIGHTS` and no time-of-day signal anywhere. | **Medium** — dimension is effectively static. |
| State update **magnitudes** (§3.2): "skip of fast track −0.12", "replay of fast track +0.06", "full listen of calm track −0.04" | Actual `WEIGHTS`: SKIP energy ∓0.10 (EARLY_SKIP ∓0.15), REPLAY energy ±0.03, FULL_LISTEN energy ±0.06. The illustrative numbers in the thesis **do not match the code**. | **Medium** — regenerate the numbers from the `WEIGHTS` table. |
| Explanation "**verified post-generation**" references a signal + musical property (§3.6) | The constraint is in the prompt (`CYCLE_SYSTEM_PROMPT`) only; there is **no post-generation verification step** that checks the explanation actually cites a signal and a musical property. | **Low** — drop "verified" or implement the check. |
| Reasoning loop "triggered **30 seconds before each track ends**" so the next pick is ready (Risk 3) | Not done. Selection is reactive via a **5-second polling fragment** plus a 2-track proactive buffer (a different, arguably better, mechanism — but not what's written, and the buffer isn't in the thesis at all). | **Low/medium.** |

### 4.2 Claims that ARE supported by the code (keep these)

- The **6-dimensional state vector** (energy, valence, focus, openness, social, arc) — names and concept match `state.py`. ✅
- **Direction-aware update rule** — the headline novel contribution. `update_state` flips the sign of the energy/valence delta based on the *track's own* profile (`direction = 1 if track.energy_est >= 0.5 else -1`). ✅ This is the thesis's strongest claim and it holds.
- **Camelot Wheel harmonic compatibility** with same/adjacent/relative positions and a 0–1 strength score — `music/camelot.py`. ✅ (data source differs, see 4.1).
- **Tempo thresholds** 8 BPM steady / 25 BPM transition — `estimate_bpm_compatibility` matches §3.3 exactly. ✅
- **ReAct reason→act→observe trace** with a `think` entry per reasoning turn and an `act` entry per tool call — `_run_react_loop`. ✅ (just not via LangGraph).
- **Exponential backoff** for rate limiting (Risk 1) — `_groq_call`. ✅
- **Streamlit + Plotly** frontend with state visualisation and agent-trace panel — `app/`. ✅
- **music21** is a declared dependency and the Camelot logic exists. ✅ (verify music21 is actually used vs the custom wheel before claiming it.)

### 4.3 How to fix Chapter 3/4 (concrete)

1. Rewrite §3.1/§3.4/§3.7 to describe the **Groq + native function-calling** loop, not LangGraph. Either remove `langchain`/`langgraph` from the dependency files or justify their presence.
2. Name the **actual model** (`openai/gpt-oss-120b` on Groq) consistently across thesis, `CLAUDE.md`, and memory. If GPT-4o/Mistral was a *planned* comparison never run, move it to Future Work.
3. Replace the §3.5 tool catalogue with the **real 12–15 tools** from `tools.py` (group them as you already do: state / music-theory / Spotify+Last.fm / queue).
4. Rewrite §3.3 data sourcing: key + BPM from **Soundcharts**, energy/valence from **Last.fm tag estimation with a sentence-transformer fallback**, and state plainly that Spotify audio-features/recommendations are unavailable (this is a legitimate, interesting constraint — feature it, don't hide it).
5. Either implement **energy-arc band filtering** and **time/energy-based arc advance**, or change the prose to say arc advances by track count and energy fit is a scoring term, not a hard band.
6. Regenerate the §3.2 **update magnitudes** from the `WEIGHTS` table, and either implement SOCIAL updates or describe it honestly as context-initialised and static.
7. Address the **trace-logging gap**: the failure taxonomy (§4.2 Part D) needs persisted traces. Add SQLite/JSON trace logging, or the evaluation as written cannot be run.

---

## 5. Code implemented but absent from the dissertation

The dissertation describes *less* than what you built in several places, and *different* things in others. The following are real, working, and **completely missing from the thesis text** — they are substantial enough that omitting them undersells the project:

0. **The entire `plan.md` feature set and NL-start.** Playback Actions, the Proactive 2-track Queue Buffer, Track Change Detection, Out-of-Band Skip handling, and Natural Language Session Start are all implemented (sections 2–3) but appear nowhere in the dissertation. Chapter 5 ("Song selection & queue management") and Chapter 4 should absorb them. The NL-start two-phase agent (structured vibe extraction → focused opener sub-loop with an anti-hallucination guard) is a genuine contribution worth its own subsection.

Additionally, these supporting components deserve coverage:

1. **Soundcharts integration** (`music/soundcharts_client.py`) — a whole external data source for BPM + Camelot key, disk-cached, replacing the Deezer/GetSongBPM lineage. This is methodologically significant (it's how harmonic mixing is even possible) yet absent from both plans. Discuss it and its rate/coverage limits.
2. **The Python pre-fetch + shortlist scoring** in `run_agent_cycle` (`_state_to_tags`, `_best_fallback_top_n`). Mid-session selection is **not** a pure ReAct loop — Python pre-scores 3 candidates and the LLM only verifies harmonic fit (`CYCLE_TOOLS` strips all search tools). This is a defensible design (saves rate-limited calls), but it qualifies the "the LLM autonomously orchestrates all tool calls" narrative. Be precise about where Python decides vs where the LLM decides.
3. **Two scored-fallback safety nets** (`_best_fallback`, plus the in-loop fallback in `run_agent_cycle`). The system frequently may *not* be the LLM's choice at all. For a thesis whose evaluation rests on the agent's reasoning, you must report how often the fallback fires.
4. **Semantic tag fallback** (sentence-transformers, prewarmed on a daemon thread in `start_session`). Good engineering, undocumented in the plans.
5. **Anti-hallucination `select_opening_track` guard** — a genuine reliability contribution; name it.

---

## 6. Bugs / dead code to fix before evaluation

1. **Dead penalty logic.** `_best_fallback` and its log read `c.get("bpm_ok", True)` / `c.get("camelot_ok", True)`, but **no code ever sets these keys** on a candidate (verified by grep). So the BPM/Camelot penalties in the fallback scorer are inert — the fallback ranks on energy+valence only. Either populate `bpm_ok`/`camelot_ok` in `_spotify_to_candidate`, or remove the dead branches so the thesis doesn't describe a scorer that doesn't run.
2. **`get_spotify_queue()` is unused** (Feature 3). Either wire it up or delete it.
3. **Leftover debug output** — `app.py` `print(st.session_state)` + `# TODO`; `spotify/client.py` `print(query)` in `search`. Remove before any user study/demo.
4. **`now_playing.py` MOCK / `queue.py` MOCK** defaults — fine for dev, but ensure they never render in the study build.
5. **`start_session` does device check last.** Move a `get_playback()`/device check *before* the expensive interpret+search so a no-device user fails fast (and so non-NO_ACTIVE_DEVICE errors don't bubble into the polling fragment uncaught).
6. **Documentation drift** (`CLAUDE.md` vs `memory/*` vs code) on model name, BPM source, and backoff timing. Reconcile all three; an examiner reading your repo will see the contradiction.

---

## 7. Prioritised action list

**Code — must do before evaluation**
1. Decide Feature 3's fate: implement real queue reconciliation via `get_spotify_queue()` on the poll, **or** re-scope and document. (Highest plan-divergence.)
2. Seed the NL-start buffer from the **opening track's** key/BPM/tags, as the plan requires — or re-scope it explicitly.
3. Fix the **Next → `full_listen`** semantic mismatch (add a neutral signal).
4. Surface the **ambiguous-description fallback** message to the user.
5. Remove dead `bpm_ok`/`camelot_ok` branches (or make them live).
6. Strip debug prints and reconcile model name / BPM-source docs.

**Code — should do**
7. Make track-change detection tab-independent (or document the limitation honestly).
8. Fast-fail the no-device case at the top of `start_session`; handle non-NO_ACTIVE_DEVICE exceptions.
9. Guard garbage/very-short input before the Groq call.

**Dissertation — must do (Chapter 3 is the priority)**
10. Rewrite **§3.1/§3.4/§3.7**: it's **Groq native function-calling, not LangGraph**; the model is **`openai/gpt-oss-120b`, not GPT-4o/Mistral**; there is **no SQLite** (and the failure taxonomy needs trace logging added — see #16).
11. Rewrite the **§3.5 tool catalogue** from the real `tools.py`, and the **§3.3 data sourcing** (Soundcharts for key/BPM, Last.fm for energy/valence; Spotify audio-features/recommendations unavailable).
12. Fix the §3.2 **update magnitudes** to match `WEIGHTS`; correct the **arc** description (track-count, not time/energy); describe **SOCIAL** as static or implement its updates; drop or implement "energy-arc band" and "post-generation verification".
13. Add sections for the **undocumented-but-implemented** features and components (section 5): the four `plan.md` features, NL start, Python pre-fetch shortlist, fallback scorers, semantic tag fallback, anti-hallucination guard.
14. State honestly **where the LLM decides vs where Python decides**, and plan to **report fallback-firing frequency** in evaluation.
15. Add the **non-English = out-of-scope** limitation and the track-change polling limitations.
16. **Trace logging:** the evaluation (§4.2 Part D failure taxonomy, and the regression on "trace features") assumes persisted traces. Implement JSON/SQLite trace logging or the evaluation as written cannot be executed.

---

*Bottom line: the engineering is in good shape — the core system, all three substantive `plan.md` features, and the NL-start feature are real and working, and the headline contribution (direction-aware state updates) is genuinely implemented. The risk to your grade is almost entirely in the **written dissertation, Chapter 3**. As submitted it describes a LangGraph/GPT-4o/SQLite/Spotify-audio-features system that you did not build — you built a leaner, arguably more interesting Groq/Soundcharts/Last.fm system, and you built five features (playback, buffer, track-change, out-of-band, NL start) the thesis never mentions. An examiner who reads Chapter 3 and then opens the repo will see the mismatch immediately. Realign Chapter 3/4 with the real architecture, document the hidden machinery, fix the trace-logging gap so your evaluation can actually run, and this becomes a strong submission.*
