# Plan — Improve Candidate Selection (single discovery+ranking tool)

*Draft for review. No implementation yet. Same branch (`feature/ADJ-16-add-lang-graph`).*

---

## Goal

Replace the hidden Python pre-fetch with **one LLM-callable tool** that gathers candidates from several endpoints into a bucket (~20), hard-filters for harmonic/tempo compatibility, ranks them by a weighted composite (`RANK_WEIGHTS`), and returns the **top 5** for the model to choose from. This unifies the two earlier ideas (#1 similarity discovery, #2 composite ranking) into a single, more agentic step.

## Design shift

**Today:** `run_agent_cycle` runs a Python pre-fetch (tag search → energy/valence rank → top 3), injects that shortlist into the prompt, and the LLM (no search tools) verifies + commits.

**Proposed:** the LLM calls a new tool `get_ranked_candidates()`, gets the top 5 already ranked, optionally verifies one with `check_transition`, and commits with `add_track_to_queue`. The Python pre-fetch is removed; the same ranking function is reused as the safety-net fallback.

```
LLM cycle:
  get_listener_state / get_current_playback   (orient)
  get_ranked_candidates()                      ← NEW: bucket → filter → rank → top 5
  check_transition(...)                        (optional confirm)
  add_track_to_queue(name, artist)             (commit)
```

---

## The new tool — `get_ranked_candidates(limit=5)`

Self-contained: it reads the currently-playing track and listener state internally (no args the LLM has to compute). Steps:

1. **Identify the current track** — current key, BPM, tags, artist (via the existing `_current_track_profile` logic).
2. **Fill the bucket (~20 identities, before enrichment), de-duped by Spotify id, skipping already-played and the current track:**
   - **Similar tracks** to the current track — Last.fm `track.getSimilar` (primary).
   - **Similar artists'** top tracks — Last.fm `artist.getSimilar` → each artist's top track(s).
   - **Tag-based** — current track's top genre tags (existing `search_tracks_by_tag`) to top up to ~20.
   - Spotify text fallback only if all of the above resolve nothing.
3. **Enrich the ~20** (Last.fm energy/valence + Soundcharts key/BPM) — cached, so repeats are free.
4. **Hard compatibility filter** vs the current track (harmonic strength ≥ 0.4 AND BPM within the arc window; unknowns pass; never returns empty — soft-degrade to unfiltered if nothing passes). Same `_compatible_pool` logic.
5. **Composite rank** (below) and **return the top `limit` (default 5)**, each with name, artist, key, BPM, energy, valence, tags, and the harmonic score vs the current track — enough for the LLM to choose and to write a grounded explanation.

Returned shape (example):
```json
{
  "from": {"key": "10B", "bpm": 176},
  "count": 5,
  "candidates": [
    {"name": "...", "artist": "...", "key": "11B", "bpm": 176, "energy": 0.60,
     "valence": 0.58, "harmonic_score": 0.85, "tags": ["reggaeton","latin"]}
  ]
}
```

Bucket size (~20) and return size (5) are named constants.

## Composite ranking (`RANK_WEIGHTS`)

Shared Python function `_rank_candidates(candidates, state, current_key, current_bpm)` used by **both** the tool and the fallback. Higher = better:

```
RANK_WEIGHTS = {"ev": 0.35, "harmonic": 0.25, "bpm": 0.15, "arc": 0.15, "familiarity": 0.10}

score(c) =  w_ev   * ev_fit       # 1 - (|e-tgt_e| + |v-tgt_v|)/2 vs listener state
          + w_harm * harmonic     # compatibility_strength(cur_key, c.key); unknown → 0.5
          + w_bpm  * bpm_close     # 1 - min(|Δbpm|/threshold, 1);          unknown → 0.5
          + w_arc  * arc_fit       # closeness of c.energy to the arc target band
          + w_fam  * fam_fit       # openness-aware: high openness rewards unfamiliar, low rewards familiar
```

- `arc_fit` uses the dissertation's arc bands (warmup 0.3–0.5 … peak 0.8–1.0) via a new `_arc_energy_target(arc_phase)`.
- `fam_fit` uses the already-computed `familiar` flag + `listeners`.
- Unknown key/BPM contribute a neutral 0.5 so missing Soundcharts data doesn't distort the order.
- Delivers, as a side effect, the long-missing use of the **arc** and **openness** dimensions in selection.

---

## Architecture / files

| File | Change |
|---|---|
| `music/lastfm_client.py` | + `get_similar_tracks(artist, track, limit)` (pylast `Track.get_similar`, disk-cached) |
| `agent/tools.py` | + `get_ranked_candidates(limit=5)` (bucket → enrich → filter → rank); reuses `_resolve`/`_spotify_to_candidate` caches; + internal `_gather_bucket()` helper |
| `agent/loop.py` | remove the Python pre-fetch from `run_agent_cycle`; `RANK_WEIGHTS` + `_arc_energy_target` + `_rank_candidates`; `_best_fallback` reuses `_rank_candidates`; update `CYCLE_SYSTEM_PROMPT` |
| `agent/lc_tools.py` | + `get_ranked_candidates` tool; `CYCLE_TOOLS` now = state/theory tools + `get_ranked_candidates` + `get_track_details` + `add_track_to_queue` (drop the four granular search tools from the cycle set — the new tool subsumes them; they stay for the opener) |
| `tests/` | new unit tests |

`run_agent_cycle` keeps: step-0 feedback application, the graph run, and the scored fallback (now via `_rank_candidates`). It loses the tag-search/shortlist pre-step and the candidate-section prompt injection.

## Tests (offline, mocked)

- `get_similar_tracks`: mock pylast → parse + cache.
- `_rank_candidates`: harmonic strength changes order; arc target shifts the pick per phase; openness flips familiar↔novel; unknown key/bpm neutral; deterministic.
- `get_ranked_candidates`: mock the three sources → bucket dedups by id, caps at ~20, excludes already-played + current, returns ≤5 ranked, never empty.
- Graph integration: a scripted LLM that calls `get_ranked_candidates` then `add_track_to_queue` queues the expected track (extend `test_graph.py`).
- Existing 61 tests stay green (the removed pre-step tests get retargeted to `_rank_candidates`).

## Trade-offs / risks

- **One extra LLM round-trip** (the tool call) vs the old injected shortlist. Acceptable, and the gpt-oss tool-call retry/sanitizer already handle the reliability risk. The old pre-step existed to save round-trips under tight Gemini limits — no longer the binding constraint.
- **Enriching ~20 per cycle** is the main cost; mitigated by disk + session caches (cold start is the only slow case).
- **Hand-set weights** — kept as named constants; calibration is the later SQLite-backed pass.
- Compatibility filter still runs *before* ranking, so harmonic/tempo safety is unchanged.

## Open questions for you

1. **`RANK_WEIGHTS` starting split** — keep `ev .35 / harmonic .25 / bpm .15 / arc .15 / fam .10`, or push harmonic/arc higher to emphasise mixing and the arc?
2. **Bucket source split** — for the ~20: roughly how much from similar-tracks vs similar-artists vs tags? *Lean: ~10 similar-tracks / ~5 similar-artists / fill with tags.*
3. **Drop the four granular search tools from `CYCLE_TOOLS`** (keep only for the opener), or leave them available to the cycle as escape hatches? *Lean: drop from cycle.*
4. Confirm **bucket ≈ 20, return = 5**.
