"""
ReAct agent orchestration — the reasoning core of the Agentic DJ system.

Both entry points (run_agent_cycle, start_session) are implemented as compiled
LangGraph ReAct graphs (see graph.py) over ChatGroq's native function-calling.
The model is given a goal and a toolbox; it decides which tools to call, in what
order, and when it has gathered enough to commit. No procedural rules.

Every graph turn logs:
  • a `think` trace entry  (the model's reasoning, read from reasoning_content)
  • one `act`   trace entry per tool call  (tool name, arguments, result)
  • an `explain` entry for the final natural-language explanation

This makes the trace a faithful record of the agent's behaviour — the
substrate the dissertation's failure taxonomy (state lag, novelty
miscalibration, arc rigidity, key tunnel vision) is built on.

Architecture:
  - Graphs are built by graph.build_dj_graph / build_opener_graph and driven by
    graph.run_graph. The tools are LangChain tool objects from lc_tools.
  - run_agent_cycle uses a Python pre-fetch step to build a scored candidate
    shortlist, then runs the cycle graph (lc_tools.CYCLE_TOOLS — no search) so
    the LLM only verifies harmonic fit and commits. Falls back to the full
    toolset (lc_tools.ALL_TOOLS) graph if the pre-fetch returns nothing, and to
    a scored fallback if the graph still does not commit.
  - start_session runs the opener graph (lc_tools.OPENER_TOOLS) which stops on
    the synthetic select_opening_track tool; its anti-hallucination guard lives
    in the opener graph node (validates against seen_tracks in graph state).
  - A structured-JSON pre-step in start_session extracts the vibe vector from
    the description via a single ChatGroq call (no tools).
  - Rate-limit backoff is delegated to ChatGroq(max_retries=4); there is no
    custom retry layer here.
"""

import json
import threading

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq

from agentic_dj.agent import tools as tool_module
from agentic_dj.agent import lc_tools
from agentic_dj.agent.graph import build_dj_graph, build_opener_graph, run_graph
from agentic_dj.agent.state import init_state_from_values
from agentic_dj.music.camelot import parse as camelot_parse, compatibility_strength
from agentic_dj.music.tags import prewarm_embedding_model

# BPM jump tolerance per arc phase (mirrors estimate_bpm_compatibility in tools.py):
# tight in steady-state phases, wide where energy is intentionally shifting.
_BPM_THRESHOLDS = {"warmup": 8.0, "build": 25.0, "peak": 8.0, "cooldown": 25.0}
# A harmonic transition scoring ≥ this is "acceptable" (matches check_transition).
_HARMONIC_MIN = 0.4

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()
MODEL  = "openai/gpt-oss-120b"

DISPLAY_LOGS: bool = True       # set False to silence the trace stream
MAX_ITERATIONS: int = 15        # safety cap per ReAct loop (graph recursion limit)

# Rate-limit backoff is handled natively by ChatGroq(max_retries=...) inside the
# compiled graphs (graph.py) — there is no custom retry layer here.

# Lazy ChatGroq used only for the structured-JSON vibe extraction in start_session
# (a single non-tool call). The graphs build their own ChatGroq instances.
_INTERPRET_LLM = None


def _get_interpret_llm():
    global _INTERPRET_LLM
    if _INTERPRET_LLM is None:
        _INTERPRET_LLM = ChatGroq(model=MODEL, temperature=0.3, max_retries=4)
    return _INTERPRET_LLM


# ══════════════════════════════════════════════════════════════════════════════
#  SYSTEM PROMPTS
# ══════════════════════════════════════════════════════════════════════════════

CYCLE_SYSTEM_PROMPT = """\
You are the reasoning core of an Agentic DJ.

Goal: pick the single best next track from the pre-scored candidates in the
user message, verify its harmonic fit, and queue it. Explain your choice in
one or two plain sentences.

Suggested tool call order (stay within this):
  1. get_listener_state + get_current_playback   — orient yourself (one call each)
  2. check_transition(from_camelot, to_camelot)  — verify harmonic fit for your top candidate
  3. get_track_details(track_name, artist)        — optional: fetch full profile if key is missing
  4. add_track_to_queue(track_name, artist)       — TERMINAL: call once, then stop

Candidates are pre-scored by energy/valence fit and listed in the user message.
Do NOT search for more — work from the shortlist. The only exception is if every
candidate in the list is already in the session history (already played), in
which case you may call get_session_history to confirm and then search once.

Commitment rule: if check_transition returns score ≥ 0.4 (verdict "acceptable"
or "smooth"), commit that candidate immediately. Do not seek a perfect score.
A single passing harmonic check is sufficient to act.

Rejection rule: only discard a candidate if check_transition returns score < 0.4
(verdict "avoid") AND you have a better-ranked alternative still available in
the shortlist. Never reject without trying the next candidate.

Ground every claim in actual tool output — never invent values. Do not queue
a track already played this session. In your final explanation reference at
least one concrete signal (arc phase or listener state dimension) and one
musical property (key, BPM, or energy)."""


SESSION_INTERPRET_SYSTEM = "Return only valid JSON. No prose, no markdown fences."

SESSION_INTERPRET_USER = """\
Interpret this listening session description and return ONLY a JSON object \
with these fields:

{{
  "energy":         <float 0.0-1.0, how energetic/intense>,
  "valence":        <float 0.0-1.0, how positive/happy vs dark/sad>,
  "focus":          <float 0.0-1.0, how focused vs party/social>,
  "openness":       <float 0.0-1.0, how open to variety>,
  "social":         <float 0.0-1.0, how social/group vs solo>,
  "search_queries": ["<3 short playlist-style strings, 2-4 words, e.g. '2016 pop hits', 'chill Sunday morning pop', 'late night R&B'>"],
  "session_label":  "<short evocative label, e.g. '2016 clubbing vibes'>",
  "confident":      <true if the description has clear musical signal, false if vague>
}}

Description: "{description}"
"""

SESSION_PICK_SYSTEM = """\
You are an Agentic DJ choosing the OPENING track for a new listening session.

Toolbox (focused subset):
  • search_tracks_by_playlist(query)      — preferred for vibe/era/mood queries
  • search_tracks(query, limit)           — use only for specific artist/track lookups
  • get_track_details(track_name, artist) — optional: full profile for one candidate
  • select_opening_track(track_name, artist) — terminal action, call once

Reason about which track best opens the described session. Search using
the suggested queries (or your own variants), examine candidates, then
commit one choice. Never invent track titles — only commit a track you
have seen in a search result."""


# ══════════════════════════════════════════════════════════════════════════════
#  TRACE
# ══════════════════════════════════════════════════════════════════════════════

def _trace_entry(
    step:        int,
    kind:        str,                 # "think" | "act" | "observe" | "explain"
    content:     str,
    tool_name:   str | None = None,
    tool_args:   dict | None = None,
    tool_result: dict | None = None,
) -> dict:
    return {
        "step":        step,
        "kind":        kind,
        "content":     content,
        "tool_name":   tool_name,
        "tool_args":   tool_args,
        "tool_result": tool_result,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  SCORED FALLBACK  (safety net when the ReAct loop can't commit)
# ══════════════════════════════════════════════════════════════════════════════

def _best_fallback(candidates: list[dict], state: dict) -> dict:
    """
    Pick the candidate closest to the current state on energy + valence,
    penalising BPM / Camelot incompatibility. Used when the model exits
    the loop without queuing a track.
    """
    if not candidates:
        return {}

    target_energy  = state.get("energy",  0.5)
    target_valence = state.get("valence", 0.5)

    def score(c: dict) -> float:
        e_diff          = abs(c.get("energy_est",  0.5) - target_energy)
        v_diff          = abs(c.get("valence_est", 0.5) - target_valence)
        bpm_penalty     = 0.2 if not c.get("bpm_ok",     True) else 0.0
        camelot_penalty = 0.2 if not c.get("camelot_ok", True) else 0.0
        return e_diff + v_diff + bpm_penalty + camelot_penalty   # lower = better

    scored = sorted(candidates, key=score)

    if DISPLAY_LOGS:
        print("\n[fallback] scored candidates:")
        for c in scored[:6]:
            s       = score(c)
            bpm_sym = "✓" if c.get("bpm_ok",     True) else "✗"
            cam_sym = "✓" if c.get("camelot_ok", True) else "✗"
            print(f"  {s:.2f} — {c['name']} — {c['artist']}  (bpm={bpm_sym}  key={cam_sym})")
        print(f"  → best: {scored[0]['name']} — {scored[0]['artist']}")

    return scored[0]


def _best_fallback_top_n(candidates: list[dict], state: dict, n: int = 3) -> list[dict]:
    """
    Return the top N candidates ranked by energy+valence distance from the
    current listener state. Used by the Python pre-step in run_agent_cycle
    to build the shortlist injected into the LLM's user message.

    Unlike _best_fallback (which returns a single best), this hands the LLM
    a ranked list so it has fallback options without needing another search call.
    """
    if not candidates:
        return []

    target_energy  = state.get("energy",  0.5)
    target_valence = state.get("valence", 0.5)

    def score(c: dict) -> float:
        e_diff = abs(c.get("energy_est", 0.5) - target_energy)
        v_diff = abs(c.get("valence_est", 0.5) - target_valence)
        return e_diff + v_diff   # lower = better fit

    return sorted(candidates, key=score)[:n]


def _current_track_profile() -> dict | None:
    """
    Full enriched profile (tags, camelot_position, bpm, energy/valence) of the
    track currently playing on Spotify. Looks it up in session history first; if
    it is not there (e.g. the listener changed track directly in Spotify), it is
    enriched on the fly via get_track_details (cached, so cheap on repeat).
    Returns None when nothing is playing or the track cannot be resolved.
    """
    playback = tool_module.get_current_playback()
    name     = (playback.get("track_name") or "").strip()
    artist   = (playback.get("artist") or "").strip()
    if not name:
        return None

    for entry in tool_module.get_session_history().get("recent", []):
        if entry.get("name", "").lower() == name.lower():
            return entry

    details = tool_module.get_track_details(name, artist)
    if isinstance(details, dict) and details.get("name") and not details.get("error"):
        return details
    return None


def _search_seed_tags(state: dict, current: dict | None) -> list[str]:
    """
    Tags that drive candidate discovery — taken from the CURRENTLY PLAYING track's
    top Last.fm tags (up to two) so the candidate pool stays on-genre with what's
    playing. The listener's artist name is skipped (it often appears as a tag).

    Energy/mood adaptation is intentionally NOT mixed in as a second tag here:
    Last.fm tag search is single-tag, so adding e.g. 'indie' or 'chill' fetches a
    separate off-genre pool (the bug where reggaeton playback pulled in indie
    rock). Adaptation is applied downstream by ranking these on-genre candidates
    on energy/valence distance. Falls back to state tags only when the current
    track has no usable tags.
    """
    if current:
        artist = (current.get("artist") or "").strip().lower()
        seeds: list[str] = []
        for t in current.get("tags", []):
            tl = (t or "").strip().lower()
            if tl and tl != artist and tl not in seeds:
                seeds.append(tl)
            if len(seeds) >= 2:
                break
        if seeds:
            return seeds
    return _state_to_tags(state)


def _compatible_pool(
    candidates:   list[dict],
    state:        dict,
    current_key:  str | None,
    current_bpm:  float | None,
) -> list[dict]:
    """
    Hard harmonic + tempo filter against the currently playing track. A candidate
    is kept if its Camelot key is compatible (strength ≥ 0.4) AND its BPM is within
    the arc-phase threshold. Unknown values (current or candidate) are treated as
    compatible so missing Soundcharts data never empties the pool. If nothing
    passes, returns the full list unchanged (soft degrade — never stall).
    """
    arc       = state.get("arc_phase", "build")
    threshold = _BPM_THRESHOLDS.get(arc, 8.0)
    from_key  = camelot_parse(current_key) if current_key else None

    def harmonic_ok(c: dict) -> bool:
        if from_key is None:
            return True
        to_key = camelot_parse(c.get("camelot_position") or "")
        if to_key is None:
            return True
        return compatibility_strength(from_key, to_key) >= _HARMONIC_MIN

    def bpm_ok(c: dict) -> bool:
        cand_bpm = c.get("bpm")
        if not current_bpm or not cand_bpm:
            return True
        return abs(current_bpm - cand_bpm) <= threshold

    compatible = [c for c in candidates if harmonic_ok(c) and bpm_ok(c)]
    return compatible if compatible else candidates


def _state_to_tags(state: dict) -> list[str]:
    """
    Map the listener state vector to a ranked list of Last.fm tags.

    Returns up to two tags used to drive the pre-step Last.fm search.
    The first tag is the most specific signal; the second is a fallback
    if the first yields too few Spotify-resolvable candidates.

    Tag choices are based on which combinations perform well in Last.fm's
    crowd-tagged index:
      - Energy axis:   energetic / chill / indie (mid)
      - Valence axis:  happy / melancholic / dark
      - Arc modifier:  anthemic (peak), mellow (cooldown)
    """
    energy  = state.get("energy",  0.5)
    valence = state.get("valence", 0.5)
    arc     = state.get("arc_phase", "build")

    tags: list[str] = []

    # ── Arc phase takes priority at extremes ─────────────────────
    if arc == "peak":
        tags.append("anthemic")
    elif arc == "cooldown":
        tags.append("mellow")

    # ── Energy dimension ──────────────────────────────────────────
    if energy > 0.65:
        tags.append("energetic")
    elif energy < 0.35:
        tags.append("chill")
    else:
        tags.append("indie")           # reliable mid-energy tag on Last.fm

    # ── Valence dimension (only if strongly positive or negative) ─
    if valence > 0.65 and "anthemic" not in tags:
        tags.append("happy")
    elif valence < 0.35 and "mellow" not in tags:
        tags.append("melancholic")

    # Return at most 2 — enough for two parallel Last.fm tag searches
    return tags[:2] if tags else ["indie"]


def _state_to_fallback_query(state: dict) -> str:
    """
    Build a Spotify text search query from the listener state.
    Used as a last resort when Last.fm tag search returns nothing resolvable.
    """
    energy  = state.get("energy",  0.5)
    valence = state.get("valence", 0.5)

    energy_word = (
        "high energy" if energy > 0.65 else
        "low energy"  if energy < 0.35 else
        "moderate"
    )
    mood_word = (
        "uplifting"    if valence > 0.65 else
        "melancholic"  if valence < 0.35 else
        "neutral"
    )
    return f"{energy_word} {mood_word}"


# ══════════════════════════════════════════════════════════════════════════════
#  COMPILED CYCLE GRAPHS  (lazy singletons — built once, reused every cycle)
# ══════════════════════════════════════════════════════════════════════════════
#
# Two variants, both terminating on add_track_to_queue:
#   • cycle  — CYCLE_TOOLS (no search; the Python pre-step already searched)
#   • full   — ALL_TOOLS  (used only when the pre-step found no candidates)

_CYCLE_GRAPH  = None
_FULL_GRAPH   = None
_OPENER_GRAPH = None


def _get_cycle_graph(full: bool):
    """Return the compiled cycle graph, building it on first use."""
    global _CYCLE_GRAPH, _FULL_GRAPH
    if full:
        if _FULL_GRAPH is None:
            _FULL_GRAPH = build_dj_graph(lc_tools.ALL_TOOLS, stop_tool="add_track_to_queue")
        return _FULL_GRAPH
    if _CYCLE_GRAPH is None:
        _CYCLE_GRAPH = build_dj_graph(lc_tools.CYCLE_TOOLS, stop_tool="add_track_to_queue")
    return _CYCLE_GRAPH


def _get_opener_graph():
    """Return the compiled natural-language opener graph, building it on first use."""
    global _OPENER_GRAPH
    if _OPENER_GRAPH is None:
        _OPENER_GRAPH = build_opener_graph(lc_tools.OPENER_TOOLS, stop_tool="select_opening_track")
    return _OPENER_GRAPH


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — run_agent_cycle
# ══════════════════════════════════════════════════════════════════════════════

def run_agent_cycle(
    feedback_event:  str | None = None,
    feedback_track:  str | None = None,
    feedback_artist: str | None = None,
    verbose:         bool = True,
) -> dict:
    """
    One per-track ReAct cycle.

    Feedback (if any) is applied in Python BEFORE the loop so that
    get_listener_state will return the post-feedback vector when the
    model reads it. The feedback is recorded at step 0 in the trace.

    A Python pre-step then searches for candidates, scores them by
    energy/valence fit, and injects the top 3 into the user message.
    The cycle graph is given lc_tools.CYCLE_TOOLS (no search tools) so it can
    only verify harmonic fit and commit — no iteration budget wasted on search.
    If the pre-step finds nothing, falls back to the full-toolset graph.

    Returns:
      {
        "explanation":  str,
        "queued_track": dict | None,
        "trace":        list[dict],
        "success":      bool,
      }
    """
    trace: list[dict] = []
    tool_module.DISPLAY_LOGS = DISPLAY_LOGS

    # ── 1. Apply feedback in Python (step 0) ──────────────────
    if feedback_event and feedback_track and feedback_artist:
        fb_result = tool_module.update_listener_state(
            feedback_event, feedback_track, feedback_artist
        )
        trace.append(_trace_entry(
            step        = 0,
            kind        = "act",
            content     = f"Applied feedback: {feedback_event} on {feedback_track}",
            tool_name   = "update_listener_state",
            tool_args   = {
                "event":      feedback_event,
                "track_name": feedback_track,
                "artist":     feedback_artist,
            },
            tool_result = fb_result,
        ))
        if verbose:
            print(f"\n[feedback] {feedback_event} on '{feedback_track}'")

    # ── 2. Python pre-step: fetch, score, build shortlist ─────
    # Candidates are gathered in Python before the LLM loop so the model
    # receives a pre-scored shortlist and does not waste iterations on search.
    #
    # Strategy: use Last.fm tag search (via _state_to_tags) rather than a
    # Spotify text query. Last.fm tags are crowdsourced mood/genre labels —
    # searching "energetic" in Last.fm returns actually energetic tracks,
    # whereas the same query on Spotify just matches track titles.
    #
    # Two tags are tried in order; results are merged and de-duplicated.
    # Falls back to Spotify text search only if both tag searches fail.
    state = tool_module.get_listener_state()
    # Seed discovery from the CURRENTLY PLAYING track's genre tags so candidates
    # stay on-genre with what's playing. Energy/mood adaptation is applied later
    # by ranking on energy/valence distance — not by mixing in an off-genre tag.
    current = _current_track_profile()
    tags    = _search_seed_tags(state, current)

    if verbose:
        now = f"{current.get('name')} ({current.get('camelot_position')}/{current.get('bpm')})" if current else "?"
        print(f"\n[pre-fetch] now playing: {now} → Last.fm seed tags: {tags}")

    raw_candidates: list[dict] = []
    seen_ids: set[str] = set()

    for tag in tags:
        if len(raw_candidates) >= 15:
            break
        result = tool_module.search_tracks_by_tag(tag, limit=10)
        for c in result.get("candidates", []):
            cid = c.get("id")
            if cid and cid not in seen_ids:
                seen_ids.add(cid)
                raw_candidates.append(c)
        if verbose:
            print(f"[pre-fetch] tag '{tag}' → {len(result.get('candidates', []))} candidates")
    # Fallback to Spotify text search if Last.fm tag search resolved nothing
    if not raw_candidates:
        query      = _state_to_fallback_query(state)
        pre_search = tool_module.search_tracks(query, limit=10)
        raw_candidates = pre_search.get("candidates", [])
        if verbose:
            print(f"[pre-fetch] Last.fm empty — Spotify fallback query: '{query}'")

    # Harmonic + tempo HARD filter against the currently playing track, then rank
    # the survivors by energy/valence fit. This enforces transition compatibility
    # deterministically rather than relying on the LLM to call check_transition.
    current_key = current.get("camelot_position") if current else None
    current_bpm = current.get("bpm") if current else None
    compatible_pool = _compatible_pool(raw_candidates, state, current_key, current_bpm)
    shortlist = _best_fallback_top_n(compatible_pool, state, n=3)

    trace.append(_trace_entry(
        step       = 0,
        kind       = "act",
        content    = (
            f"Pre-fetch: tags={tags} → {len(raw_candidates)} candidates → "
            f"{len(compatible_pool)} compatible (from key={current_key or '?'} "
            f"BPM={current_bpm or '?'}) → shortlist of {len(shortlist)}"
        ),
        tool_name  = "search_tracks_by_tag",
        tool_args  = {"tags": tags, "from_key": current_key, "from_bpm": current_bpm},
        tool_result= {"count": len(raw_candidates),
                      "compatible": len(compatible_pool),
                      "shortlist_size": len(shortlist)},
    ))

    if DISPLAY_LOGS and shortlist:
        print(f"[pre-fetch] shortlist ({len(shortlist)}):")
        for c in shortlist:
            key_s = c.get("camelot_position") or "?"
            bpm_s = f"BPM={c['bpm']}" if c.get("bpm") else "BPM=?"
            print(f"  • {c['name']} — {c['artist']}  key={key_s}  {bpm_s}"
                  f"  E={c.get('energy_est', 0.5):.2f}  V={c.get('valence_est', 0.5):.2f}")

    # Build the candidate section injected into the user message
    if shortlist:
        lines = []
        for i, c in enumerate(shortlist, 1):
            key_s  = c.get("camelot_position") or "?"
            bpm_s  = f"BPM={c['bpm']}" if c.get("bpm") else "BPM=unknown"
            tags_s = ", ".join(c.get("tags", [])[:3]) or "—"
            lines.append(
                f"  {i}. \"{c['name']}\" by {c['artist']}"
                f"  —  key={key_s}  {bpm_s}"
                f"  energy={c.get('energy_est', 0.5):.2f}"
                f"  valence={c.get('valence_est', 0.5):.2f}"
                f"  tags=[{tags_s}]"
            )
        current_line = (
            f"\n\nCurrently playing: key={current_key or 'unknown'} "
            f"BPM={current_bpm or 'unknown'}. The candidates below have already been "
            f"filtered to be harmonically and tempo compatible with it."
            if (current_key or current_bpm) else ""
        )
        candidate_section = (
            current_line
            + f"\n\nPre-scored candidates from Last.fm tag search {tags}"
            " (already compatibility-filtered, ranked best-first by energy/valence fit):\n"
            + "\n".join(lines)
            + (f"\n\nConfirm harmonic fit of candidate #1 with check_transition"
               f"(from_camelot={current_key!r}, to_camelot=<candidate key>), then commit "
               f"with add_track_to_queue."
               if current_key else
               "\n\nCommit candidate #1 with add_track_to_queue.")
        )
        use_full_tools = False   # search already done — no need for LLM to search
    else:
        # No pre-fetched candidates — give the LLM full tools so it can search
        candidate_section = ""
        use_full_tools    = True

    # ── 3. Build initial messages ─────────────────────────────
    context_note = (
        f" The listener just triggered a '{feedback_event}' on the current track —"
        " factor that into your reasoning."
        if feedback_event else ""
    )
    messages = [
        SystemMessage(content=CYCLE_SYSTEM_PROMPT),
        HumanMessage(content=(
            f"Select and queue the next track.{context_note}{candidate_section}"
        )),
    ]

    if verbose:
        print("\n[react] starting cycle…")

    # ── 4. Run the ReAct loop (LangGraph) ─────────────────────
    # The graph builds its own trace starting at step 1; we extend the trace
    # already holding the step-0 feedback + pre-fetch entries.
    graph     = _get_cycle_graph(full=use_full_tools)
    graph_out = run_graph(graph, messages, max_iterations=MAX_ITERATIONS)
    trace.extend(graph_out["trace"])
    result = {
        "queued":      graph_out["queued"],
        "explanation": graph_out["explanation"],
    }

    # ── 5. Fallback if no track queued ────────────────────────
    if not result.get("queued"):
        if verbose or DISPLAY_LOGS:
            print("[fallback] loop did not queue — using scored fallback")
        # Reuse the compatibility-filtered pool; only re-search if there was nothing
        candidates = compatible_pool or tool_module.search_tracks(query, limit=8).get("candidates", [])
        best       = _best_fallback(candidates, state)
        if best:
            queue_result     = tool_module.add_track_to_queue(best["name"], best["artist"])
            result["queued"] = queue_result
            if not result.get("explanation"):
                result["explanation"] = (
                    f"Up next: {best['name']} by {best['artist']} — chosen as the closest "
                    f"match to the current state when the agent could not converge."
                )

    # ── 6. Normalise return shape ─────────────────────────────
    queued      = result.get("queued") or {}
    explanation = result.get("explanation", "")

    # If the explain turn returned nothing, surface the model's own reasoning from
    # the commit turn (last `think` entry) before falling back to a generic line.
    if not explanation:
        thinks = [t.get("content") for t in trace
                  if t.get("kind") == "think" and t.get("content")]
        if thinks:
            explanation = thinks[-1]

    if not explanation:
        q = queued.get("queued", {}) if isinstance(queued, dict) else {}
        if q:
            explanation = f"Up next: {q.get('name')} by {q.get('artist')}."

    success = isinstance(queued, dict) and queued.get("success", False)

    if verbose:
        print(f"\n[explain] {explanation}")
        print(f"\n{'='*55}")
        print(f"Cycle complete — {'success' if success else 'failed'}")
        print(f"{'='*55}\n")

    return {
        "explanation":  explanation,
        "queued_track": queued.get("queued") if isinstance(queued, dict) else None,
        "trace":        trace,
        "success":      success,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PUBLIC API — start_session
# ══════════════════════════════════════════════════════════════════════════════

def start_session(description: str, verbose: bool = False) -> dict:
    """
    Bootstrap a new session from a natural-language description.

    Two-phase design:
      1. Structured extraction (single Groq call): parse the description
         into a vibe vector — energy, valence, focus, openness, social,
         search_queries, session_label, confident.
      2. ReAct sub-loop (focused 3-tool subset): the model searches
         Spotify, inspects candidates, and commits one opener via the
         synthetic select_opening_track tool.

    Python handles the Spotify play() and session-state recording around
    the loop — those are mechanics, not reasoning.

    Returns:
      success:        {"success": True, "session_label", "opening_track",
                       "fallback_used", "explanation", "trace"}
      failure:        {"success": False, "error": "no_device"|"no_candidates",
                       "session_label", "trace"}
    """
    import random

    trace: list[dict] = []
    tool_module.DISPLAY_LOGS = DISPLAY_LOGS

    if verbose:
        print(f"\n[start_session] '{description}'")

    # ── 1. Interpret the description (structured JSON) ────────
    vibe = _interpret_description(description, trace, verbose=verbose)

    session_label   = (vibe or {}).get("session_label", description[:50]) or description[:50]
    confident       = bool((vibe or {}).get("confident", False))
    search_queries  = (vibe or {}).get("search_queries", []) or []
    target_energy   = float((vibe or {}).get("energy",   0.5))
    target_valence  = float((vibe or {}).get("valence",  0.5))
    target_focus    = float((vibe or {}).get("focus",    0.5))
    target_openness = float((vibe or {}).get("openness", 0.7))
    target_social   = float((vibe or {}).get("social",   0.3))

    # ── 2. Reset session & seed the state vector ──────────────
    tool_module.reset_session("general")
    if vibe:
        tool_module._state = init_state_from_values(
            energy   = target_energy,
            valence  = target_valence,
            focus    = target_focus,
            openness = target_openness,
            social   = target_social,
        )

    # ── 3. ReAct sub-loop to pick the opener ──────────────────
    # Run the opener whenever interpretation gave us something searchable. We do
    # NOT gate on `confident`: a valid description the model marks low-confidence
    # should still drive the opener, not jump straight to random favourites. The
    # top-tracks fallback only triggers if the opener can't commit a real track.
    selected: dict | None = None
    if search_queries:
        selected = _react_pick_opener(
            label          = session_label,
            vibe           = vibe or {},
            search_queries = search_queries,
            trace          = trace,
            verbose        = verbose,
        )

    # ── 4. Resolve to an actual SpotifyTrack ──────────────────
    fallback_used = False
    sp_track      = None

    if selected:
        results = tool_module._spotify.search(
            f"{selected['name']} {selected['artist']}", limit=1
        )
        if results:
            sp_track = results[0]

    if sp_track is None:
        # Either the model gave nothing usable or vibe parsing failed
        fallback_used = True
        top_tracks    = tool_module._spotify.get_top_tracks(limit=20)
        if not top_tracks:
            return {
                "success":       False,
                "error":         "no_candidates",
                "session_label": session_label,
                "trace":         trace,
            }
        sp_track = random.choice(top_tracks)
        if not confident:
            session_label = "Your recent favourites"

    # ── 5. Start immediate playback ───────────────────────────
    if not tool_module._spotify.play(sp_track):
        return {
            "success":       False,
            "error":         "no_device",
            "session_label": session_label,
            "trace":         trace,
        }

    # ── 5b. Pre-warm the sentence-transformer in the background ──
    # The model loads on first use inside get_track_details → estimate_features.
    # Starting it here (as a daemon thread) means it will be ready before
    # the first mid-session cycle runs, eliminating the cold-start spike.
    _prewarm_thread = threading.Thread(
        target=prewarm_embedding_model, daemon=True, name="st-prewarm"
    )
    _prewarm_thread.start()

    # ── 6. Record the opening track in session state ──────────
    candidate = tool_module._record_played_track(sp_track)

    if verbose:
        print(f"[start_session] Playing: {sp_track.name} — {sp_track.artist}")
        print(f"[start_session] Label: {session_label}  fallback={fallback_used}")

    return {
        "success":       True,
        "session_label": session_label,
        "opening_track": candidate,
        "fallback_used": fallback_used,
        "trace":         trace,
        "explanation":   (
            f"Starting with '{sp_track.name}' by {sp_track.artist} "
            f"for your '{session_label}' session."
        ),
    }


# ── start_session internals ───────────────────────────────────────────────────

def _interpret_description(
    description: str,
    trace:       list[dict],
    verbose:     bool = False,
) -> dict | None:
    """
    Single structured-JSON call via ChatGroq (no tools). Returns the parsed vibe
    dict on success, None on parse failure. Records a 'think' trace entry either
    way so the bootstrap reasoning is visible to post-hoc analysis.
    """
    user_msg = SESSION_INTERPRET_USER.format(description=description)

    try:
        response = _get_interpret_llm().invoke([
            SystemMessage(content=SESSION_INTERPRET_SYSTEM),
            HumanMessage(content=user_msg),
        ])
        # gpt-oss often returns the JSON on the reasoning channel with empty
        # .content — fall back to reasoning_content (same issue as the explain node).
        raw = response.content if isinstance(response.content, str) else ""
        raw = raw.strip()
        if not raw:
            raw = (getattr(response, "additional_kwargs", {}) or {}).get("reasoning_content", "") or ""
            raw = raw.strip()

        # Strip markdown fences if the model added them despite the system prompt
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()
        # Be robust to any surrounding prose: take the outermost JSON object.
        if "{" in raw and "}" in raw:
            raw = raw[raw.index("{"): raw.rindex("}") + 1]
        parsed = json.loads(raw)

        trace.append(_trace_entry(
            step       = 0,
            kind       = "think",
            content    = (
                f"Interpreted vibe → label='{parsed.get('session_label', '?')}' "
                f"energy={parsed.get('energy')} valence={parsed.get('valence')} "
                f"confident={parsed.get('confident')}"
            ),
            tool_result= parsed,
        ))
        if verbose:
            print(f"[interpret] {parsed.get('session_label')}  confident={parsed.get('confident')}")
        return parsed

    except Exception as exc:
        trace.append(_trace_entry(
            step    = 0,
            kind    = "think",
            content = f"Vibe interpretation failed: {exc}",
        ))
        if verbose:
            print(f"[interpret] failed: {exc}")
        return None


def _react_pick_opener(
    label:          str,
    vibe:           dict,
    search_queries: list[str],
    trace:          list[dict],
    verbose:        bool = False,
) -> dict | None:
    """
    Focused ReAct sub-loop that asks the model to pick an opening track.

    Uses the opener graph (lc_tools.OPENER_TOOLS):
      - search_tracks_by_playlist, search_tracks, get_track_details (real tools)
      - select_opening_track (synthetic terminal; the model may only commit a
        track it has actually seen in a prior search result — the guard lives in
        the opener graph node, validating against seen_tracks in DJState).

    Returns the chosen {name, artist} dict, or None if the model fails
    to commit within the iteration budget.
    """
    queries_text = "\n".join(f"  • {q}" for q in search_queries[:3])
    user_msg = (
        f"Session vibe: {label}\n"
        f"Inferred listener state — "
        f"energy={float(vibe.get('energy', 0.5)):.2f}, "
        f"valence={float(vibe.get('valence', 0.5)):.2f}, "
        f"focus={float(vibe.get('focus', 0.5)):.2f}.\n\n"
        f"Suggested search queries:\n{queries_text}\n\n"
        f"Pick and commit the opening track."
    )

    messages = [
        SystemMessage(content=SESSION_PICK_SYSTEM),
        HumanMessage(content=user_msg),
    ]

    if verbose:
        print("\n[start_session/react] picking opener…")

    graph     = _get_opener_graph()
    graph_out = run_graph(graph, messages, max_iterations=8)
    trace.extend(graph_out["trace"])

    choice = graph_out.get("choice")
    if graph_out.get("queued") and choice and choice.get("name") and choice.get("artist"):
        return {"name": choice["name"], "artist": choice["artist"]}
    return None
