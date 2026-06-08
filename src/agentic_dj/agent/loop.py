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
from agentic_dj.music.tags import prewarm_embedding_model

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

Goal: choose the single best next track and queue it. Explain your choice in
one or two plain sentences.

Tool call order (stay within this):
  1. get_ranked_candidates()                      — gather the top compatible options
  2. (optional) check_transition(from_camelot, to_camelot) — confirm the harmonic fit
  3. add_track_to_queue(track_name, artist)       — TERMINAL: call once, then stop

get_ranked_candidates returns up to 5 options already filtered to be harmonically
and tempo compatible with the current track and ranked best-first by fit to the
listener state and session arc. Each option includes its key, BPM, energy,
valence, tags, and harmonic_score. Work ONLY from this list — do not invent
tracks. Normally commit candidate #1; only move down the list if you have a
concrete musical reason.

Ground every claim in actual tool output — never invent values. Do not queue a
track already played this session. In your final explanation reference at least
one concrete signal (arc phase or listener state dimension) and one musical
property (key, BPM, or energy)."""


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
#  COMPILED CYCLE GRAPHS  (lazy singletons — built once, reused every cycle)
# ══════════════════════════════════════════════════════════════════════════════

_CYCLE_GRAPH  = None
_OPENER_GRAPH = None


def _get_cycle_graph():
    """Return the compiled mid-session cycle graph (CYCLE_TOOLS), building it once."""
    global _CYCLE_GRAPH
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

    The cycle graph (lc_tools.CYCLE_TOOLS) drives discovery itself: the model
    calls get_ranked_candidates (gather → compatibility-filter → composite rank
    → top 5), optionally confirms with check_transition, and commits with
    add_track_to_queue. If the graph fails to commit, a Python fallback queues
    the top-ranked candidate from the same pipeline.

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

    # ── 2. Build initial messages ─────────────────────────────
    # Discovery + ranking now live in the get_ranked_candidates tool, which the
    # LLM calls itself — no Python pre-fetch / injected shortlist.
    context_note = (
        f" The listener just triggered a '{feedback_event}' on the current track —"
        " factor that into your reasoning."
        if feedback_event else ""
    )
    messages = [
        SystemMessage(content=CYCLE_SYSTEM_PROMPT),
        HumanMessage(content=f"Select and queue the next track.{context_note}"),
    ]

    if verbose:
        print("\n[react] starting cycle…")

    # ── 3. Run the ReAct loop (LangGraph) ─────────────────────
    # The graph builds its own trace starting at step 1; we extend the trace
    # already holding the step-0 feedback entry.
    graph     = _get_cycle_graph()
    graph_out = run_graph(graph, messages, max_iterations=MAX_ITERATIONS)
    trace.extend(graph_out["trace"])
    result = {
        "queued":      graph_out["queued"],
        "explanation": graph_out["explanation"],
    }

    # ── 4. Fallback if no track queued ────────────────────────
    # Reuse the same gather→filter→rank pipeline and take the #1 candidate.
    if not result.get("queued"):
        if verbose or DISPLAY_LOGS:
            print("[fallback] loop did not queue — using top ranked candidate")
        ranked = tool_module.get_ranked_candidates(limit=1).get("candidates", [])
        if ranked:
            best             = ranked[0]
            queue_result     = tool_module.add_track_to_queue(best["name"], best["artist"])
            result["queued"] = queue_result
            if not result.get("explanation"):
                result["explanation"] = (
                    f"Up next: {best['name']} by {best['artist']} — the top-ranked compatible "
                    f"match when the agent could not converge."
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
