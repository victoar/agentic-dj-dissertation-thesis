"""
ReAct agent loop — the reasoning core of the Agentic DJ system.

Both entry points (run_agent_cycle, start_session) are implemented as true
Reason → Act → Observe loops on Groq's native function-calling. The model
is given a goal and a toolbox; it decides which tools to call, in what
order, and when it has gathered enough to commit. No procedural rules.

Every iteration logs:
  • a `think` trace entry  (the model's natural-language reasoning, if any)
  • one `act`   trace entry per tool call  (tool name, arguments, result)

This makes the trace a faithful record of the agent's behaviour — the
substrate the dissertation's failure taxonomy (state lag, novelty
miscalibration, arc rigidity, key tunnel vision) is built on.

Architecture:
  - Single generic _run_react_loop driver, parametrised by tool registry
    and the name of the terminal "stop tool".
  - run_agent_cycle uses a Python pre-fetch step to build a scored candidate
    shortlist, then passes it to a focused LLM loop (CYCLE_TOOLS — no search)
    that only needs to verify harmonic fit and commit. Falls back to full
    GROQ_TOOLS if the pre-fetch returns nothing.
  - start_session uses a focused inner toolset (search + select-opening)
    and stops on a synthetic select_opening_track tool defined inline.
  - A structured-JSON pre-step in start_session extracts the vibe vector
    from the description. Extraction is one-shot by design — reasoning
    happens in the sub-loop that picks the actual track.
"""

import json
import os
import threading
import time
from typing import Any

from groq import Groq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from agentic_dj.agent import tools as tool_module
from agentic_dj.agent import lc_tools
from agentic_dj.agent.graph import build_dj_graph, build_opener_graph, run_graph
from agentic_dj.agent.state import init_state_from_values
from agentic_dj.music.tags import prewarm_embedding_model

# ── Configuration ─────────────────────────────────────────────────────────────

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "openai/gpt-oss-120b"

DISPLAY_LOGS: bool = True       # set False to silence the trace stream
MAX_ITERATIONS:       int = 15  # safety cap on a single ReAct loop
MAX_BACKOFF_ATTEMPTS: int = 4   # exponential backoff retries on 429


# ══════════════════════════════════════════════════════════════════════════════
#  TOOL CATALOGUE  —  the 12 tools the cycle loop reasons over
# ══════════════════════════════════════════════════════════════════════════════
#
# Plain JSON schemas, OpenAI / Groq compatible. Each declaration is the
# single source of truth: GROQ_TOOLS wraps it in the API envelope, and
# TOOL_REGISTRY maps the name back to the Python function.

TOOL_DECLARATIONS: list[dict] = [
    {
        "name": "get_listener_state",
        "description": "Return the current listener state vector — energy, valence, focus, openness, social, arc_phase.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "update_listener_state",
        "description": "Update the listener state based on a feedback event.",
        "parameters": {
            "type": "object",
            "properties": {
                "event":      {"type": "string", "description": "skip, early_skip, replay, full_listen, partial_listen, thumbs_up, thumbs_down"},
                "track_name": {"type": "string"},
                "artist":     {"type": "string"},
            },
            "required": ["event", "track_name", "artist"],
        },
    },
    {
        "name": "get_session_arc",
        "description": "Return the current arc phase and a description of what it means for track selection.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_compatible_keys",
        "description": "Return all harmonically compatible Camelot positions for a given key.",
        "parameters": {
            "type": "object",
            "properties": {
                "camelot_position": {"type": "string", "description": "e.g. '8B', '4A'"},
            },
            "required": ["camelot_position"],
        },
    },
    {
        "name": "check_transition",
        "description": "Check harmonic smoothness between two Camelot positions. Returns a score 0.0–1.0 and a verdict.",
        "parameters": {
            "type": "object",
            "properties": {
                "from_camelot": {"type": "string"},
                "to_camelot":   {"type": "string"},
            },
            "required": ["from_camelot", "to_camelot"],
        },
    },
    {
        "name": "estimate_bpm_compatibility",
        "description": "Check whether a BPM jump between two tracks is smooth for the current arc phase.",
        "parameters": {
            "type": "object",
            "properties": {
                "current_bpm":   {"type": "number"},
                "candidate_bpm": {"type": "number"},
                "arc_phase":     {"type": "string"},
            },
            "required": ["current_bpm", "candidate_bpm"],
        },
    },
    {
        "name": "search_tracks_by_tag",
        "description": (
            "Search for tracks by Last.fm tag — the accurate way to find music by mood or genre. "
            "Use this for descriptive queries like 'energetic', 'chill', 'melancholic', 'dark', "
            "'happy', 'anthemic', 'ambient', 'upbeat', 'mellow'. "
            "Last.fm tags are crowdsourced by millions of listeners and map directly to moods/genres. "
            "Prefer this over search_tracks for any mood or energy query."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "tag":   {"type": "string",  "description": "A Last.fm mood/genre tag e.g. 'energetic', 'chill', 'dark'"},
                "limit": {"type": "integer", "description": "Max candidates (default 15)"},
            },
            "required": ["tag"],
        },
    },
    {
        "name": "search_artist_tracks",
        "description": (
            "Get an artist's most popular tracks, resolved to Spotify. "
            "Use this when the listener wants tracks from a specific artist — it fetches their "
            "top tracks from Last.fm ranked by listener count, far more reliable than a Spotify "
            "text search which may return covers or tribute acts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "artist": {"type": "string",  "description": "Artist name e.g. 'Radiohead'"},
                "limit":  {"type": "integer", "description": "Max candidates (default 15)"},
            },
            "required": ["artist"],
        },
    },
    {
        "name": "search_tracks",
        "description": "Search Spotify by text query for a specific track or artist name. Use for precise lookups like 'Creep Radiohead' or 'One More Time Daft Punk'. For mood/energy queries use search_tracks_by_tag instead; for artist catalogues use search_artist_tracks instead.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "search_tracks_by_playlist",
        "description": "Search for Spotify playlists matching a vibe, era, or mood description and return a sample of tracks from those playlists. Use for era-specific queries like '2016 pop hits' or 'chill Sunday morning'. Returns raw metadata only — no BPM or key data.",
        "parameters": {
            "type": "object",
            "properties": {
                "query":       {"type": "string",  "description": "Short playlist-style search string, e.g. '2016 pop hits'"},
                "sample_size": {"type": "integer", "description": "Max candidates to return (default 20)"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_track_details",
        "description": "Get full details for a specific track — metadata, tags, energy, valence, BPM, Camelot key.",
        "parameters": {
            "type": "object",
            "properties": {
                "track_name": {"type": "string"},
                "artist":     {"type": "string"},
            },
            "required": ["track_name", "artist"],
        },
    },
    {
        "name": "get_current_playback",
        "description": "Return what is currently playing on Spotify — track name, artist, progress.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_queue_state",
        "description": "Return the current planned queue and upcoming tracks.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_session_history",
        "description": "Return tracks played so far this session, most recent first.",
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "add_track_to_queue",
        "description": "Add a track to the Spotify playback queue. This is the terminal action — call it once with the single best candidate.",
        "parameters": {
            "type": "object",
            "properties": {
                "track_name": {"type": "string"},
                "artist":     {"type": "string"},
            },
            "required": ["track_name", "artist"],
        },
    },
]

# Groq/OpenAI function-call envelope
GROQ_TOOLS: list[dict] = [
    {"type": "function", "function": decl} for decl in TOOL_DECLARATIONS
]

# Focused schema for mid-session cycles — search is done in the Python pre-step,
# so we strip all four search tools from the tool list.
# This prevents the LLM from burning iterations on a redundant second search
# when a scored shortlist is already injected into the user message.
# Falls back to full GROQ_TOOLS when the pre-step returns no candidates.
_SEARCH_TOOL_NAMES = {
    "search_tracks_by_tag",
    "search_artist_tracks",
    "search_tracks",
    "search_tracks_by_playlist",
}
CYCLE_TOOLS: list[dict] = [
    t for t in GROQ_TOOLS
    if t["function"]["name"] not in _SEARCH_TOOL_NAMES
]

# Name → Python function lookup for tool execution
TOOL_REGISTRY: dict[str, Any] = {
    "get_listener_state":         tool_module.get_listener_state,
    "update_listener_state":      tool_module.update_listener_state,
    "get_session_arc":            tool_module.get_session_arc,
    "get_compatible_keys":        tool_module.get_compatible_keys,
    "check_transition":           tool_module.check_transition,
    "estimate_bpm_compatibility": tool_module.estimate_bpm_compatibility,
    "search_tracks_by_tag":       tool_module.search_tracks_by_tag,
    "search_artist_tracks":       tool_module.search_artist_tracks,
    "search_tracks":              tool_module.search_tracks,
    "search_tracks_by_playlist":  tool_module.search_tracks_by_playlist,
    "get_track_details":          tool_module.get_track_details,
    "get_current_playback":       tool_module.get_current_playback,
    "get_queue_state":            tool_module.get_queue_state,
    "get_session_history":        tool_module.get_session_history,
    "add_track_to_queue":         tool_module.add_track_to_queue,
}


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
#  GROQ CALL WITH EXPONENTIAL BACKOFF
# ══════════════════════════════════════════════════════════════════════════════

def _groq_call(
    messages:    list[dict],
    tools:       list[dict] | None = None,
    tool_choice: str = "auto",
) -> Any:
    """
    Wrap client.chat.completions.create with retry on 429 / rate-limit errors.
    Waits 2s, 4s, 8s, 16s between attempts.
    Other exceptions propagate immediately.
    """
    kwargs: dict[str, Any] = {
        "model":       MODEL,
        "messages":    messages,
        "temperature": 0.3,
    }
    if tools is not None:
        kwargs["tools"]       = tools
        kwargs["tool_choice"] = tool_choice

    last_exc: Exception | None = None
    for attempt in range(MAX_BACKOFF_ATTEMPTS):
        try:
            return client.chat.completions.create(**kwargs)
        except Exception as e:
            last_exc = e
            err = str(e).lower()
            if "429" in str(e) or "rate_limit" in err or "too many" in err:
                wait = (2 ** attempt) * 2   # 2s, 4s, 8s, 16s
                if DISPLAY_LOGS:
                    print(f"[rate limit] waiting {wait}s (retry {attempt + 1}/{MAX_BACKOFF_ATTEMPTS})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(
        f"Groq rate limit exceeded after {MAX_BACKOFF_ATTEMPTS} retries"
    ) from last_exc


# ══════════════════════════════════════════════════════════════════════════════
#  REACT LOOP DRIVER
# ══════════════════════════════════════════════════════════════════════════════

def _run_react_loop(
    messages:       list[dict],
    trace:          list[dict],
    max_iterations: int = MAX_ITERATIONS,
    registry:       dict[str, Any] | None = None,
    tools_schema:   list[dict] | None = None,
    stop_tool:      str = "add_track_to_queue",
    log_prefix:     str = "react",
) -> dict:
    """
    Drive a Reason→Act→Observe loop until `stop_tool` fires successfully
    or `max_iterations` is reached.

    On each iteration:
      1. Call Groq with messages + tools.
      2. If the assistant produced reasoning text, append a 'think' trace entry.
      3. Append the assistant turn to messages (so its tool-call envelopes
         are visible to the next iteration).
      4. If no tool calls, the assistant has given its final answer — record
         an 'explain' entry and return.
      5. Otherwise execute each tool call, log an 'act' entry per call,
         and append the result as a 'tool' message.
      6. If `stop_tool` succeeded, ask Groq once more (tools disabled) for
         its plain-language explanation and return.

    Returns: {"queued": <stop_tool_result_or_None>, "explanation": <str>}
    """
    if registry is None:
        registry = TOOL_REGISTRY
    if tools_schema is None:
        tools_schema = GROQ_TOOLS

    terminal:    dict | None = None
    explanation: str         = ""
    step:        int         = 1

    for iteration in range(max_iterations):
        # ── 1. Ask the model ──────────────────────────────────
        response = _groq_call(messages, tools=tools_schema, tool_choice="auto")
        msg      = response.choices[0].message
        reasoning = (msg.content or "").strip()

        # ── 2. Log reasoning text as a 'think' entry ──────────
        if reasoning:
            trace.append(_trace_entry(step, "think", reasoning))
            if DISPLAY_LOGS:
                print(f"\n[{log_prefix}/think] {reasoning[:240]}")
            step += 1

        # ── 3. Append assistant turn to the message history ───
        assistant_msg: dict = {"role": "assistant", "content": msg.content or ""}
        if msg.tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id":   tc.id,
                    "type": "function",
                    "function": {
                        "name":      tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_msg)

        # ── 4. No tool calls → final text response ────────────
        if not msg.tool_calls:
            explanation = reasoning
            if explanation:
                trace.append(_trace_entry(step, "explain", explanation))
                step += 1
            break

        # ── 5. Execute each tool call ─────────────────────────
        stop_fired = False
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}

            if DISPLAY_LOGS:
                print(f"[{log_prefix}/act ] → {name}({json.dumps(args)[:140]})")

            if name in registry:
                try:
                    result = registry[name](**args)
                except Exception as exc:
                    result = {"error": f"Tool '{name}' raised: {exc}"}
            else:
                result = {"error": f"Unknown tool: {name}"}

            if DISPLAY_LOGS:
                print(f"[{log_prefix}/obs ] ← {str(result)[:200]}")

            trace.append(_trace_entry(
                step       = step,
                kind       = "act",
                content    = name,
                tool_name  = name,
                tool_args  = args,
                tool_result= result,
            ))
            step += 1

            messages.append({
                "role":         "tool",
                "tool_call_id": tool_call.id,
                "content":      json.dumps(result),
            })

            if (
                name == stop_tool
                and isinstance(result, dict)
                and result.get("success")
            ):
                terminal   = result
                stop_fired = True

        # ── 6. If the stop tool fired, get the explanation ────
        if stop_fired:
            try:
                final = _groq_call(messages, tools=None)
                explanation = (final.choices[0].message.content or "").strip()
                if explanation:
                    trace.append(_trace_entry(step, "explain", explanation))
                    step += 1
            except Exception:
                # Empty explanation — the public-facing function will
                # synthesise a fallback string from the queued track.
                pass
            break

    return {"queued": terminal, "explanation": explanation}


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
    The LLM loop is given CYCLE_TOOLS (no search tools) so it can only
    verify harmonic fit and commit — no iteration budget wasted on search.
    If the pre-step finds nothing, falls back to full GROQ_TOOLS.

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
    tags  = _state_to_tags(state)

    if verbose:
        print(f"\n[pre-fetch] Last.fm tags: {tags}")

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

    shortlist = _best_fallback_top_n(raw_candidates, state, n=3)

    trace.append(_trace_entry(
        step       = 0,
        kind       = "act",
        content    = f"Pre-fetch: tags={tags} → {len(raw_candidates)} candidates → shortlist of {len(shortlist)}",
        tool_name  = "search_tracks_by_tag",
        tool_args  = {"tags": tags},
        tool_result= {"count": len(raw_candidates), "shortlist_size": len(shortlist)},
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
        candidate_section = (
            f"\n\nPre-scored candidates from Last.fm tag search {tags}"
            " (ranked best-first by energy/valence fit):\n"
            + "\n".join(lines)
            + "\n\nVerify harmonic fit of candidate #1, then commit with add_track_to_queue."
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
        # Reuse the pre-fetched candidates; only re-search if there were none
        candidates = raw_candidates or tool_module.search_tracks(query, limit=8).get("candidates", [])
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
    selected: dict | None = None
    if confident and search_queries:
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
    Single structured-JSON Groq call. Returns the parsed vibe dict on
    success, None on parse failure. Records a 'think' trace entry either
    way so the bootstrap reasoning is visible to post-hoc analysis.
    """
    user_msg = SESSION_INTERPRET_USER.format(description=description)

    try:
        response = _groq_call(
            messages = [
                {"role": "system", "content": SESSION_INTERPRET_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            tools = None,
        )
        raw = (response.choices[0].message.content or "").strip()

        # Strip markdown fences if the model added them despite the system prompt
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())

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
