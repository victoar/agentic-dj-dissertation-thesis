"""
ReAct agent loop — the reasoning core of the Agentic DJ system.

The loop follows the ReAct pattern (Reasoning + Acting):
  1. Observe  — read current state, playback, session history
  2. Think    — LLM reasons about what to do next
  3. Act      — call a tool
  4. Observe  — read the tool result
  5. Repeat   — until the LLM decides it has enough to make a decision
  6. Respond  — generate a plain-language explanation for the listener

The LLM has access to all 12 tools defined in agent/tools.py.
It selects and sequences them autonomously based on the situation.
"""

import json
import os
import time
from typing import Any

from google import genai
from google.genai import types
from dotenv import load_dotenv

from agentic_dj.agent import tools as tool_module
from agentic_dj.music.camelot import parse as camelot_parse, compatibility_strength
from agentic_dj.music import lastfm_client

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

DISPLAY_LOGS: bool = True   # flip to True to see the full recommendation trace


# ── System prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the reasoning core of an Agentic DJ system.
Your job is to select the best next track for the listener and explain
your choice in one or two plain sentences.

You have access to tools that let you:
- Read the listener's current state (energy, valence, focus, openness)
- Check the session arc phase (warmup, build, peak, cooldown)
- Search for candidate tracks using Spotify + Last.fm enrichment
- Check harmonic compatibility between tracks (Camelot Wheel)
- Check BPM compatibility between tracks
- Add the chosen track to the playback queue

RULES you must always follow:
1. Always call get_listener_state first to orient your reasoning.
2. Always call get_session_arc to understand where in the session we are.
3. Always call get_current_playback to know what is currently playing.
4. Search for candidates using search_tracks before selecting.
5. Check harmonic compatibility using check_transition before queuing.
6. Check BPM compatibility using estimate_bpm_compatibility before queuing.
7. Only call add_track_to_queue once — for the single best candidate.
8. After queuing, generate a one or two sentence explanation for the listener.
   Reference at least one concrete signal (a skip, replay, the arc phase)
   and at least one musical property (key compatibility, BPM, energy level).
9. Never select a track already in recent session history.
10. If no compatible track is found, explain why and suggest the closest option.
"""


# ── Tool definitions for Gemini ───────────────────────────────────────────────
# Gemini uses a function declaration schema to understand what tools exist
# and what parameters they take.

TOOL_DECLARATIONS = [
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
        "description": "Return the current arc phase and description of what it means for track selection.",
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
        "description": "Check harmonic smoothness between two Camelot positions. Score 0.0-1.0.",
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
        "description": "Check if a BPM jump between two tracks is smooth for the current arc phase.",
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
        "name": "search_tracks",
        "description": "Search Spotify for candidate tracks enriched with Last.fm tags and energy/valence estimates.",
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
        "name": "get_track_details",
        "description": "Get full details for a specific track — metadata, tags, energy, valence.",
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
        "description": "Add a track to the Spotify playback queue. Call this once with the single best candidate.",
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

# Map tool name strings to actual Python functions
TOOL_REGISTRY: dict[str, Any] = {
    "get_listener_state":        tool_module.get_listener_state,
    "update_listener_state":     tool_module.update_listener_state,
    "get_session_arc":           tool_module.get_session_arc,
    "get_compatible_keys":       tool_module.get_compatible_keys,
    "check_transition":          tool_module.check_transition,
    "estimate_bpm_compatibility":tool_module.estimate_bpm_compatibility,
    "search_tracks":             tool_module.search_tracks,
    "get_track_details":         tool_module.get_track_details,
    "get_current_playback":      tool_module.get_current_playback,
    "get_queue_state":           tool_module.get_queue_state,
    "get_session_history":       tool_module.get_session_history,
    "add_track_to_queue":        tool_module.add_track_to_queue,
}


# ── Trace entry ───────────────────────────────────────────────────────────────

def _make_trace_entry(
    step:        int,
    kind:        str,   # "think" | "act" | "observe" | "explain"
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


# ── Main agent loop ───────────────────────────────────────────────────────────

def _call_gemini_with_retry(
    prompt:      str,
    system:      str,
    max_retries: int = 4,
) -> str:
    """
    Call Gemini with exponential backoff on rate limit errors.
    Returns the raw text response.
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system,
                    temperature=0.3,
                )
            )
            return response.text.strip()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                wait = (2 ** attempt) * 12
                print(f"\n[rate limit] Waiting {wait}s before retry {attempt + 1}/{max_retries}...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini rate limit exceeded after all retries.")

def _best_fallback(
    candidates: list[dict],
    state:      dict,
) -> dict:
    """
    When Gemini JSON parsing fails, pick the most compatible candidate
    based on energy and valence distance from the current state rather
    than just taking the first result.
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
        print("\n[fallback] Scored candidates:")
        for c in scored[:6]:
            s = score(c)
            bpm_sym = "✓" if c.get("bpm_ok", True) else "✗"
            cam_sym = "✓" if c.get("camelot_ok", True) else "✗"
            print(f"  {s:.2f} — {c['name']} — {c['artist']}  (bpm={bpm_sym}  key={cam_sym})")
        print(f"  → Best: {scored[0]['name']} — {scored[0]['artist']}")

    return scored[0]

def run_agent_cycle(
    feedback_event:  str | None = None,
    feedback_track:  str | None = None,
    feedback_artist: str | None = None,
    verbose:         bool = True,
) -> dict:
    """
    Single-shot agent cycle — collect all context in Python, then make
    one Gemini call to reason and select the next track.
    """
    trace: list[dict] = []
    tool_module.DISPLAY_LOGS = DISPLAY_LOGS

    # ── Step 1: Apply feedback if provided ───────────────────
    if feedback_event and feedback_track and feedback_artist:
        result = tool_module.update_listener_state(
            feedback_event, feedback_track, feedback_artist
        )
        trace.append(_make_trace_entry(
            step=0, kind="act",
            content=f"Applied feedback: {feedback_event} on {feedback_track}",
            tool_name="update_listener_state",
            tool_args={"event": feedback_event, "track_name": feedback_track, "artist": feedback_artist},
            tool_result=result,
        ))
        if verbose:
            print(f"\n[feedback] {feedback_event} on '{feedback_track}'")

    # ── Step 2: Collect all context in Python ─────────────────
    if verbose:
        print("\n[observe] Collecting session context...")

    state   = tool_module.get_listener_state()
    arc     = tool_module.get_session_arc()
    playback= tool_module.get_current_playback()
    history = tool_module.get_session_history()
    queue   = tool_module.get_queue_state()

    # Record each context call individually in the trace
    for tool_name, result in [
        ("get_listener_state",  state),
        ("get_session_arc",     arc),
        ("get_current_playback",playback),
        ("get_session_history", history),
        ("get_queue_state",     queue),
    ]:
        trace.append(_make_trace_entry(
            step=1, kind="act",
            content=f"Collected {tool_name}",
            tool_name=tool_name,
            tool_args={},
            tool_result=result,
        ))

    # ── Step 3: Search for candidates ────────────────────────
    if verbose:
        print("[observe] Searching for candidates...")

    current_track_id = playback.get("track_id", "")
    current_artist   = playback.get("artist", "").lower()

    # Use Last.fm similar artists as search seeds for stylistically relevant results.
    # Fall back to a mood query if the current artist is unknown or Last.fm returns nothing.
    similar_artists = lastfm_client.get_similar_artists(current_artist, limit=5) if current_artist else []

    if similar_artists:
        searches = [tool_module.search_tracks(f"artist:{a}", limit=2) for a in similar_artists]
        query_desc = f"similar artists: {', '.join(similar_artists)}"
    else:
        energy_word  = "energetic upbeat" if state["energy"] > 0.6 else "calm relaxing"
        valence_word = "happy positive"   if state["valence"] > 0.6 else "melancholic dark"
        searches     = [tool_module.search_tracks(f"{energy_word} {valence_word}", limit=8)]
        query_desc   = f"{energy_word} {valence_word}"

    all_candidates = []
    for s in searches:
        all_candidates.extend(s.get("candidates", []))

    # Deduplicate by track id
    seen = set()
    candidates = []
    for c in all_candidates:
        if c["id"] not in seen:
            seen.add(c["id"])
            candidates.append(c)

    # Exclude the currently playing track and all tracks by the same artist
    if current_artist:
        candidates = [
            c for c in candidates
            if c["id"] != current_track_id
            and c["artist"].lower() != current_artist
        ]

    trace.append(_make_trace_entry(step=2, kind="observe",
        content=f"Found {len(candidates)} candidates",
        tool_result={"count": len(candidates), "query": query_desc}))

    if verbose:
        print(f"[observe] Found {len(candidates)} candidates")

    if DISPLAY_LOGS:
        print(f"\n[candidates] {len(candidates)} tracks found:")
        for i, c in enumerate(candidates, 1):
            bpm = c.get("bpm") or "—"
            key = c.get("camelot_position") or "—"
            print(f"  {i:2}. {c['name']} — {c['artist']}  (BPM={bpm}  Key={key})")

    # ── Step 4: Check compatibility for each candidate ───────
    if verbose:
        print("[observe] Checking compatibility...")

    hist_items   = history.get("recent", [])
    current_bpm  = hist_items[0].get("bpm") if hist_items else None
    current_camelot = hist_items[0].get("camelot_position") if hist_items else None

    if DISPLAY_LOGS:
        print(f"\n[compare loop] current track BPM={current_bpm or '—'}  Key={current_camelot or '—'}")

    # Add compatibility scores to each candidate
    enriched_candidates = []
    for c in candidates[:8]:   # limit to top 8
        enriched = dict(c)

        # BPM compatibility
        if current_bpm and c.get("bpm"):
            bpm_check = tool_module.estimate_bpm_compatibility(
                current_bpm, c["bpm"], arc["arc_phase"]
            )
            enriched["bpm_ok"]   = bpm_check["acceptable"]
            enriched["bpm_diff"] = bpm_check["difference"]
        else:
            enriched["bpm_ok"]   = True
            enriched["bpm_diff"] = 0

        # Camelot (harmonic) compatibility
        current_key   = camelot_parse(current_camelot or "")
        candidate_key = camelot_parse(c.get("camelot_position") or "")
        if current_key and candidate_key:
            score = compatibility_strength(current_key, candidate_key)
            enriched["camelot_ok"]    = score >= 0.4
            enriched["camelot_score"] = round(score, 2)
        else:
            enriched["camelot_ok"]    = True   # unknown key — no penalty
            enriched["camelot_score"] = None

        if DISPLAY_LOGS:
            bpm_sym = "✓" if enriched["bpm_ok"] else "✗"
            cam_sym = "✓" if enriched["camelot_ok"] else "✗"
            print(f"\n  [compare] {c['name']} — {c['artist']}")
            if current_bpm and c.get("bpm"):
                print(f"    BPM    : {current_bpm} → {c['bpm']}  diff={enriched['bpm_diff']}  ok={bpm_sym}")
            else:
                print(f"    BPM    : unknown (no penalty)")
            if enriched.get("camelot_score") is not None:
                print(f"    Key    : {current_camelot} → {c.get('camelot_position')}  "
                      f"score={enriched['camelot_score']}  ok={cam_sym}")
            else:
                print(f"    Key    : unknown (no penalty)")
            e_dist = abs(c.get("energy_est", 0.5) - state.get("energy", 0.5))
            v_dist = abs(c.get("valence_est", 0.5) - state.get("valence", 0.5))
            print(f"    Energy : |{state.get('energy', 0.5):.2f} - {c.get('energy_est', 0.5):.2f}| = {e_dist:.2f}")
            print(f"    Valence: |{state.get('valence', 0.5):.2f} - {c.get('valence_est', 0.5):.2f}| = {v_dist:.2f}")

        enriched_candidates.append(enriched)

    trace.append(_make_trace_entry(step=3, kind="observe",
        content="Compatibility checked for all candidates",
        tool_result={"candidates_checked": len(enriched_candidates)}))

    # ── Step 5: Single Gemini call to reason and select ──────
    if verbose:
        print("[think] Asking Gemini to select the best track...")

    # Apply name-based exclusion as a second layer for robustness.
    played_names = tool_module._queued_names   # direct access to the set

    fresh_candidates = [
        c for c in enriched_candidates
        if c.get("name", "").lower() not in played_names
    ]

    if not fresh_candidates:
        fresh_candidates = enriched_candidates   # nothing new — rare edge case

    # Build a compact prompt with all context
    prompt = f"""You are an Agentic DJ. Select the best next track.


LISTENER STATE:
{json.dumps(state, indent=2)}

SESSION ARC:
{json.dumps(arc, indent=2)}

CURRENT PLAYBACK:
{json.dumps(playback, indent=2)}

CANDIDATE TRACKS (already enriched with tags and compatibility):
{json.dumps(fresh_candidates, indent=2)}

INSTRUCTIONS:
1. Select the single best track from the candidates above.
2. Consider: energy match, valence match, arc phase, BPM compatibility.
3. Avoid tracks with bpm_ok=False unless no alternative exists.
4. Respond with valid JSON only — no markdown, no extra text:

{{
  "selected_name": "<exact track name from candidates>",
  "selected_artist": "<exact artist from candidates>",
  "reasoning": "<2-3 sentences explaining the choice, mentioning at least one musical property>",
  "explanation_for_listener": "<1-2 plain sentences for the listener, no jargon>"
}}"""

    try:
        raw = _call_gemini_with_retry(
            prompt=prompt,
            system="You are an expert DJ assistant. Always respond with valid JSON only.",
        )
        raw         = raw.replace("```json", "").replace("```", "").strip()
        selection   = json.loads(raw)

        # Validate the selection refers to an actual candidate
        valid_names = {c["name"].lower() for c in fresh_candidates}
        if selection.get("selected_name", "").lower() not in valid_names:
            raise ValueError(f"Gemini selected unknown track: {selection.get('selected_name')}")

        if DISPLAY_LOGS:
            print(f"\n[selected] Gemini → \"{selection['selected_name']}\" by {selection['selected_artist']}")
            print(f"  Reasoning: {selection.get('reasoning', '')[:200]}")

    except Exception as e:
        if verbose or DISPLAY_LOGS:
            print(f"[fallback] Gemini parse failed ({e}) — using scored fallback")
        best = _best_fallback(fresh_candidates, state)
        if not best:
            return {
                "explanation":  "Could not find a suitable track.",
                "queued_track": None,
                "trace":        trace,
                "steps":        4,
                "success":      False,
            }
        selection = {
            "selected_name":            best["name"],
            "selected_artist":          best["artist"],
            "reasoning":                f"Fallback selection — closest energy/valence match to current state.",
            "explanation_for_listener": f"Up next: {best['name']} by {best['artist']}.",
        }

    trace.append(_make_trace_entry(step=4, kind="think",
        content=selection.get("reasoning", ""),
        tool_result=selection))

    if verbose:
        print(f"\n[think] {selection.get('reasoning', '')[:200]}")

    # ── Step 7: Queue the selected track ─────────────────────
    queue_result = tool_module.add_track_to_queue(
        selection["selected_name"],
        selection["selected_artist"],
    )

    if not queue_result.get("success") and queue_result.get("duplicate"):
        if verbose:
            print(f"[retry] '{selection['selected_name']}' already played "
                  f"— trying next best candidate")

        remaining = sorted(
            [
                c for c in fresh_candidates
                if c.get("name", "").lower() != selection["selected_name"].lower()
                and c.get("name", "").lower() not in tool_module._queued_names
            ],
            key=lambda c: (
                abs(c.get("energy_est",  0.5) - state["energy"]) +
                abs(c.get("valence_est", 0.5) - state["valence"])
            ),
        )

        for alt in remaining:
            queue_result = tool_module.add_track_to_queue(
                alt["name"], alt["artist"]
            )
            if queue_result.get("success"):
                selection["selected_name"]            = alt["name"]
                selection["selected_artist"]          = alt["artist"]
                selection["explanation_for_listener"] = (
                    f"Up next: {alt['name']} by {alt['artist']}."
                )
                break

    explanation = selection.get("explanation_for_listener", "")

    trace.append(_make_trace_entry(
        step=5, kind="act",
        content=f"Queued {selection['selected_name']}",
        tool_name="add_track_to_queue",
        tool_args={
            "track_name": selection["selected_name"],
            "artist":     selection["selected_artist"],
        },
        tool_result=queue_result,
    ))

    if verbose:
        print(f"\n[explain] {explanation}")
        print(f"\n{'='*55}")
        success = queue_result.get("success", False)
        print(f"Cycle complete — {'success' if success else 'no active device'}")
        if queue_result.get("queued"):
            q = queue_result["queued"]
            print(f"Queued: {q.get('name')} — {q.get('artist')}")
        print(f"{'='*55}\n")

    return {
        "explanation":  explanation,
        "queued_track": queue_result.get("queued"),
        "trace":        trace,
        "steps":        6,
        "success":      queue_result.get("success", False),
    }