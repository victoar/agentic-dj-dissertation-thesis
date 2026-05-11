"""
bridge.py — connects the Streamlit UI to the agent tools.

Handles:
  - Importing the agent package (src layout path fix)
  - Session state initialisation
  - Adapters that map tool output to component field names
  - Feedback handler that runs the agent cycle and refreshes state
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st

try:
    import agentic_dj.agent.tools as tool_module
    from agentic_dj.agent.loop import run_agent_cycle
    _agent_available = True
except Exception:
    _agent_available = False


def init_session() -> None:
    """Initialise session state keys on first load."""
    if "initialised" in st.session_state:
        return
    if not _agent_available:
        st.session_state.initialised = False
        return
    try:
        st.session_state.current_playback = tool_module.get_current_playback()
        st.session_state.listener_state   = tool_module.get_listener_state()
        st.session_state.queue_state      = tool_module.get_queue_state()
        st.session_state.session_history  = tool_module.get_session_history()
        st.session_state.last_trace       = []
        st.session_state.last_explanation = ""
        st.session_state.initialised      = True
    except Exception:
        st.session_state.initialised = False


def refresh() -> None:
    """Re-fetch all tool data and store in session state."""
    st.session_state.current_playback = tool_module.get_current_playback()
    st.session_state.listener_state   = tool_module.get_listener_state()
    st.session_state.queue_state      = tool_module.get_queue_state()
    st.session_state.session_history  = tool_module.get_session_history()


def refresh_playback() -> None:
    """Lightweight refresh — only current playback. Used by the polling fragment."""
    st.session_state.current_playback = tool_module.get_current_playback()


def adapt_now_playing() -> dict:
    """Map tool outputs to the fields now_playing.render() expects."""
    playback    = st.session_state.current_playback
    history     = st.session_state.session_history
    explanation = st.session_state.last_explanation

    name   = playback.get("track_name", "")
    artist = playback.get("artist", "")

    energy_est, valence_est, bpm, camelot = 0.5, 0.5, None, None
    for entry in history.get("recent", []):
        if entry.get("name", "").lower() == name.lower():
            energy_est  = entry.get("energy_est",  0.5)
            valence_est = entry.get("valence_est", 0.5)
            bpm         = entry.get("bpm")
            camelot     = entry.get("camelot_position")
            break

    return {
        "track_name":  name,
        "artist":      artist,
        "album":       playback.get("album", ""),
        "energy_est":  energy_est,
        "valence_est": valence_est,
        "bpm":         bpm,
        "key":         camelot,
        "progress_ms": playback.get("progress_ms", 0),
        "duration_ms": playback.get("duration_ms", 1),
        "reasoning":   explanation or "Waiting for first agent cycle…",
    }


def adapt_trace() -> list:
    """Convert raw agent trace entries to the format agent_trace.render() expects."""
    raw = st.session_state.get("last_trace", [])
    adapted = []
    for entry in raw:
        kind = entry.get("kind", "act")
        if kind == "explain":
            kind = "think"
        content = entry.get("content", "")
        # For act entries, append the tool name to the content when available
        tool = entry.get("tool_name")
        if entry.get("kind") == "act" and tool and tool not in content:
            content = f"{tool}\n{content}" if content else tool
        adapted.append({"kind": kind, "content": content})
    return adapted


def adapt_queue() -> tuple[dict, list]:
    """Map tool outputs to the fields queue.render() expects."""
    playback = st.session_state.current_playback
    history  = {
        e["name"].lower(): e
        for e in st.session_state.session_history.get("recent", [])
    }

    now_playing = {
        "name":   playback.get("track_name", ""),
        "artist": playback.get("artist", ""),
        "bpm":    None,
        "key":    None,
    }

    queue_items = []
    for item in st.session_state.queue_state.get("upcoming", []):
        hist_entry = history.get(item["name"].lower(), {})
        queue_items.append({
            "name":   item["name"],
            "artist": item["artist"],
            "bpm":    hist_entry.get("bpm") or None,
            "key":    hist_entry.get("camelot_position") or None,
            "energy": hist_entry.get("energy_est") or None,
            "note":   "",
        })

    return now_playing, queue_items


def handle_feedback(event: str) -> None:
    """Run the agent cycle for a feedback event, then refresh all state."""
    playback   = st.session_state.current_playback
    track_name = playback.get("track_name", "")
    artist     = playback.get("artist", "")

    with st.spinner("Agent selecting next track…"):
        result = run_agent_cycle(
            feedback_event=event,
            feedback_track=track_name,
            feedback_artist=artist,
            verbose=False,
        )

    refresh()
    st.session_state.last_trace       = result.get("trace", [])
    st.session_state.last_explanation = result.get("explanation", "")
    st.rerun()
