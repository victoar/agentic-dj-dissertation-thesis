"""
bridge.py — connects the Streamlit UI to the agent tools.

Handles:
  - Importing the agent package (src layout path fix)
  - Session state initialisation
  - Adapters that map tool output to component field names
  - Feedback handler that runs the agent cycle and refreshes state
  - Session logging for Chapter 6 evaluation (§6.4)
"""

import atexit
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import streamlit as st

try:
    import agentic_dj.agent.tools as tool_module
    from agentic_dj.agent.loop import run_agent_cycle, start_session
    _agent_available = True
except Exception:
    _agent_available = False


# ── Session-log accumulator (module-level — persists across Streamlit reruns) ─

_LOG_DIR = Path(__file__).resolve().parent.parent / "chapter6_schemas" / "session_logs"

# Mutable module-level state for the current evaluation session
_eval: dict = {
    "participant_id": "",
    "condition":      "",          # "agenticdj" | "spotify"
    "session_label":  "",
    "started_at":     "",
    "cycles":         [],          # list of per-cycle dicts {trace, queued_track}
}


def init_eval_logging(participant_id: str, condition: str) -> None:
    """Call once when participant + condition are known (from the UI form)."""
    _eval["participant_id"] = participant_id.strip()
    _eval["condition"]      = condition.strip()
    _eval["started_at"]     = datetime.now().isoformat(timespec="seconds")
    _eval["cycles"]         = []


def _accumulate_cycle(result: dict) -> None:
    """Append one agent-cycle result to the module-level accumulator."""
    _eval["cycles"].append({
        "trace":        result.get("trace", []),
        "queued_track": result.get("queued_track"),
        "explanation":  result.get("explanation", ""),
        "success":      result.get("success", False),
    })


# ── Metrics derivation ────────────────────────────────────────────────────────

_ARC_TARGET   = {"warmup": (0.3, 0.5), "build": (0.5, 0.7),
                 "peak": (0.8, 1.0), "cooldown": (0.2, 0.4)}
_ARC_THRESHOLDS = {
    "warmup":   4,
    "build":    12,   # warmup(4) + build(8)
    "peak":     16,   # warmup(4) + build(8) + peak(4)
}


def _phase_for_position(pos: int) -> str:
    """Return the expected arc phase for the N-th track (1-indexed)."""
    if pos <= _ARC_THRESHOLDS["warmup"]:
        return "warmup"
    if pos <= _ARC_THRESHOLDS["build"]:
        return "build"
    if pos <= _ARC_THRESHOLDS["peak"]:
        return "peak"
    return "cooldown"


def _compute_metrics(all_traces: list[list[dict]]) -> dict:
    """Derive the four §6.4 trace features from the accumulated cycle traces."""
    n_state_updates        = 0
    n_constraint_violations = 0

    flat = [entry for cycle in all_traces for entry in cycle]

    for entry in flat:
        tool = entry.get("tool_name") or ""
        if tool == "update_listener_state":
            n_state_updates += 1
        if tool == "check_transition":
            result = entry.get("tool_result") or {}
            if isinstance(result.get("score"), (int, float)) and result["score"] < 0.4:
                n_constraint_violations += 1

    # Arc deviation — compare each committed track's energy to its expected band
    history = []
    if _agent_available:
        try:
            history = tool_module.get_session_history().get("recent", [])
        except Exception:
            pass

    deviations = []
    for i, track in enumerate(reversed(history), start=1):  # reversed → chronological
        phase = _phase_for_position(i)
        lo, hi = _ARC_TARGET[phase]
        e = track.get("energy_est", 0.5)
        dist = max(0.0, lo - e, e - hi)
        deviations.append(dist)

    arc_deviation       = round(sum(deviations) / len(deviations), 4) if deviations else 0.0
    session_length_tracks = len(history)

    return {
        "session_length_tracks":  session_length_tracks,
        "n_state_updates":         n_state_updates,
        "n_constraint_violations": n_constraint_violations,
        "arc_deviation":           arc_deviation,
    }


# ── Save / load ───────────────────────────────────────────────────────────────

def save_session_log(force: bool = False) -> Path | None:
    """
    Compute metrics from accumulated traces and write a JSON log to
    chapter6_schemas/session_logs/.

    Returns the saved file path, or None if skipped (no cycles recorded
    and force=False).
    """
    if not _eval["cycles"] and not force:
        return None

    all_traces = [c["trace"] for c in _eval["cycles"]]
    metrics    = _compute_metrics(all_traces)

    pid   = _eval["participant_id"] or "unknown"
    cond  = _eval["condition"]      or "unknown"
    ts    = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = f"{pid}_{cond}_{ts}.json"

    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _LOG_DIR / fname

    payload = {
        "session_id":     f"{pid}_{cond}_{ts}",
        "participant_id": pid,
        "condition":      cond,
        "session_label":  _eval.get("session_label", ""),
        "started_at":     _eval.get("started_at", ""),
        "saved_at":       datetime.now().isoformat(timespec="seconds"),
        "metrics":        metrics,
        "history":        [],
        "cycles":         _eval["cycles"],
    }

    # Attach full session history
    if _agent_available:
        try:
            payload["history"] = tool_module.get_session_history().get("recent", [])
        except Exception:
            pass

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)

    return out_path


def _atexit_save() -> None:
    """Auto-save on process exit (handles Ctrl+C / Option+C cleanly)."""
    try:
        path = save_session_log(force=False)
        if path:
            print(f"\n[session-log] auto-saved → {path}")
    except Exception as e:
        print(f"\n[session-log] auto-save failed: {e}")


atexit.register(_atexit_save)


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
        st.session_state.session_label    = ""
        st.session_state.start_status     = "idle"
        st.session_state.start_error      = ""
        st.session_state.initialised      = True
    except Exception as e:
        st.session_state.initialised = False
        st.session_state.init_error = str(e)
        return


def refresh() -> None:
    """Re-fetch all tool data and store in session state."""
    st.session_state.current_playback = tool_module.get_current_playback()
    st.session_state.listener_state   = tool_module.get_listener_state()
    st.session_state.queue_state      = tool_module.get_queue_state()
    st.session_state.session_history  = tool_module.get_session_history()


def refresh_playback() -> None:
    """Lightweight refresh — only current playback. Used by the polling fragment."""
    st.session_state.current_playback = tool_module.get_current_playback()


def ensure_buffer(target: int = 2) -> None:
    """Run agent cycles until `target` tracks are buffered ahead in Spotify's queue."""
    attempts = 0
    while tool_module.get_lookahead_depth() < target and attempts < target:
        result = run_agent_cycle(verbose=False)
        attempts += 1
        if not result.get("success"):
            break


def detect_and_handle_track_change() -> bool:
    """
    Refresh playback and compare the new track ID to the previously known one.
    If changed, consume the lookahead slot (handles out-of-band skips too).
    Returns True if the track changed, False otherwise.
    """
    prev_id = st.session_state.current_playback.get("track_id", "")
    refresh_playback()
    new_id = st.session_state.current_playback.get("track_id", "")
    if new_id and new_id != prev_id:
        tool_module.consume_lookahead_up_to(new_id)
        return True
    return False


def start_session_from_description(description: str) -> dict:
    """Interpret a natural language description, start playback, and fill the buffer."""
    result = start_session(description, verbose=False)
    if result.get("success"):
        label = result.get("session_label", "")
        st.session_state.session_label    = label
        st.session_state.start_status     = "idle"
        st.session_state.start_error      = ""
        # Surface the opener's reasoning/trace so the Now Playing + Agent Trace
        # tabs show why the first track was chosen (handle_feedback does the same).
        st.session_state.last_explanation = result.get("explanation", "")
        st.session_state.last_trace       = result.get("trace", [])
        # Accumulate for session log
        _eval["session_label"] = label
        _accumulate_cycle(result)
        refresh()
        ensure_buffer(2)
    else:
        error = result.get("error", "unknown")
        if error == "no_device":
            st.session_state.start_error = (
                "No active Spotify device found. "
                "Open Spotify on any device and try again."
            )
        else:
            st.session_state.start_error = (
                "Couldn't find a track for that description. "
                "Try something more specific."
            )
        st.session_state.start_status = "error"
    return result


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
        "track_name":    name,
        "artist":        artist,
        "album":         playback.get("album", ""),
        "energy_est":    energy_est,
        "valence_est":   valence_est,
        "bpm":           bpm,
        "key":           camelot,
        "progress_ms":   playback.get("progress_ms", 0),
        "duration_ms":   playback.get("duration_ms", 1),
        "reasoning":     explanation or "Waiting for first agent cycle…",
        "session_label": st.session_state.get("session_label", ""),
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
    """Dispatch a Spotify playback action, then run the agent cycle to replenish the buffer."""
    playback   = st.session_state.current_playback
    track_name = playback.get("track_name", "")
    artist     = playback.get("artist", "")
    track_id   = playback.get("track_id", "")

    # Immediate Spotify playback action
    if event in ("skip", "full_listen"):
        tool_module._spotify.skip()
    elif event == "replay":
        tool_module._spotify.seek_to_beginning()
    elif event == "thumbs_up":
        tool_module._spotify.save_track(track_id)

    # Replay: state update only — no new track needed, track restarts in place
    if event == "replay":
        tool_module.update_listener_state("replay", track_name, artist)
        refresh()
        st.rerun()
        return

    # All other events: run agent cycle to record feedback and replenish the buffer
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
    _accumulate_cycle(result)
    st.rerun()
