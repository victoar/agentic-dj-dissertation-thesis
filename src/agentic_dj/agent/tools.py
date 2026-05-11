"""
Agent tool catalogue — the 12 functions the LLM can call during reasoning.

Each tool has:
  - A clear single responsibility
  - A docstring written for the LLM (this becomes the tool description)
  - Typed inputs and a structured return dict
  - No side effects beyond what is documented

Tools are grouped into four categories:
  1. State tools       — read and update the listener state vector
  2. Music theory tools — harmonic and tempo compatibility checks
  3. Spotify tools     — search and playback control
  4. Queue tools       — manage the upcoming track queue
"""

from dataclasses import asdict
from typing import Any

from agentic_dj.agent.state import (
    ListenerState, Track, FeedbackEvent,
    update_state, advance_track, init_state,
)
from agentic_dj.music.camelot import (
    from_spotify as camelot_from_spotify,
    parse as camelot_parse,
    compatible_positions,
    compatibility_strength,
)
from agentic_dj.music.tags import estimate_features_with_fallback
from agentic_dj.music.lastfm_client import enrich_track, fetch_enrichment
from agentic_dj.music.getsongkey_client import fetch_track_info
from agentic_dj.spotify.client import SpotifyClient, SpotifyTrack


# ── Shared instances (one per session) ──────────────────────────────────────
_spotify = SpotifyClient()
_queue:   list[SpotifyTrack] = []
_state:   ListenerState      = init_state("general")
_history: list[dict]         = []   # every track played this session

DISPLAY_LOGS: bool = False   # synced from loop.py at the start of each cycle


# ════════════════════════════════════════════════════════════════════════════
#  1. STATE TOOLS
# ════════════════════════════════════════════════════════════════════════════

def get_listener_state() -> dict:
    """
    Return the current listener state vector.

    Returns all six dimensions (energy, valence, focus, openness, social,
    arc_phase) plus the number of tracks played so far this session.
    Call this at the start of every reasoning cycle to orient decisions.
    """
    return _state.to_dict()


def update_listener_state(event: str, track_name: str, artist: str) -> dict:
    """
    Update the listener state based on a feedback event for the given track.

    Args:
        event: one of skip, early_skip, replay, full_listen,
               partial_listen, thumbs_up, thumbs_down
        track_name: name of the track the event applies to
        artist: artist of the track

    Returns the updated state vector.
    """
    global _state

    try:
        fb = FeedbackEvent(event)
    except ValueError:
        return {
            "error": f"Unknown event '{event}'. Valid events: "
                     f"{[e.value for e in FeedbackEvent]}",
            "state": _state.to_dict(),
        }

    # Find the track in history so we have energy/valence estimates
    track = _find_in_history(track_name, artist)
    if track is None:
        # Fall back to a neutral track if not found
        track = Track(id="unknown", name=track_name, artist=artist)

    _state = update_state(_state, fb, track)
    return {"updated": True, "state": _state.to_dict()}


def get_session_arc() -> dict:
    """
    Return the current arc phase and a plain-language description of what
    that means for track selection.

    Use this to decide whether to raise, maintain, or lower energy.
    """
    descriptions = {
        "warmup":   "Session just started. Choose moderate-energy tracks to "
                    "ease the listener in. Avoid extreme high or low energy.",
        "build":    "Energy should be rising. Each track can be slightly more "
                    "intense than the last. This is the growth phase.",
        "peak":     "Maximum energy. Choose the most powerful, high-energy "
                    "tracks now. This is the climax of the session.",
        "cooldown": "Session winding down. Gradually reduce energy and tempo. "
                    "Familiar and comforting tracks work well here.",
    }
    phase = _state.arc_phase.value
    return {
        "arc_phase":   phase,
        "description": descriptions.get(phase, ""),
        "tracks_played": _state._tracks_played,
    }


# ════════════════════════════════════════════════════════════════════════════
#  2. MUSIC THEORY TOOLS
# ════════════════════════════════════════════════════════════════════════════

def get_compatible_keys(camelot_position: str) -> dict:
    """
    Return all Camelot Wheel positions that form a harmonically smooth
    transition from the given position.

    Args:
        camelot_position: a Camelot position string e.g. '8B', '4A', '12B'

    Returns a list of compatible positions and their human-readable key names.
    Use this to filter candidate tracks before evaluating them.
    """
    from agentic_dj.music.camelot import CAMELOT_TO_NAME

    k = camelot_parse(camelot_position)
    if k is None:
        return {"error": f"Invalid Camelot position: '{camelot_position}'"}

    positions = compatible_positions(k)
    return {
        "current":    camelot_position,
        "compatible": [
            {"position": p, "key_name": CAMELOT_TO_NAME.get(p, "unknown")}
            for p in positions
        ],
    }


def check_transition(
    from_camelot: str,
    to_camelot:   str,
) -> dict:
    """
    Check how smooth a harmonic transition is between two tracks.

    Args:
        from_camelot: Camelot position of the current track e.g. '8B'
        to_camelot:   Camelot position of the candidate track

    Returns a smoothness score (0.0–1.0) and a plain-language verdict.
    Score >= 0.75 = smooth, 0.4–0.74 = acceptable, < 0.4 = avoid.
    """
    k1 = camelot_parse(from_camelot)
    k2 = camelot_parse(to_camelot)

    if k1 is None or k2 is None:
        return {"error": "Invalid Camelot position — use format like '8B' or '4A'"}

    score = compatibility_strength(k1, k2)

    if score >= 0.75:
        verdict = "smooth — go ahead"
    elif score >= 0.4:
        verdict = "acceptable — slight tension but workable"
    else:
        verdict = "avoid — jarring transition"

    return {
        "from":    from_camelot,
        "to":      to_camelot,
        "score":   round(score, 2),
        "verdict": verdict,
    }


def estimate_bpm_compatibility(
    current_bpm: float,
    candidate_bpm: float,
    arc_phase: str = "build",
) -> dict:
    """
    Check whether a BPM transition is smooth given the current arc phase.
    Thresholds per phase:
      warmup   → max 10 BPM difference
      build    → max 15 BPM difference
      peak     → max 20 BPM difference
      cooldown → max 10 BPM difference
    """
    thresholds = {
        "warmup":   10.0,
        "build":    15.0,
        "peak":     20.0,
        "cooldown": 10.0,
    }

    threshold  = thresholds.get(arc_phase, 15.0)
    diff       = abs(current_bpm - candidate_bpm)
    acceptable = diff <= threshold
    direction  = (
        "up"   if candidate_bpm > current_bpm else
        "down" if candidate_bpm < current_bpm else
        "same"
    )

    return {
        "current_bpm":   current_bpm,
        "candidate_bpm": candidate_bpm,
        "difference":    round(diff, 1),
        "threshold":     threshold,
        "acceptable":    acceptable,
        "direction":     direction,
        "note": f"BPM {'within' if acceptable else 'exceeds'} threshold "
                f"of {threshold} for {arc_phase} phase",
    }


# ════════════════════════════════════════════════════════════════════════════
#  3. SPOTIFY + LAST.FM TOOLS
# ════════════════════════════════════════════════════════════════════════════

def search_tracks(
    query:      str,
    limit:      int  = 8,
    enrich:     bool = True,
) -> dict:
    """
    Search Spotify for tracks matching the query, then enrich each result
    with Last.fm tags and estimated energy/valence values.

    Args:
        query:  any search string — artist name, track name, genre, mood
        limit:  max results to return (default 8, max 10)
        enrich: if True, fetch Last.fm tags for each result (adds ~1s per
                track on first fetch, instant on cache hit)

    Returns a list of enriched track candidates ready for the agent to
    evaluate and rank.
    """
    limit   = min(limit, 10)
    results = _spotify.search(query, limit=limit)

    candidates = []
    for sp_track in results:
        # Skip tracks already played this session before enriching
        if sp_track.id in _queued_ids:
            continue
        if sp_track.name.lower() in _queued_names:
            continue
        candidate = _spotify_to_candidate(sp_track, enrich=enrich)
        candidates.append(candidate)

    return {
        "query":      query,
        "count":      len(candidates),
        "candidates": candidates,
    }


def get_track_details(track_name: str, artist: str) -> dict:
    """
    Get full details for a specific track — Spotify metadata plus
    Last.fm tags, energy estimate, valence estimate, and listener count.

    Use this when you already know which track you want and need its
    complete feature profile before adding it to the queue.
    """
    results = _spotify.search(f"{track_name} {artist}", limit=1)
    if not results:
        return {"error": f"Track not found: {track_name} by {artist}"}

    sp_track  = results[0]
    candidate = _spotify_to_candidate(sp_track, enrich=True)
    return candidate


def get_current_playback() -> dict:
    """
    Return what is currently playing on Spotify.

    Returns track name, artist, progress through the track, and whether
    Spotify is actively playing. Returns an empty dict if nothing is playing.
    Use this to ground your reasoning in what the listener is actually hearing.
    """
    state = _spotify.get_playback()
    if not state:
        return {"playing": False, "message": "No active Spotify device found."}

    result: dict[str, Any] = {
        "playing":     state.is_playing,
        "device":      state.device_name,
        "progress_ms": state.progress_ms,
        "progress_pct": 0,
    }

    if state.track:
        t = state.track
        result.update({
            "track_id":    t.id,
            "track_name":  t.name,
            "artist":      t.artist,
            "album":       t.album,
            "duration_ms": t.duration_ms,
            "progress_pct": round(
                state.progress_ms / t.duration_ms * 100
            ) if t.duration_ms else 0,
        })

    return result


# ════════════════════════════════════════════════════════════════════════════
#  4. QUEUE TOOLS
# ════════════════════════════════════════════════════════════════════════════

_queued_ids:   set[str] = set()    # track ids queued this session — never repeat
_queued_names: set[str] = set()    # lowercase names as backup when id is unavailable

def add_track_to_queue(track_name: str, artist: str) -> dict:
    """
    Search for a track and add it to the Spotify playback queue.

    This is the primary action the agent takes to schedule the next song.
    The track is also recorded in session history so future state updates
    can reference its features.

    Args:
        track_name: exact or approximate track title
        artist:     artist name

    Returns success status and the track that was queued.
    """
    global _queue, _history, _state

    results = _spotify.search(f"{track_name} {artist}", limit=1)
    if not results:
        return {"success": False, "error": f"Could not find '{track_name}' by {artist}"}

    sp_track = results[0]

    if sp_track.id in _queued_ids:
        return {
            "success":   False,
            "duplicate": True,
            "error":     f"'{sp_track.name}' by {sp_track.artist} was already "
                         f"played in this session — skipping.",
        }

    # Fallback name check for edge cases where id might differ
    if sp_track.name.lower() in _queued_names:
        return {
            "success":   False,
            "duplicate": True,
            "error":     f"'{sp_track.name}' was already played in this session.",
        }

    success = _spotify.add_to_queue(sp_track)

    if success:
        candidate = _spotify_to_candidate(sp_track, enrich=True)
        _queue.append(sp_track)
        _history.append(candidate)
        _queued_ids.add(sp_track.id)
        _queued_names.add(sp_track.name.lower())
        _state = advance_track(_state, Track(
            id=sp_track.id,
            name=sp_track.name,
            artist=sp_track.artist,
        ))
        return {
            "success":    True,
            "queued":     candidate,
            "queue_size": len(_queue),
        }
    else:
        return {
            "success": False,
            "error":   "No active Spotify device.",
        }


def get_queue_state() -> dict:
    """
    Return the current planned queue — tracks the agent has scheduled
    but that have not played yet.

    Use this to avoid scheduling the same artist or sound twice in a row,
    and to check how many tracks are already lined up.
    """
    return {
        "queue_size": len(_queue),
        "upcoming": [
            {"name": t.name, "artist": t.artist}
            for t in _queue[-5:]   # last 5 queued
        ],
    }


def get_session_history() -> dict:
    """
    Return the tracks played so far this session, most recent first.

    Use this to avoid repeating artists, to see what energy trajectory
    the session has followed, and to understand the listener's pattern
    of skips and full listens.
    """
    return {
        "tracks_played": len(_history),
        "recent": list(reversed(_history[-8:])),
    }


def reset_session(context: str = "general") -> dict:
    """
    Reset the session state for a new listening context.

    Args:
        context: one of study, workout, party, focus, chill, general

    Resets the listener state vector, arc, queue, and play history.
    The played-track guard (_queued_ids, _queued_names) is intentionally
    preserved so a track cannot be replayed even across context switches.
    """
    global _state, _queue, _history
    _state   = init_state(context)
    _queue   = []
    _history = []
    return {
        "reset":   True,
        "context": context,
        "state":   _state.to_dict(),
    }


# ════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _spotify_to_candidate(sp_track: SpotifyTrack, enrich: bool = True) -> dict:
    """
    Convert a SpotifyTrack into a rich candidate dict the agent can reason over.
    Optionally enriches with Last.fm tags and feature estimates.
    """
    candidate: dict[str, Any] = {
        "id":          sp_track.id,
        "name":        sp_track.name,
        "artist":      sp_track.artist,
        "album":       sp_track.album,
        "duration_s":  round(sp_track.duration_ms / 1000),
        "uri":         sp_track.uri,
        "tags":        [],
        "energy_est":  0.5,
        "valence_est": 0.5,
        "confidence":  0.0,
        "listeners":   0,
        "familiar":    _spotify.is_saved(sp_track.id),
    }

    if enrich:
        enr = fetch_enrichment(sp_track.artist, sp_track.name)
        if enr.found:
            estimate = estimate_features_with_fallback(
                tags=enr.tags,
                tag_weights=enr.tag_weights,
                use_semantic=True,
            )
            candidate.update({
                "tags":        enr.tags[:8],
                "energy_est":  estimate.energy,
                "valence_est": estimate.valence,
                "confidence":  estimate.confidence,
                "listeners":   enr.listeners,
            })

        info = fetch_track_info(sp_track.artist, sp_track.name)
        candidate["bpm"]              = info["bpm"]
        candidate["camelot_position"] = info["camelot_position"]

        if DISPLAY_LOGS:
            status  = "found" if info["found"] else "miss "
            bpm_str = f"BPM={candidate['bpm']}" if candidate.get("bpm") else "BPM=—"
            key_str = f"Key={candidate['camelot_position']}" if candidate.get("camelot_position") else "Key=—"
            print(f"  [getsongkey/{status}] {sp_track.artist} — {sp_track.name}  →  {bpm_str}  {key_str}")

    candidate["key"] = candidate.get("camelot_position")

    return candidate


def _find_in_history(track_name: str, artist: str) -> Track | None:
    """Look up a track in session history by name and artist."""
    for entry in reversed(_history):
        if (entry.get("name", "").lower()   == track_name.lower() and
                entry.get("artist", "").lower() == artist.lower()):
            return Track(
                id=entry.get("id", ""),
                name=entry["name"],
                artist=entry["artist"],
                tags=entry.get("tags", []),
                energy_est=entry.get("energy_est", 0.5),
                valence_est=entry.get("valence_est", 0.5),
            )
    return None