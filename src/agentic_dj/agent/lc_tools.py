"""
LangChain tool objects for the AgDJ agent (LangGraph migration).

This module is the SINGLE SOURCE OF TRUTH for the tools the LLM can call. Each
function here is a thin ``@tool``-decorated wrapper around the corresponding
plain function in ``agentic_dj.agent.tools`` — the business logic stays there
and remains directly callable (``bridge.py`` and the Python pre-step/fallback
in the loop still call ``tool_module.<fn>()`` normally).

Design rules (see plan_langgraph_migration.md §6.1):
  • Wrap, do NOT decorate in place — decorating ``tools.py`` would turn the
    functions into BaseTool objects and break every direct caller.
  • The wrapper signature exposes EXACTLY the parameters the old
    ``TOOL_DECLARATIONS`` exposed, so what the LLM sees does not change.
    (Notably ``search_tracks`` does not expose ``enrich`` — it is fixed True.)
  • The docstring IS the tool description sent to the model. The text below is
    lifted from the old TOOL_DECLARATIONS so the descriptions are identical.

The two exported lists drive both binding and execution:
  • ``ALL_TOOLS``   — every tool (used by start_session's opener loop + fallback)
  • ``CYCLE_TOOLS`` — ALL_TOOLS minus the four search tools (mid-session cycle;
    search is done in the Python pre-step, so the LLM must not search again)
"""

from langchain_core.tools import tool

import agentic_dj.agent.tools as tool_module


# ════════════════════════════════════════════════════════════════════════════
#  1. STATE TOOLS
# ════════════════════════════════════════════════════════════════════════════

@tool
def get_listener_state() -> dict:
    """Return the current listener state vector — energy, valence, focus, openness, social, arc_phase."""
    return tool_module.get_listener_state()


@tool
def update_listener_state(event: str, track_name: str, artist: str) -> dict:
    """Update the listener state based on a feedback event.

    event: one of skip, early_skip, replay, full_listen, partial_listen, thumbs_up, thumbs_down.
    """
    return tool_module.update_listener_state(event, track_name, artist)


@tool
def get_session_arc() -> dict:
    """Return the current arc phase and a description of what it means for track selection."""
    return tool_module.get_session_arc()


# ════════════════════════════════════════════════════════════════════════════
#  2. MUSIC THEORY TOOLS
# ════════════════════════════════════════════════════════════════════════════

@tool
def get_compatible_keys(camelot_position: str) -> dict:
    """Return all harmonically compatible Camelot positions for a given key.

    camelot_position: e.g. '8B', '4A'.
    """
    return tool_module.get_compatible_keys(camelot_position)


@tool
def check_transition(from_camelot: str, to_camelot: str) -> dict:
    """Check harmonic smoothness between two Camelot positions. Returns a score 0.0-1.0 and a verdict."""
    return tool_module.check_transition(from_camelot, to_camelot)


@tool
def estimate_bpm_compatibility(
    current_bpm: float,
    candidate_bpm: float,
    arc_phase: str = "build",
) -> dict:
    """Check whether a BPM jump between two tracks is smooth for the current arc phase."""
    return tool_module.estimate_bpm_compatibility(current_bpm, candidate_bpm, arc_phase)


# ════════════════════════════════════════════════════════════════════════════
#  3. SPOTIFY + LAST.FM TOOLS
# ════════════════════════════════════════════════════════════════════════════

@tool
def search_tracks_by_tag(tag: str, limit: int = 15) -> dict:
    """Search for tracks by Last.fm tag — the accurate way to find music by mood or genre.

    Use this for descriptive queries like 'energetic', 'chill', 'melancholic', 'dark',
    'happy', 'anthemic', 'ambient', 'upbeat', 'mellow'.
    Last.fm tags are crowdsourced by millions of listeners and map directly to moods/genres.
    Prefer this over search_tracks for any mood or energy query.

    tag: a Last.fm mood/genre tag e.g. 'energetic', 'chill', 'dark'. limit: max candidates (default 15).
    """
    return tool_module.search_tracks_by_tag(tag, limit)


@tool
def search_artist_tracks(artist: str, limit: int = 15) -> dict:
    """Get an artist's most popular tracks, resolved to Spotify.

    Use this when the listener wants tracks from a specific artist — it fetches their
    top tracks from Last.fm ranked by listener count, far more reliable than a Spotify
    text search which may return covers or tribute acts.

    artist: artist name e.g. 'Radiohead'. limit: max candidates (default 15).
    """
    return tool_module.search_artist_tracks(artist, limit)


@tool
def search_tracks(query: str, limit: int = 8) -> dict:
    """Search Spotify by text query for a specific track or artist name. Use for precise lookups like 'Creep Radiohead' or 'One More Time Daft Punk'. For mood/energy queries use search_tracks_by_tag instead; for artist catalogues use search_artist_tracks instead."""
    # enrich is fixed True (matches the old default; not exposed to the LLM).
    return tool_module.search_tracks(query, limit, enrich=True)


@tool
def search_tracks_by_playlist(query: str, sample_size: int = 20) -> dict:
    """Search for Spotify playlists matching a vibe, era, or mood description and return a sample of tracks from those playlists. Use for era-specific queries like '2016 pop hits' or 'chill Sunday morning'. Returns raw metadata only — no BPM or key data.

    query: short playlist-style search string, e.g. '2016 pop hits'. sample_size: max candidates (default 20).
    """
    return tool_module.search_tracks_by_playlist(query, sample_size)


@tool
def get_track_details(track_name: str, artist: str) -> dict:
    """Get full details for a specific track — metadata, tags, energy, valence, BPM, Camelot key."""
    return tool_module.get_track_details(track_name, artist)


@tool
def get_current_playback() -> dict:
    """Return what is currently playing on Spotify — track name, artist, progress."""
    return tool_module.get_current_playback()


# ════════════════════════════════════════════════════════════════════════════
#  4. QUEUE TOOLS
# ════════════════════════════════════════════════════════════════════════════

@tool
def get_queue_state() -> dict:
    """Return the current planned queue and upcoming tracks."""
    return tool_module.get_queue_state()


@tool
def get_session_history() -> dict:
    """Return tracks played so far this session, most recent first."""
    return tool_module.get_session_history()


@tool
def add_track_to_queue(track_name: str, artist: str, reason: str = "") -> dict:
    """Add a track to the Spotify playback queue. This is the terminal action — call it once with the single best candidate. Pass `reason`: one sentence explaining why this track, referencing a listener-state or arc signal AND a musical property (key/BPM/energy). This becomes the explanation shown to the listener."""
    return tool_module.add_track_to_queue(track_name, artist, reason)


@tool
def get_ranked_candidates(limit: int = 5) -> dict:
    """Find the best next-track options for the current session. Gathers candidates similar to what's playing (similar tracks, similar artists) plus genre matches, keeps only those harmonically and tempo compatible with the current track, ranks them by fit to the listener state and session arc, and returns the top `limit` (default 5). Call this once, then pick one and queue it with add_track_to_queue. Each candidate includes key, BPM, energy, valence, tags, and harmonic_score vs the current track."""
    return tool_module.get_ranked_candidates(limit)


# ════════════════════════════════════════════════════════════════════════════
#  EXPORTED TOOL LISTS
# ════════════════════════════════════════════════════════════════════════════

#: Tools removed for the mid-session cycle (search is done in the Python pre-step).
_SEARCH_TOOL_NAMES = {
    "search_tracks_by_tag",
    "search_artist_tracks",
    "search_tracks",
    "search_tracks_by_playlist",
}

#: Every LLM-callable tool. Order mirrors the old TOOL_DECLARATIONS.
ALL_TOOLS = [
    get_listener_state,
    update_listener_state,
    get_session_arc,
    get_compatible_keys,
    check_transition,
    estimate_bpm_compatibility,
    search_tracks_by_tag,
    search_artist_tracks,
    search_tracks,
    search_tracks_by_playlist,
    get_track_details,
    get_current_playback,
    get_queue_state,
    get_session_history,
    add_track_to_queue,
]

#: Focused subset for run_agent_cycle: the granular search tools are dropped
#: (the single get_ranked_candidates tool subsumes them — it gathers, filters,
#: and ranks in one call) and get_ranked_candidates is added.
CYCLE_TOOLS = [t for t in ALL_TOOLS if t.name not in _SEARCH_TOOL_NAMES] + [get_ranked_candidates]

#: name -> tool object, for execution inside the custom tools node.
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}


# ════════════════════════════════════════════════════════════════════════════
#  OPENER TOOLSET  (natural-language session start — start_session)
# ════════════════════════════════════════════════════════════════════════════
# Kept separate from ALL_TOOLS. The opener exposes only the three search tools
# plus a synthetic terminal. select_opening_track's body is never executed —
# the opener graph node validates the choice against seen_tracks in graph state.

@tool
def select_opening_track(track_name: str, artist: str) -> dict:
    """Terminal action: commit the chosen opening track. The track MUST have been returned by search_tracks_by_playlist, search_tracks, or get_track_details earlier in this session — any other name will be rejected with an error."""
    # Handled by the opener graph's tools node (validates against graph state).
    return {"success": False, "error": "select_opening_track must be handled by the opener graph node"}


#: Tools the natural-language session-start agent can call.
OPENER_TOOLS = [
    search_tracks_by_playlist,
    search_tracks,
    get_track_details,
    select_opening_track,
]
