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
from agentic_dj.music.lastfm_client import (
    enrich_track,
    fetch_enrichment,
    get_top_tracks_for_tag,
    get_artist_top_tracks_lastfm,
    get_similar_tracks,
    get_similar_artists,
)
from agentic_dj.music.soundcharts_client import fetch_track_info
from agentic_dj.spotify.client import SpotifyClient, SpotifyTrack


# ── Shared instances (one per session) ──────────────────────────────────────
_spotify = SpotifyClient()
_queue:     list[SpotifyTrack] = []
_state:     ListenerState      = init_state("general")
_history:   list[dict]         = []   # every track played this session
_lookahead: list[str]          = []   # track IDs queued ahead but not yet playing

# Session-scoped memoisation so the same track is not re-resolved / re-enriched
# at each stage (pre-fetch → get_track_details → add_track_to_queue). Without
# this a single committed track triggers 3× Spotify searches + enrichment.
_resolve_cache:   dict[str, list] = {}   # search query  -> [SpotifyTrack]
_candidate_cache: dict[str, dict] = {}   # spotify id    -> enriched candidate dict

DISPLAY_LOGS: bool = False   # synced from loop.py at the start of each cycle


def _resolve(query: str, limit: int = 1) -> list:
    """Cached single-track Spotify resolve — avoids repeated identical searches."""
    key = f"{query}|{limit}"
    if key not in _resolve_cache:
        _resolve_cache[key] = _spotify.search(query, limit=limit)
    return _resolve_cache[key]


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

    Steady-state phases (warmup, peak) use a tight ±8 BPM window.
    Transition phases (build, cooldown) allow ±25 BPM because energy is
    intentionally shifting and larger tempo steps are acceptable.
    """
    thresholds = {
        "warmup":   8.0,   # steady — establishing the session
        "build":    25.0,  # transition — energy intentionally rising
        "peak":     8.0,   # steady — holding at maximum energy
        "cooldown": 25.0,  # transition — energy intentionally winding down
    }

    threshold  = thresholds.get(arc_phase, 8.0)
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

    # Filter out candidates that have no BPM AND no Camelot key.
    # These tracks cannot be evaluated for harmonic or tempo compatibility
    # and should never be committed ahead of tracks that do have data.
    eligible   = [c for c in candidates if c.get("bpm") or c.get("camelot_position")]
    filtered_n = len(candidates) - len(eligible)

    if DISPLAY_LOGS and filtered_n:
        print(f"  [search] dropped {filtered_n} candidate(s) missing both BPM and key")

    # Fall back to the full set only if every candidate is missing data —
    # this keeps the agent from receiving an empty result and searching again.
    output = eligible if eligible else candidates

    return {
        "query":                query,
        "count":                len(output),
        "candidates":           output,
        "filtered_no_data":     filtered_n,
        "all_missing_data":     not bool(eligible),
    }


def search_tracks_by_playlist(
    query:       str,
    sample_size: int = 20,
) -> dict:
    """
    Search for Spotify playlists matching a vibe, era, or mood description,
    then return a random sample of tracks from the top results.

    Use this instead of search_tracks when the input is a descriptive phrase
    such as '2016 pop hits', 'chill Sunday morning', or 'late night R&B'.
    Playlist context is a far more reliable genre/era filter than free-text
    track search for these kinds of queries.

    Returns raw Spotify metadata only (name, artist, album, duration).
    No energy, valence, tags, BPM, or key data — those are not needed when
    choosing an opening track. Call get_track_details on your chosen candidate
    if you need the full profile before committing.

    Falls back to a raw track search if no playlists are found.
    """
    import random

    playlists = _spotify.search_playlists(query, limit=5)

    raw_tracks: list = []
    playlists_used: list[str] = []

    # Try all returned playlists until we have a usable pool.
    # Spotify editorial playlists return 403 in Development Mode —
    # get_playlist_tracks silently returns [] for those, so we keep
    # trying until we find a user-created playlist that responds.
    for pl in playlists:
        if len(raw_tracks) >= sample_size * 2:
            break
        tracks = _spotify.get_playlist_tracks(pl["id"], limit=50)
        if tracks:
            raw_tracks.extend(tracks)
            playlists_used.append(pl["name"])

    # Fall back to regular track search if every playlist was empty/forbidden.
    # Cap at 10 — Spotify's Development Mode rejects search limits above that.
    if not raw_tracks:
        fallback = _spotify.search(query, limit=min(sample_size, 10))
        raw_tracks = fallback
        playlists_used = []

    random.shuffle(raw_tracks)

    seen_ids: set[str] = set()
    candidates: list[dict] = []
    for sp_track in raw_tracks:
        if len(candidates) >= sample_size:
            break
        if sp_track.id in _queued_ids:
            continue
        if sp_track.name.lower() in _queued_names:
            continue
        if sp_track.id in seen_ids:
            continue
        seen_ids.add(sp_track.id)
        candidates.append({
            "id":         sp_track.id,
            "name":       sp_track.name,
            "artist":     sp_track.artist,
            "album":      sp_track.album,
            "duration_s": round(sp_track.duration_ms / 1000),
            "uri":        sp_track.uri,
        })

    return {
        "query":          query,
        "count":          len(candidates),
        "candidates":     candidates,
        "source":         "playlist" if playlists_used else "track_search",
        "playlists_used": playlists_used,
    }


def search_tracks_by_tag(tag: str, limit: int = 15) -> dict:
    """
    Search for tracks by Last.fm tag — the accurate way to find music by mood,
    energy, or genre.

    Use this instead of search_tracks when the query is descriptive:
    e.g. 'energetic', 'chill', 'melancholic', 'dark', 'happy', 'anthemic',
    'ambient', 'upbeat', 'mellow'. Last.fm's tag index is crowdsourced by
    millions of listeners and maps directly to moods and genres. Spotify text
    search does not understand these terms — it just matches them against
    track titles.

    Args:
        tag:   a Last.fm mood/genre tag e.g. 'energetic', 'chill', 'dark'
        limit: max candidates to return (default 15, max 20)

    Returns enriched candidates with energy/valence estimates, same format
    as search_tracks.
    """
    limit       = min(limit, 20)
    raw_items   = get_top_tracks_for_tag(tag, limit=limit * 2)  # fetch extra — some won't resolve on Spotify

    candidates = []
    for item in raw_items:
        if len(candidates) >= limit:
            break
        name   = item.get("name", "")
        artist = item.get("artist", "")
        if not name or not artist:
            continue

        results = _resolve(f"{name} {artist}", limit=1)
        if not results:
            continue
        sp = results[0]

        if sp.id in _queued_ids or sp.name.lower() in _queued_names:
            continue

        candidates.append(_spotify_to_candidate(sp, enrich=True))

    return {
        "tag":        tag,
        "count":      len(candidates),
        "candidates": candidates,
    }


def search_artist_tracks(artist: str, limit: int = 15) -> dict:
    """
    Get an artist's most popular tracks, resolved to Spotify.

    Use this when the listener has expressed interest in a specific artist —
    it fetches that artist's top tracks from Last.fm (ranked by listener count)
    and resolves each to a Spotify-playable candidate. This is far more
    reliable than search_tracks('artist name') which may return tribute acts,
    covers, or unrelated results.

    Args:
        artist: artist name e.g. 'Radiohead', 'Daft Punk'
        limit:  max candidates to return (default 15, max 20)

    Returns enriched candidates with energy/valence estimates, same format
    as search_tracks.
    """
    limit     = min(limit, 20)
    raw_items = get_artist_top_tracks_lastfm(artist, limit=limit * 2)

    candidates = []
    for item in raw_items:
        if len(candidates) >= limit:
            break
        name        = item.get("name", "")
        item_artist = item.get("artist", artist)
        if not name:
            continue

        results = _resolve(f"{name} {item_artist}", limit=1)
        if not results:
            continue
        sp = results[0]

        if sp.id in _queued_ids or sp.name.lower() in _queued_names:
            continue

        candidates.append(_spotify_to_candidate(sp, enrich=True))

    return {
        "artist":     artist,
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
    results = _resolve(f"{track_name} {artist}", limit=1)
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

def add_track_to_queue(track_name: str, artist: str, reason: str = "") -> dict:
    """
    Search for a track and add it to the Spotify playback queue.

    This is the primary action the agent takes to schedule the next song.
    The track is also recorded in session history so future state updates
    can reference its features.

    Args:
        track_name: exact or approximate track title
        artist:     artist name
        reason:     one sentence explaining the choice, referencing a listener-state
                    or arc signal and a musical property (key/BPM/energy). Surfaced
                    to the listener as the agent's explanation.

    Returns success status and the track that was queued.
    """
    global _queue, _history, _state

    results = _resolve(f"{track_name} {artist}", limit=1)
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
        _lookahead.append(sp_track.id)
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
            "reason":     reason,   # the agent's one-line justification → explanation
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


def get_lookahead_depth() -> int:
    """Return the number of tracks currently buffered ahead of the playing track."""
    return len(_lookahead)


def consume_lookahead_up_to(track_id: str) -> int:
    """
    Mark `track_id` (and every track before it) as consumed — it is now playing.
    Handles out-of-band skips: if the user jumped two tracks ahead, both are
    removed in one call.
    Returns the number of slots freed (0 if track_id not found in lookahead).
    """
    global _lookahead
    if track_id not in _lookahead:
        return 0
    idx = _lookahead.index(track_id)
    freed = idx + 1
    _lookahead = _lookahead[idx + 1:]
    return freed


def reset_session(context: str = "general") -> dict:
    """
    Reset the session state for a new listening context.

    Args:
        context: one of study, workout, party, focus, chill, general

    Resets the listener state vector, arc, queue, and play history.
    The played-track guard (_queued_ids, _queued_names) is intentionally
    preserved so a track cannot be replayed even across context switches.
    """
    global _state, _queue, _history, _lookahead
    _state     = init_state(context)
    _queue     = []
    _history   = []
    _lookahead = []
    _resolve_cache.clear()
    _candidate_cache.clear()
    return {
        "reset":   True,
        "context": context,
        "state":   _state.to_dict(),
    }


# ════════════════════════════════════════════════════════════════════════════
#  CANDIDATE SELECTION  —  discovery bucket + composite ranking
# ════════════════════════════════════════════════════════════════════════════
#
# A single LLM-callable tool (get_ranked_candidates) gathers candidates from
# several sources into a bucket, hard-filters for harmonic/tempo compatibility
# with the current track, ranks by a weighted composite, and returns the top N.
# The same _rank_candidates function backs the loop's safety-net fallback.

# Tunable ranking weights — calibrate these against logged sessions later.
RANK_WEIGHTS: dict[str, float] = {
    "ev":          0.35,   # energy/valence fit vs the listener state
    "harmonic":    0.25,   # Camelot compatibility vs the current track
    "bpm":         0.15,   # tempo closeness vs the current track
    "arc":         0.15,   # alignment with the arc-phase energy band
    "familiarity": 0.10,   # openness-aware familiar/novel preference
}

# BPM jump tolerance per arc phase (mirrors estimate_bpm_compatibility):
_BPM_THRESHOLDS = {"warmup": 8.0, "build": 25.0, "peak": 8.0, "cooldown": 25.0}
_HARMONIC_MIN   = 0.4   # "acceptable" transition (matches check_transition)
# Arc-phase target energy bands (from the dissertation's session arc).
_ARC_TARGET = {"warmup": (0.3, 0.5), "build": (0.5, 0.7),
               "peak": (0.8, 1.0), "cooldown": (0.2, 0.4)}

_BUCKET_SIZE       = 12    # candidates enriched + ranked each cycle
_SIMILAR_TRACKS_N  = 12    # from Last.fm similar-tracks (primary)
_SIMILAR_ARTISTS_N = 4     # similar artists, one top track each (secondary)


def _arc_energy_target(arc_phase: str) -> tuple[float, float, float]:
    lo, hi = _ARC_TARGET.get(arc_phase, (0.5, 0.7))
    return (lo + hi) / 2.0, lo, hi


def _state_to_tags(state: dict) -> list[str]:
    """Map the listener state vector to up to two Last.fm tags (energy/valence/arc)."""
    energy  = state.get("energy",  0.5)
    valence = state.get("valence", 0.5)
    arc     = state.get("arc_phase", "build")
    tags: list[str] = []
    if arc == "peak":
        tags.append("anthemic")
    elif arc == "cooldown":
        tags.append("mellow")
    if energy > 0.65:
        tags.append("energetic")
    elif energy < 0.35:
        tags.append("chill")
    else:
        tags.append("indie")
    if valence > 0.65 and "anthemic" not in tags:
        tags.append("happy")
    elif valence < 0.35 and "mellow" not in tags:
        tags.append("melancholic")
    return tags[:2] if tags else ["indie"]


def _current_track_profile() -> dict | None:
    """
    Full enriched profile (tags, camelot_position, bpm, energy/valence) of the
    track currently playing. Looks it up in session history first; if absent
    (e.g. the listener changed track directly in Spotify) it is enriched on the
    fly via get_track_details (cached). None if nothing is playing / unresolved.
    """
    playback = get_current_playback()
    name     = (playback.get("track_name") or "").strip()
    artist   = (playback.get("artist") or "").strip()
    if not name:
        return None
    for entry in get_session_history().get("recent", []):
        if entry.get("name", "").lower() == name.lower():
            return entry
    details = get_track_details(name, artist)
    if isinstance(details, dict) and details.get("name") and not details.get("error"):
        return details
    return None


def _seed_tags(state: dict, current: dict | None) -> list[str]:
    """Genre tags for the tag-based bucket fill: the current track's top tags
    (artist-name tag skipped), falling back to state-derived tags."""
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
    candidates:  list[dict],
    state:       dict,
    current_key: str | None,
    current_bpm: float | None,
) -> list[dict]:
    """
    Hard harmonic + tempo filter vs the current track: keep a candidate if its
    Camelot key is compatible (strength ≥ 0.4) AND its BPM is within the arc
    threshold. Unknown values pass (so missing data never empties the pool).
    Returns the full list unchanged if nothing passes (soft degrade).
    """
    threshold = _BPM_THRESHOLDS.get(state.get("arc_phase", "build"), 8.0)
    from_key  = camelot_parse(current_key) if current_key else None

    def harmonic_ok(c: dict) -> bool:
        if from_key is None:
            return True
        to_key = camelot_parse(c.get("camelot_position") or "")
        return to_key is None or compatibility_strength(from_key, to_key) >= _HARMONIC_MIN

    def bpm_ok(c: dict) -> bool:
        cb = c.get("bpm")
        return not (current_bpm and cb) or abs(current_bpm - cb) <= threshold

    compatible = [c for c in candidates if harmonic_ok(c) and bpm_ok(c)]
    return compatible if compatible else candidates


def _rank_candidates(
    candidates:  list[dict],
    state:       dict,
    current_key: str | None,
    current_bpm: float | None,
) -> list[dict]:
    """
    Sort candidates best-first by the RANK_WEIGHTS composite: energy/valence fit,
    harmonic compatibility, BPM closeness, arc-band alignment, and openness-aware
    familiarity. Unknown key/BPM contribute a neutral 0.5 so missing data does
    not distort the order.
    """
    tgt_e    = state.get("energy",  0.5)
    tgt_v    = state.get("valence", 0.5)
    openness = state.get("openness", 0.5)
    arc      = state.get("arc_phase", "build")
    _, arc_lo, arc_hi = _arc_energy_target(arc)
    threshold = _BPM_THRESHOLDS.get(arc, 8.0)
    from_key  = camelot_parse(current_key) if current_key else None

    def score(c: dict) -> float:
        e = c.get("energy_est", 0.5)
        v = c.get("valence_est", 0.5)
        ev_fit = 1.0 - (abs(e - tgt_e) + abs(v - tgt_v)) / 2.0

        to_key = camelot_parse(c.get("camelot_position") or "")
        harmonic = compatibility_strength(from_key, to_key) if (from_key and to_key) else 0.5

        cb = c.get("bpm")
        bpm_close = (1.0 - min(abs(current_bpm - cb) / threshold, 1.0)) if (current_bpm and cb) else 0.5

        if arc_lo <= e <= arc_hi:
            arc_fit = 1.0
        else:
            dist = (arc_lo - e) if e < arc_lo else (e - arc_hi)
            arc_fit = max(0.0, 1.0 - dist / 0.3)

        familiar = bool(c.get("familiar"))
        fam_fit = (0.0 if familiar else 1.0) if openness >= 0.5 else (1.0 if familiar else 0.0)

        return (RANK_WEIGHTS["ev"]          * ev_fit
                + RANK_WEIGHTS["harmonic"]    * harmonic
                + RANK_WEIGHTS["bpm"]         * bpm_close
                + RANK_WEIGHTS["arc"]         * arc_fit
                + RANK_WEIGHTS["familiarity"] * fam_fit)

    return sorted(candidates, key=score, reverse=True)


def search_similar_tracks(artist: str, track: str, limit: int = 15) -> dict:
    """
    Resolve Last.fm 'similar tracks' for (artist, track) to enriched Spotify
    candidates — the strongest continuity signal for the next track.
    Same candidate shape as search_artist_tracks.
    """
    raw = get_similar_tracks(artist, track, limit=limit * 2)
    candidates: list[dict] = []
    for item in raw:
        if len(candidates) >= limit:
            break
        name        = item.get("name", "")
        item_artist = item.get("artist", "")
        if not name:
            continue
        results = _resolve(f"{name} {item_artist}", limit=1)
        if not results:
            continue
        sp = results[0]
        if sp.id in _queued_ids or sp.name.lower() in _queued_names:
            continue
        candidates.append(_spotify_to_candidate(sp, enrich=True))
    return {"track": track, "artist": artist, "count": len(candidates), "candidates": candidates}


def _gather_bucket(current: dict | None, state: dict) -> list[dict]:
    """
    Fill a de-duped bucket (≤ _BUCKET_SIZE) of enriched candidates from, in order:
    similar tracks to the current track, similar artists' top tracks, then genre
    tags. Candidates are resolved + enriched one by one (serial). Skips
    already-played tracks and the current track.
    """
    bucket: list[dict] = []
    seen: set[str] = set()
    current_id = current.get("id") if current else None

    def add(cands: list[dict]) -> bool:
        for c in cands:
            cid = c.get("id")
            if not cid or cid in seen or cid in _queued_ids or cid == current_id:
                continue
            seen.add(cid)
            bucket.append(c)
            if len(bucket) >= _BUCKET_SIZE:
                return True
        return False

    if current:
        if add(search_similar_tracks(current.get("artist", ""), current.get("name", ""),
                                     limit=_SIMILAR_TRACKS_N).get("candidates", [])):
            return bucket
        for art in get_similar_artists(current.get("artist", ""), limit=_SIMILAR_ARTISTS_N):
            if add(search_artist_tracks(art, limit=1).get("candidates", [])):
                return bucket

    for tag in _seed_tags(state, current):
        if add(search_tracks_by_tag(tag, limit=10).get("candidates", [])):
            return bucket
    return bucket


def get_ranked_candidates(limit: int = 5) -> dict:
    """
    Find the best next-track options for the current session.

    Gathers candidates from several sources (tracks similar to what's playing,
    similar artists, and genre tags), keeps only those harmonically and tempo
    compatible with the current track, ranks them by fit to the listener state
    and session arc, and returns the top `limit` (default 5).

    Call this once at the start of a cycle, then pick the best candidate and
    queue it with add_track_to_queue. Each returned candidate includes its key,
    BPM, energy, valence, tags, and harmonic_score relative to the current track.
    """
    state       = _state.to_dict()
    current     = _current_track_profile()
    current_key = current.get("camelot_position") if current else None
    current_bpm = current.get("bpm") if current else None

    bucket = _gather_bucket(current, state)
    pool   = _compatible_pool(bucket, state, current_key, current_bpm)
    ranked = _rank_candidates(pool, state, current_key, current_bpm)[:max(1, limit)]

    from_key = camelot_parse(current_key) if current_key else None
    out: list[dict] = []
    for c in ranked:
        to_key = camelot_parse(c.get("camelot_position") or "")
        harm   = round(compatibility_strength(from_key, to_key), 2) if (from_key and to_key) else None
        out.append({
            "name":           c.get("name"),
            "artist":         c.get("artist"),
            "key":            c.get("camelot_position"),
            "bpm":            c.get("bpm"),
            "energy":         round(c.get("energy_est", 0.5), 2),
            "valence":        round(c.get("valence_est", 0.5), 2),
            "harmonic_score": harm,
            "tags":           c.get("tags", [])[:3],
        })

    if DISPLAY_LOGS:
        print(f"  [ranked] bucket={len(bucket)} compatible={len(pool)} → top {len(out)}"
              f" (from key={current_key or '?'} bpm={current_bpm or '?'})")

    return {
        "from":       {"name": current.get("name") if current else None,
                       "key": current_key, "bpm": current_bpm},
        "count":      len(out),
        "candidates": out,
    }


# ════════════════════════════════════════════════════════════════════════════
#  INTERNAL HELPERS
# ════════════════════════════════════════════════════════════════════════════

def _record_played_track(sp_track: SpotifyTrack) -> dict:
    """
    Record a track that was started via play() rather than queued.
    Updates all session-tracking globals (_history, _queued_ids, _queued_names,
    _state) without touching _queue or _lookahead — those are for buffered-ahead
    tracks only.
    """
    global _state
    candidate = _spotify_to_candidate(sp_track, enrich=True)
    _history.append(candidate)
    _queued_ids.add(sp_track.id)
    _queued_names.add(sp_track.name.lower())
    _state = advance_track(_state, Track(
        id=sp_track.id,
        name=sp_track.name,
        artist=sp_track.artist,
    ))
    return candidate


def _spotify_to_candidate(sp_track: SpotifyTrack, enrich: bool = True) -> dict:
    """
    Convert a SpotifyTrack into a rich candidate dict the agent can reason over.
    Optionally enriches with Last.fm tags and feature estimates.

    Enriched results are memoised by Spotify id for the session so the same
    track is not re-enriched (Last.fm + Soundcharts + is_saved) at each stage.
    """
    if enrich and sp_track.id and sp_track.id in _candidate_cache:
        return dict(_candidate_cache[sp_track.id])   # copy — callers may mutate

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

        info = fetch_track_info(sp_track.id)
        candidate["bpm"]              = info["bpm"]
        candidate["camelot_position"] = info["camelot_position"]

        if DISPLAY_LOGS:
            status  = "found" if info["found"] else "miss "
            bpm_str = f"BPM={candidate['bpm']}" if candidate.get("bpm") else "BPM=—"
            key_str = f"Key={candidate['camelot_position']}" if candidate.get("camelot_position") else "Key=—"
            print(f"  [soundcharts/{status}] {sp_track.artist} — {sp_track.name}  →  {bpm_str}  {key_str}")

    candidate["key"] = candidate.get("camelot_position")

    if enrich and sp_track.id:
        _candidate_cache[sp_track.id] = dict(candidate)

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