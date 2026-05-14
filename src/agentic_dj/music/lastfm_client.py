"""
Last.fm client — enriches a (track_name, artist) pair with tags and estimated
audio features. Returns fully populated Track objects ready for the agent.

This module is the bridge between Last.fm's tag-based world and the state
model's numerical world. It combines three things we've already built:
  - pylast (Last.fm API)
  - tag → feature translation (music/tags.py)
  - the Track dataclass (agent/state.py)

Results are cached on disk because Last.fm rate-limits at 5 requests/second.
"""

import os
import json
import hashlib
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import pylast
from dotenv import load_dotenv

from agentic_dj.agent.state import Track
from agentic_dj.music.tags import estimate_features_with_fallback, TagEstimate


# ── Initialisation ───────────────────────────────────────────────────────
load_dotenv()

_network: Optional[pylast.LastFMNetwork] = None
_cache_dir = Path(".cache_lastfm")


def _get_network() -> pylast.LastFMNetwork:
    """Lazy-init the pylast network so importing this module is cheap."""
    global _network
    if _network is None:
        api_key = os.getenv("LASTFM_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LASTFM_API_KEY is not set in .env — cannot use Last.fm client."
            )
        _network = pylast.LastFMNetwork(api_key=api_key)
    return _network


def _cache_key(artist: str, track_name: str) -> str:
    """Stable hash for a (artist, track) pair used as the cache filename."""
    raw = f"{artist.lower().strip()}|{track_name.lower().strip()}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _cache_path(key: str) -> Path:
    return _cache_dir / f"{key}.json"


def _read_cache(key: str) -> Optional[dict]:
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(key: str, data: dict) -> None:
    _cache_dir.mkdir(parents=True, exist_ok=True)
    with _cache_path(key).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── Public API ───────────────────────────────────────────────────────────

@dataclass
class LastFmEnrichment:
    """Raw Last.fm data for a track, before being turned into a Track."""
    tags:         list[str]
    tag_weights:  list[float]
    listeners:    int
    playcount:    int
    found:        bool


def fetch_enrichment(
    artist:     str,
    track_name: str,
    top_n_tags: int = 10,
    use_cache:  bool = True,
) -> LastFmEnrichment:
    """
    Fetch raw Last.fm data for a track. Cached to disk.
    Returns an empty-but-valid LastFmEnrichment if the track is not found —
    never raises for "track unknown", only for auth/network failures.
    """
    key = _cache_key(artist, track_name)

    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return LastFmEnrichment(**cached)

    network = _get_network()

    try:
        track = network.get_track(artist, track_name)
        top_tags = track.get_top_tags(limit=top_n_tags) or []
        tags     = [t.item.name for t in top_tags]
        weights  = [float(t.weight) if getattr(t, "weight", None) else 1.0
                    for t in top_tags]
        # playcount and listeners can fail independently
        try:
            listeners = int(track.get_listener_count() or 0)
        except pylast.WSError:
            listeners = 0
        try:
            playcount = int(track.get_playcount() or 0)
        except pylast.WSError:
            playcount = 0
        enrichment = LastFmEnrichment(
            tags=tags, tag_weights=weights,
            listeners=listeners, playcount=playcount,
            found=True,
        )
    except pylast.WSError:
        # Track not found on Last.fm — return empty enrichment
        enrichment = LastFmEnrichment(
            tags=[], tag_weights=[],
            listeners=0, playcount=0,
            found=False,
        )

    if use_cache:
        _write_cache(key, asdict(enrichment))

    return enrichment


def enrich_track(
    track_id:    str,
    artist:      str,
    track_name:  str,
    bpm:         float = 120.0,
    familiar:    bool  = False,
    use_cache:   bool  = True,
) -> tuple[Track, TagEstimate]:
    """
    Build a fully-populated Track from a (artist, name) pair.
    Returns the Track plus the TagEstimate diagnostics so callers can see
    how confident we are in the energy/valence estimates.
    """
    enr = fetch_enrichment(artist, track_name, use_cache=use_cache)

    # Use Last.fm listener count as mainstream proxy (0–10M+ range)
    # Normalise to 0–100 scale loosely comparable to Spotify popularity
    lastfm_popularity = min(100, int(enr.listeners / 100_000)) if enr.listeners else None

    estimate = estimate_features_with_fallback(
        tags=enr.tags,
        tag_weights=enr.tag_weights,
        use_semantic=True,
        popularity=lastfm_popularity,
    )

    track = Track(
        id=track_id,
        name=track_name,
        artist=artist,
        tags=enr.tags,
        bpm=bpm,
        energy_est=estimate.energy,
        valence_est=estimate.valence,
        familiar=familiar,
    )

    return track, estimate


def get_similar_artists(artist: str, limit: int = 5, use_cache: bool = True) -> list[str]:
    """
    Return up to `limit` artist names similar to `artist` according to Last.fm.
    Returns [] if the artist is not found or the API fails.
    Results are disk-cached using the same cache directory as track enrichments.
    """
    key = hashlib.sha1(f"similar|{artist.lower().strip()}".encode("utf-8")).hexdigest()[:16]

    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached.get("names", [])

    network = _get_network()
    try:
        similar = network.get_artist(artist).get_similar(limit=limit)
        names = [s.item.name for s in similar]
    except pylast.WSError:
        names = []

    if use_cache:
        _write_cache(key, {"names": names})

    return names


def clear_cache() -> int:
    """Delete all cached Last.fm responses. Returns the number of files removed."""
    if not _cache_dir.exists():
        return 0
    removed = 0
    for f in _cache_dir.glob("*.json"):
        f.unlink()
        removed += 1
    return removed