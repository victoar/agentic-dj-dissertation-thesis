"""
Soundcharts client — two API v2.25 endpoints for song lookup.

  1. fetch_by_platform(platform, identifier) — resolve a platform-specific ID
     (e.g. Spotify track ID) to a Soundcharts song UUID.
  2. fetch_song(uuid) — fetch full song metadata by Soundcharts UUID.

Auth: set SOUNDCHARTS_APP_ID and SOUNDCHARTS_API_KEY in .env.
Results are disk-cached under .cache_soundcharts/ to avoid repeated calls.
"""

import os
import json
import hashlib
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from agentic_dj.music.camelot import from_spotify as camelot_from_spotify

load_dotenv()

_BASE = "https://customer.api.soundcharts.com"
_cache_dir = Path(".cache_soundcharts")


# ── Cache helpers ──────────────────────────────────────────────────────────────

def _cache_key(raw: str) -> str:
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


def _auth_headers() -> dict:
    return {
        "x-app-id":  os.getenv("SOUNDCHARTS_APP_ID",  ""),
        "x-api-key": os.getenv("SOUNDCHARTS_API_KEY", ""),
        "User-Agent": "AgenticDJ/1.0",
    }


def _get(url: str) -> dict:
    """HTTP GET → parsed JSON. Raises on any network or HTTP error."""
    req = urllib.request.Request(url, headers=_auth_headers())
    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code} {e.reason} — {url}\n  body: {body}") from e


# ── Public API ─────────────────────────────────────────────────────────────────

def fetch_by_platform(
    platform: str,
    identifier: str,
    use_cache: bool = True,
) -> dict:
    """
    Resolve a platform-specific track ID to a Soundcharts song UUID.

    Args:
        platform:   Platform slug, e.g. ``"spotify"``.
        identifier: Platform track ID, e.g. a Spotify track ID.
        use_cache:  Read/write the on-disk cache (default True).

    Returns:
        ``{"uuid": str | None, "found": bool, "raw": dict, "error": str | None}``
        Never raises — returns ``found=False`` on any error or miss.
        ``error`` is a human-readable description of what went wrong, or ``None``
        on success.
    """
    ck = _cache_key(f"{platform}|{identifier}")

    if use_cache:
        cached = _read_cache(ck)
        if cached is not None:
            return cached

    result: dict = {"uuid": None, "found": False, "raw": {}, "error": None}

    if not os.getenv("SOUNDCHARTS_APP_ID") or not os.getenv("SOUNDCHARTS_API_KEY"):
        result["error"] = "missing SOUNDCHARTS_APP_ID or SOUNDCHARTS_API_KEY in environment"
        print(f"  [soundcharts] by-platform/{platform}/{identifier}: {result['error']}")
        return result

    try:
        url = f"{_BASE}/api/v2.25/song/by-platform/{platform}/{identifier}"
        data = _get(url)
        song = data.get("object") or {}
        uuid = song.get("uuid")
        if uuid:
            result = {"uuid": uuid, "found": True, "raw": song, "error": None}
        else:
            result["error"] = f"response had no 'uuid' — keys: {list(song.keys())}"
            print(f"  [soundcharts/miss] by-platform/{platform}/{identifier}: {result['error']}")
    except Exception as exc:
        result["error"] = str(exc)
        print(f"  [soundcharts/error] by-platform/{platform}/{identifier}: {exc}")
        traceback.print_exc()

    if use_cache:
        _write_cache(ck, result)

    return result


def fetch_song(uuid: str, use_cache: bool = True) -> dict:
    """
    Fetch full song metadata from Soundcharts by UUID.

    Args:
        uuid:      Soundcharts song UUID (obtained via ``fetch_by_platform``).
        use_cache: Read/write the on-disk cache (default True).

    Returns:
        ``{"found": bool, "raw": dict, "error": str | None}``
        ``raw`` contains the full Soundcharts song object (BPM, key, etc.).
        Never raises — returns ``found=False`` on any error or miss.
        ``error`` is a human-readable description of what went wrong, or ``None``
        on success.
    """
    ck = _cache_key(f"song|{uuid}")

    if use_cache:
        cached = _read_cache(ck)
        if cached is not None:
            return cached

    result: dict = {"found": False, "raw": {}, "error": None}

    if not os.getenv("SOUNDCHARTS_APP_ID") or not os.getenv("SOUNDCHARTS_API_KEY"):
        result["error"] = "missing SOUNDCHARTS_APP_ID or SOUNDCHARTS_API_KEY in environment"
        print(f"  [soundcharts] song/{uuid}: {result['error']}")
        return result

    try:
        url = f"{_BASE}/api/v2.25/song/{uuid}"
        data = _get(url)
        song = data.get("object") or {}
        if song:
            result = {"found": True, "raw": song, "error": None}
        else:
            result["error"] = f"response 'object' was empty — top-level keys: {list(data.keys())}"
            print(f"  [soundcharts/miss] song/{uuid}: {result['error']}")
    except Exception as exc:
        result["error"] = str(exc)
        print(f"  [soundcharts/error] song/{uuid}: {exc}")
        traceback.print_exc()

    if use_cache:
        _write_cache(ck, result)

    return result


# ── Audio feature accessors ────────────────────────────────────────────────────

def get_tempo(raw: dict) -> float | None:
    """Return BPM from a Soundcharts song object, or None if unavailable."""
    return raw.get("audio", {}).get("tempo")


def get_key(raw: dict) -> int | None:
    """Return the Spotify-format key integer (0–11) from a Soundcharts song object."""
    return raw.get("audio", {}).get("key")


def get_mode(raw: dict) -> int | None:
    """Return the Spotify-format mode (0 = minor, 1 = major) from a Soundcharts song object."""
    return raw.get("audio", {}).get("mode")


# ── Pipeline integration ───────────────────────────────────────────────────────

def fetch_track_info(spotify_id: str, use_cache: bool = True) -> dict:
    """
    Return BPM and Camelot position for a Spotify track via Soundcharts.

    Uses the by-platform endpoint which returns audio features in a single
    request — no second API call needed.

    Returns:
        {"bpm": float | None, "camelot_position": str | None, "found": bool}
    Never raises.
    """
    r = fetch_by_platform("spotify", spotify_id, use_cache=use_cache)
    if not r["found"]:
        return {"bpm": None, "camelot_position": None, "found": False}

    raw  = r["raw"]
    bpm  = get_tempo(raw)
    key  = get_key(raw)
    mode = get_mode(raw)

    camelot = None
    if key is not None and mode is not None:
        ck = camelot_from_spotify(key, mode)
        camelot = ck.position if ck else None

    return {"bpm": bpm, "camelot_position": camelot, "found": True}
