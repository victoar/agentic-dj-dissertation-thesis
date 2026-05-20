"""
Deezer client — fetches BPM for a (artist, title) pair via Deezer's public API
(no auth required).

Two-step fetch: search for a matching track ID, then fetch the track detail
which carries the BPM field. Results are cached to disk so repeated lookups
are instant. Key is not available from Deezer; callers should leave it as None.
"""

import re
import json
import hashlib
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional

_cache_dir = Path(".cache_deezer")


def _cache_key(artist: str, title: str) -> str:
    raw = f"{artist.lower().strip()}|{title.lower().strip()}"
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


def _normalise(s: str) -> set[str]:
    return set(re.sub(r"[^\w\s]", "", s.lower()).split())


def _titles_match(a: str, b: str) -> bool:
    na, nb = _normalise(a), _normalise(b)
    if not na or not nb:
        return False
    return len(na & nb) / max(len(na), len(nb)) >= 0.5


def _get(url: str) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "AgenticDJ/1.0"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def fetch_bpm(artist: str, title: str, use_cache: bool = True) -> dict:
    """
    Return BPM for a track via Deezer's public API.

    Returns {"bpm": float | None, "found": bool}.
    Never raises — returns found=False on any error or miss.
    """
    key = _cache_key(artist, title)

    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached

    result = {"bpm": None, "found": False}

    try:
        query = urllib.parse.quote(f'artist:"{artist}" track:"{title}"')
        search_data = _get(f"https://api.deezer.com/search?q={query}&limit=5")

        track_id = None
        for item in search_data.get("data", []):
            item_title  = item.get("title", "")
            item_artist = item.get("artist", {}).get("name", "")
            if (
                _titles_match(item_title, title)
                and item_artist.lower().strip() == artist.lower().strip()
            ):
                track_id = item.get("id")
                break

        if track_id:
            detail = _get(f"https://api.deezer.com/track/{track_id}")
            bpm_val = detail.get("bpm") or 0
            if bpm_val > 0:
                result = {"bpm": float(bpm_val), "found": True}

    except Exception:
        pass

    if use_cache:
        _write_cache(key, result)

    return result
