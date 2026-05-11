"""
GetSongKey client — fetches BPM and musical key for a (artist, title) pair.

Wraps the GetSongKey API (getsongkey.com), which returns both tempo and
harmonic key in a single request. Results are cached on disk so repeated
lookups are instant.

API key: set GETSONGKEY_API_KEY in .env.
"""

import re
import os
import json
import hashlib
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from agentic_dj.music.camelot import from_spotify as camelot_from_spotify

load_dotenv()

_cache_dir = Path(".cache_getsongkey")

# Note name → Spotify integer (0–11)
_NOTE_TO_INT = {
    "C": 0,  "C#": 1, "Db": 1,
    "D": 2,  "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5,  "F#": 6, "Gb": 6,
    "G": 7,  "G#": 8, "Ab": 8,
    "A": 9,  "A#": 10, "Bb": 10,
    "B": 11,
}


# ── Cache helpers ─────────────────────────────────────────────────────────────

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


# ── Key parsing ───────────────────────────────────────────────────────────────

def _key_string_to_camelot(key_str: str) -> str | None:
    """
    Parse a key string like "Ab Major", "Am", "F# Minor", "A#/Bb Major", "C♯"
    into a Camelot position string like "4B". Returns None if unparseable.
    """
    s = key_str.strip()
    # Normalise Unicode accidentals to ASCII
    s = s.replace("♯", "#").replace("♭", "b")
    # Handle "X/Y" enharmonic notation — take the first spelling
    s = s.split("/")[0].strip()

    s_lower = s.lower()

    if "minor" in s_lower:
        mode = 0
        note_part = re.sub(r"\bminor\b", "", s, flags=re.IGNORECASE).strip()
    elif "major" in s_lower:
        mode = 1
        note_part = re.sub(r"\bmajor\b", "", s, flags=re.IGNORECASE).strip()
    elif s_lower.endswith("m"):
        mode = 0
        note_part = s[:-1]
    else:
        mode = 1
        note_part = s

    note_part = note_part.strip()
    if not note_part:
        return None

    # Normalise: "f#" → "F#", "ab" → "Ab"
    note_part = note_part[0].upper() + note_part[1:].lower()

    key_int = _NOTE_TO_INT.get(note_part)
    if key_int is None:
        return None

    camelot = camelot_from_spotify(key_int, mode)
    return camelot.position if camelot else None


# ── Public API ────────────────────────────────────────────────────────────────

def fetch_track_info(artist: str, title: str, use_cache: bool = True) -> dict:
    """
    Return BPM and Camelot position for a track via the GetSongKey API.

    Returns:
        {"bpm": float | None, "camelot_position": str | None, "found": bool}
    Never raises — returns found=False on any error or miss.
    """
    key = _cache_key(artist, title)

    if use_cache:
        cached = _read_cache(key)
        if cached is not None:
            return cached

    result: dict = {"bpm": None, "camelot_position": None, "found": False}

    api_key = os.getenv("GETSONGKEY_API_KEY")
    if not api_key:
        return result

    try:
        lookup = urllib.parse.quote(f"song:{title} artist:{artist}")
        url = (
            f"https://api.getsong.co/search/"
            f"?api_key={api_key}&type=both&lookup={lookup}"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "AgenticDJ/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        # Response may be {"search": [...]} or a bare list
        items = data.get("search", data) if isinstance(data, dict) else data

        for item in items:
            item_title  = item.get("title", "")
            item_artist = item.get("artist", {})
            if isinstance(item_artist, dict):
                item_artist = item_artist.get("name", "")

            title_match  = item_title.lower().strip()  == title.lower().strip()
            artist_match = item_artist.lower().strip() == artist.lower().strip()

            if title_match and artist_match:
                bpm_raw = item.get("tempo") or item.get("bpm")
                key_raw = item.get("key_of") or item.get("key")

                bpm     = float(bpm_raw) if bpm_raw else None
                camelot = _key_string_to_camelot(str(key_raw)) if key_raw else None

                result = {"bpm": bpm, "camelot_position": camelot, "found": True}
                break

    except Exception:
        pass

    if use_cache:
        _write_cache(key, result)

    return result
