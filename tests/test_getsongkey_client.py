"""
Tests for the GetSongKey BPM + key client.

Section A: unit tests — offline, use monkeypatching to avoid network calls.
Section B: integration tests — hit the live api.getsong.co on first run, then
           use the on-disk cache on subsequent runs. Delete .cache_getsongkey/
           to force fresh fetches.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import agentic_dj.music.getsongkey_client as gsk_module
from agentic_dj.music.getsongkey_client import (
    fetch_track_info,
    _cache_key,
    _key_string_to_camelot,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_response(items: list[dict]) -> MagicMock:
    """Return a mock that mimics urllib.request.urlopen for a search response."""
    body = json.dumps({"search": items}).encode("utf-8")
    cm = MagicMock()
    cm.__enter__ = lambda s: s
    cm.__exit__ = MagicMock(return_value=False)
    cm.read.return_value = body
    return cm


def _track_item(title: str, artist: str, tempo: str, key_of: str) -> dict:
    return {"id": "1", "title": title, "artist": {"name": artist},
            "tempo": tempo, "key_of": key_of}


def run_tests():
    passed = 0
    failed = 0

    def check(label, condition, got=None, expected=None):
        nonlocal passed, failed
        if condition:
            print(f"  ✓  {label}")
            passed += 1
        else:
            print(f"  ✗  {label}  (got {got!r}, expected {expected!r})")
            failed += 1

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION A — Unit tests (no network)
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("GetSongKey Client — Unit Tests (offline)")
    print("=" * 60)

    # ── A1: _key_string_to_camelot — minor shorthand ─────────────────────────
    print("\n[A1] _key_string_to_camelot — minor shorthand (e.g. 'Am')")
    check("Am  → 8A",  _key_string_to_camelot("Am")  == "8A",
          got=_key_string_to_camelot("Am"),  expected="8A")
    check("Bm  → 10A", _key_string_to_camelot("Bm")  == "10A",
          got=_key_string_to_camelot("Bm"),  expected="10A")
    check("Dm  → 7A",  _key_string_to_camelot("Dm")  == "7A",
          got=_key_string_to_camelot("Dm"),  expected="7A")

    # ── A2: _key_string_to_camelot — written-out minor ───────────────────────
    print("\n[A2] _key_string_to_camelot — 'F# Minor' style")
    check("F# Minor → 11A", _key_string_to_camelot("F# Minor") == "11A",
          got=_key_string_to_camelot("F# Minor"), expected="11A")
    check("A minor  → 8A",  _key_string_to_camelot("A minor")  == "8A",
          got=_key_string_to_camelot("A minor"),  expected="8A")

    # ── A3: _key_string_to_camelot — written-out major ───────────────────────
    print("\n[A3] _key_string_to_camelot — 'Ab Major' style")
    check("Ab Major → 4B", _key_string_to_camelot("Ab Major") == "4B",
          got=_key_string_to_camelot("Ab Major"), expected="4B")
    check("C Major  → 8B", _key_string_to_camelot("C Major")  == "8B",
          got=_key_string_to_camelot("C Major"),  expected="8B")
    check("G        → 9B", _key_string_to_camelot("G")        == "9B",
          got=_key_string_to_camelot("G"),        expected="9B")

    # ── A4: _key_string_to_camelot — Unicode accidentals ─────────────────────
    print("\n[A4] _key_string_to_camelot — Unicode ♯ / ♭ accidentals")
    check("C♯  → 3B",  _key_string_to_camelot("C♯")  == "3B",
          got=_key_string_to_camelot("C♯"),  expected="3B")
    check("D♯  → 5B",  _key_string_to_camelot("D♯")  == "5B",
          got=_key_string_to_camelot("D♯"),  expected="5B")
    check("A♭m → 1A",  _key_string_to_camelot("A♭m") == "1A",
          got=_key_string_to_camelot("A♭m"),  expected="1A")

    # ── A5: _key_string_to_camelot — enharmonic slash notation ───────────────
    print("\n[A5] _key_string_to_camelot — enharmonic 'A#/Bb' notation")
    check("A#/Bb Major → 6B", _key_string_to_camelot("A#/Bb Major") == "6B",
          got=_key_string_to_camelot("A#/Bb Major"), expected="6B")
    check("D#/Eb       → 5B", _key_string_to_camelot("D#/Eb")       == "5B",
          got=_key_string_to_camelot("D#/Eb"),       expected="5B")

    # ── A6: _key_string_to_camelot — invalid / empty → None ─────────────────
    print("\n[A6] _key_string_to_camelot — bad input returns None")
    check("empty string  → None", _key_string_to_camelot("") is None)
    check("'Major' only  → None", _key_string_to_camelot("Major") is None)
    check("garbage       → None", _key_string_to_camelot("XQ99") is None)

    # ── A7: _cache_key is stable and case-insensitive ────────────────────────
    print("\n[A7] _cache_key — stable, case-insensitive")
    k1 = _cache_key("M83", "Midnight City")
    k2 = _cache_key("m83", "midnight city")
    k3 = _cache_key("M83", "Different Track")
    check("same artist/title → same key",    k1 == k2, got=k1, expected=k2)
    check("different title → different key", k1 != k3, got=k1)
    check("key is 16 hex chars",             len(k1) == 16 and k1.isalnum())

    # ── A8: successful mocked fetch — BPM and key both returned ─────────────
    print("\n[A8] fetch_track_info — successful fetch (mocked network)")
    item = _track_item("Midnight City", "M83", "103", "Bm")
    resp = _make_response([item])

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cache = gsk_module._cache_dir
        gsk_module._cache_dir = Path(tmpdir)
        try:
            with patch("urllib.request.urlopen", return_value=resp):
                result = fetch_track_info("M83", "Midnight City", use_cache=False)
            check("found=True",              result["found"])
            check("bpm == 103.0",            result["bpm"] == 103.0,
                  got=result["bpm"], expected=103.0)
            check("camelot_position == 10A", result["camelot_position"] == "10A",
                  got=result["camelot_position"], expected="10A")
        finally:
            gsk_module._cache_dir = orig_cache

    # ── A9: Unicode sharp in key_of handled correctly ────────────────────────
    print("\n[A9] fetch_track_info — Unicode ♯ in key_of field")
    item_unicode = _track_item("Midnight City", "M83", "103", "C♯")
    resp_unicode = _make_response([item_unicode])

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cache = gsk_module._cache_dir
        gsk_module._cache_dir = Path(tmpdir)
        try:
            with patch("urllib.request.urlopen", return_value=resp_unicode):
                result = fetch_track_info("M83", "Midnight City", use_cache=False)
            check("found=True",              result["found"])
            check("camelot_position == 3B",  result["camelot_position"] == "3B",
                  got=result["camelot_position"], expected="3B")
        finally:
            gsk_module._cache_dir = orig_cache

    # ── A10: missing API key returns graceful miss ────────────────────────────
    print("\n[A10] fetch_track_info — missing API key returns graceful miss")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cache = gsk_module._cache_dir
        gsk_module._cache_dir = Path(tmpdir)
        try:
            with patch.dict("os.environ", {}, clear=False):
                import os
                orig_key = os.environ.pop("GETSONGKEY_API_KEY", None)
                try:
                    result = fetch_track_info("M83", "Midnight City", use_cache=False)
                finally:
                    if orig_key is not None:
                        os.environ["GETSONGKEY_API_KEY"] = orig_key
            check("found=False",             not result["found"])
            check("bpm is None",             result["bpm"] is None)
            check("camelot_position is None", result["camelot_position"] is None)
        finally:
            gsk_module._cache_dir = orig_cache

    # ── A11: network error returns graceful miss ──────────────────────────────
    print("\n[A11] fetch_track_info — network error returns graceful miss")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cache = gsk_module._cache_dir
        gsk_module._cache_dir = Path(tmpdir)
        try:
            with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
                result = fetch_track_info("M83", "Midnight City", use_cache=False)
            check("found=False on error",            not result["found"])
            check("bpm is None on error",            result["bpm"] is None)
            check("camelot_position is None on err", result["camelot_position"] is None)
            check("no exception raised",             True)
        finally:
            gsk_module._cache_dir = orig_cache

    # ── A12: disk cache round-trip ────────────────────────────────────────────
    print("\n[A12] fetch_track_info — cache written on miss, read on hit")
    item_cache = _track_item("Mr. Brightside", "The Killers", "148", "F# Minor")
    resp_cache = _make_response([item_cache])

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cache = gsk_module._cache_dir
        gsk_module._cache_dir = Path(tmpdir)
        try:
            with patch("urllib.request.urlopen", return_value=resp_cache) as mock_open:
                r1 = fetch_track_info("The Killers", "Mr. Brightside")
            calls_after_first = mock_open.call_count

            with patch("urllib.request.urlopen",
                       side_effect=AssertionError("network called on cache hit")):
                r2 = fetch_track_info("The Killers", "Mr. Brightside")

            check("network called once on first fetch", calls_after_first == 1,
                  got=calls_after_first)
            check("second call returns same bpm (cache hit)",
                  r1["bpm"] == r2["bpm"], got=r2["bpm"], expected=r1["bpm"])
            check("second call returns same key (cache hit)",
                  r1["camelot_position"] == r2["camelot_position"],
                  got=r2["camelot_position"], expected=r1["camelot_position"])
            check("cache file was written",
                  len(list(Path(tmpdir).glob("*.json"))) == 1)
        finally:
            gsk_module._cache_dir = orig_cache

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION B — Integration tests (live API + cache)
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("GetSongKey Client — Integration Tests (live API)")
    print("=" * 60)
    print("  (first run hits api.getsong.co; subsequent runs use .cache_getsongkey/)")

    # ── B1: well-known track — BPM and Camelot key ───────────────────────────
    print("\n[B1] M83 — Midnight City")
    r = fetch_track_info("M83", "Midnight City")
    check("found on GetSongKey",             r["found"])
    check("bpm is a float",                  isinstance(r.get("bpm"), float))
    check("bpm ≈ 103 (within ±5)",           abs((r.get("bpm") or 0) - 103) <= 5,
          got=r.get("bpm"), expected="≈103")
    check("camelot_position == 10A",         r.get("camelot_position") == "10A",
          got=r.get("camelot_position"), expected="10A")
    print(f"      BPM: {r.get('bpm')}  Key: {r.get('camelot_position')}")

    # ── B2: second well-known track ──────────────────────────────────────────
    print("\n[B2] The Killers — Mr. Brightside")
    r = fetch_track_info("The Killers", "Mr. Brightside")
    check("found on GetSongKey",             r["found"])
    check("bpm ≈ 148 (within ±5)",           abs((r.get("bpm") or 0) - 148) <= 5,
          got=r.get("bpm"), expected="≈148")
    check("camelot_position == 3B",          r.get("camelot_position") == "3B",
          got=r.get("camelot_position"), expected="3B")
    print(f"      BPM: {r.get('bpm')}  Key: {r.get('camelot_position')}")

    # ── B3: third track — different key type (major) ─────────────────────────
    print("\n[B3] Foster the People — Pumped Up Kicks")
    r = fetch_track_info("Foster the People", "Pumped Up Kicks")
    check("found on GetSongKey",             r["found"])
    check("bpm ≈ 127 (within ±10)",          abs((r.get("bpm") or 0) - 127) <= 10,
          got=r.get("bpm"), expected="≈127")
    check("camelot_position is a string",    isinstance(r.get("camelot_position"), str))
    print(f"      BPM: {r.get('bpm')}  Key: {r.get('camelot_position')}")

    # ── B4: repeated call uses cache ─────────────────────────────────────────
    print("\n[B4] Cache hit on repeated call")
    r1 = fetch_track_info("M83", "Midnight City")
    r2 = fetch_track_info("M83", "Midnight City")
    check("both calls return same bpm", r1["bpm"] == r2["bpm"],
          got=r2["bpm"], expected=r1["bpm"])
    check("both calls return same key", r1["camelot_position"] == r2["camelot_position"],
          got=r2["camelot_position"], expected=r1["camelot_position"])

    # ── B5: unknown track handled gracefully ─────────────────────────────────
    print("\n[B5] Unknown track — graceful miss")
    r = fetch_track_info("NotAnArtistXYZ999", "NotASongABC888")
    check("found=False",                     not r["found"])
    check("bpm is None",                     r["bpm"] is None)
    check("camelot_position is None",        r["camelot_position"] is None)
    check("no exception",                    True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed  {failed} failed")
    print(f"{'='*60}\n")
    return failed == 0


if __name__ == "__main__":
    run_tests()
