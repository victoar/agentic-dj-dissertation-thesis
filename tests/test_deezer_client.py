"""
Tests for the Deezer BPM client.

Section A: unit tests — offline, use monkeypatching to avoid network calls.
Section B: integration tests — hit the live Deezer API on first run, then
           use the on-disk cache on subsequent runs. Delete .cache_deezer/
           to force fresh fetches.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import agentic_dj.music.deezer_client as deezer_module
from agentic_dj.music.deezer_client import (
    fetch_bpm,
    _cache_key,
    _cache_path,
    _titles_match,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_search_response(tracks: list[dict]) -> MagicMock:
    """Return a mock that mimics urllib.request.urlopen for a search response."""
    body = json.dumps({"data": tracks}).encode("utf-8")
    cm = MagicMock()
    cm.__enter__ = lambda s: s
    cm.__exit__ = MagicMock(return_value=False)
    cm.read.return_value = body
    return cm


def _make_detail_response(bpm: float) -> MagicMock:
    body = json.dumps({"id": 1, "title": "Test", "bpm": bpm}).encode("utf-8")
    cm = MagicMock()
    cm.__enter__ = lambda s: s
    cm.__exit__ = MagicMock(return_value=False)
    cm.read.return_value = body
    return cm


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
    print("Deezer Client — Unit Tests (offline)")
    print("=" * 60)

    # ── A1: _titles_match ────────────────────────────────────────────────────
    print("\n[A1] _titles_match — fuzzy title comparison")
    check("exact match",                    _titles_match("Midnight City", "Midnight City"))
    check("case-insensitive match",         _titles_match("midnight city", "MIDNIGHT CITY"))
    check("partial overlap (>=50%)",        _titles_match("Midnight City (Remix)", "Midnight City"))
    check("completely different → False",   not _titles_match("Blue", "Midnight City"))
    check("empty string → False",           not _titles_match("", "Midnight City"))

    # ── A2: _cache_key is stable ─────────────────────────────────────────────
    print("\n[A2] _cache_key — stable, case-insensitive")
    k1 = _cache_key("M83", "Midnight City")
    k2 = _cache_key("m83", "midnight city")
    k3 = _cache_key("M83", "Different Track")
    check("same artist/title → same key",    k1 == k2, got=k1, expected=k2)
    check("different title → different key", k1 != k3, got=k1)
    check("key is 16 hex chars",             len(k1) == 16 and k1.isalnum())

    # ── A3: successful two-step fetch ───────────────────────────────────────
    print("\n[A3] fetch_bpm — successful two-step fetch (mocked network)")
    search_resp = _make_search_response([
        {"id": 92198180, "title": "Midnight City", "artist": {"name": "M83"}}
    ])
    detail_resp = _make_detail_response(104.9)

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cache = deezer_module._cache_dir
        deezer_module._cache_dir = Path(tmpdir)
        try:
            with patch("urllib.request.urlopen", side_effect=[search_resp, detail_resp]):
                result = fetch_bpm("M83", "Midnight City", use_cache=False)
            check("found=True",          result["found"])
            check("bpm ≈ 104.9",        abs(result["bpm"] - 104.9) < 0.1,
                  got=result["bpm"], expected=104.9)
        finally:
            deezer_module._cache_dir = orig_cache

    # ── A4: no matching search result ────────────────────────────────────────
    print("\n[A4] fetch_bpm — no result in search response")
    empty_resp = _make_search_response([])
    with patch("urllib.request.urlopen", return_value=empty_resp):
        result = fetch_bpm("NotAnArtist", "NotATrack", use_cache=False)
    check("found=False",  not result["found"])
    check("bpm is None",  result["bpm"] is None)

    # ── A5: bpm=0 in detail treated as miss ─────────────────────────────────
    print("\n[A5] fetch_bpm — bpm=0 in detail response treated as miss")
    search_resp2 = _make_search_response([
        {"id": 1, "title": "Test Track", "artist": {"name": "Test Artist"}}
    ])
    detail_zero = _make_detail_response(0)
    with patch("urllib.request.urlopen", side_effect=[search_resp2, detail_zero]):
        result = fetch_bpm("Test Artist", "Test Track", use_cache=False)
    check("found=False when bpm=0", not result["found"])
    check("bpm is None",            result["bpm"] is None)

    # ── A6: network error returns graceful miss ──────────────────────────────
    print("\n[A6] fetch_bpm — network error returns graceful miss")
    with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
        result = fetch_bpm("M83", "Midnight City", use_cache=False)
    check("found=False on error",   not result["found"])
    check("bpm is None on error",   result["bpm"] is None)
    check("no exception raised",    True)

    # ── A7: disk cache round-trip ────────────────────────────────────────────
    print("\n[A7] fetch_bpm — cache written on miss, read on hit")
    search_resp3 = _make_search_response([
        {"id": 5, "title": "Pumped Up Kicks", "artist": {"name": "Foster the People"}}
    ])
    detail_resp3 = _make_detail_response(128.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        orig_cache = deezer_module._cache_dir
        deezer_module._cache_dir = Path(tmpdir)
        try:
            with patch("urllib.request.urlopen", side_effect=[search_resp3, detail_resp3]) as mock_open:
                r1 = fetch_bpm("Foster the People", "Pumped Up Kicks")
            calls_after_first = mock_open.call_count

            # Second call — should hit disk cache, no network
            with patch("urllib.request.urlopen", side_effect=AssertionError("network called on cache hit")):
                r2 = fetch_bpm("Foster the People", "Pumped Up Kicks")

            check("network called exactly twice on first fetch", calls_after_first == 2,
                  got=calls_after_first)
            check("second call returns same bpm (cache hit)", r1["bpm"] == r2["bpm"],
                  got=r2["bpm"], expected=r1["bpm"])
            check("cache file was written",
                  len(list(Path(tmpdir).glob("*.json"))) == 1)
        finally:
            deezer_module._cache_dir = orig_cache

    # ═════════════════════════════════════════════════════════════════════════
    #  SECTION B — Integration tests (live API + cache)
    # ═════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("Deezer Client — Integration Tests (live API)")
    print("=" * 60)
    print("  (first run hits Deezer; subsequent runs use .cache_deezer/)")

    # ── B1: well-known track returns a realistic BPM ─────────────────────────
    print("\n[B1] M83 — Midnight City")
    r = fetch_bpm("M83", "Midnight City")
    check("found on Deezer",          r["found"])
    check("bpm is a float",           isinstance(r["bpm"], float))
    check("bpm in plausible range (80–200)", 80 <= (r["bpm"] or 0) <= 200,
          got=r["bpm"])
    check("bpm ≈ 105 (within ±10)",   abs((r["bpm"] or 0) - 105) <= 10,
          got=r["bpm"], expected="≈105")
    print(f"      BPM: {r['bpm']}")

    # ── B2: second well-known track ──────────────────────────────────────────
    print("\n[B2] The Killers — Mr. Brightside")
    r = fetch_bpm("The Killers", "Mr. Brightside")
    check("found on Deezer",          r["found"])
    check("bpm ≈ 148 (within ±10)",   abs((r["bpm"] or 0) - 148) <= 10,
          got=r["bpm"], expected="≈148")
    print(f"      BPM: {r['bpm']}")

    # ── B3: repeated call uses cache ────────────────────────────────────────
    print("\n[B3] Cache hit on repeated call")
    r1 = fetch_bpm("M83", "Midnight City")
    r2 = fetch_bpm("M83", "Midnight City")
    check("both calls return same bpm", r1["bpm"] == r2["bpm"],
          got=r2["bpm"], expected=r1["bpm"])

    # ── B4: unknown track handled gracefully ─────────────────────────────────
    print("\n[B4] Unknown track — graceful miss")
    r = fetch_bpm("NotAnArtistXYZ999", "NotASongABC888")
    check("found=False",   not r["found"])
    check("bpm is None",   r["bpm"] is None)
    check("no exception",  True)

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed  {failed} failed")
    print(f"{'='*60}\n")
    return failed == 0


if __name__ == "__main__":
    run_tests()
