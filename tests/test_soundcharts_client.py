"""
Tests for the Soundcharts client.

Section A: unit tests — offline, mock urllib to avoid network calls.
Section B: integration tests — hit customer.api.soundcharts.com on first run,
           then use the on-disk cache on subsequent runs.
           Requires SOUNDCHARTS_APP_ID and SOUNDCHARTS_API_KEY in .env.
           Delete .cache_soundcharts/ to force fresh fetches.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import agentic_dj.music.soundcharts_client as sc_module
from agentic_dj.music.soundcharts_client import fetch_by_platform, fetch_song, fetch_track_info


# ── Mock helpers ──────────────────────────────────────────────────────────────

def _mock_urlopen(body: dict) -> MagicMock:
    """Build a urlopen context-manager mock that returns the given JSON body."""
    raw = json.dumps(body).encode("utf-8")
    cm = MagicMock()
    cm.__enter__ = lambda s: s
    cm.__exit__ = MagicMock(return_value=False)
    cm.read.return_value = raw
    return cm


def _platform_response(uuid: str, name: str = "Test Song") -> dict:
    return {"type": "song", "object": {"uuid": uuid, "name": name, "creditName": "Test Artist"}, "errors": []}


def _song_response(uuid: str, bpm: float = 120.0, key: str = "A minor") -> dict:
    return {"type": "song", "object": {"uuid": uuid, "name": "Test Song", "bpm": bpm, "key": key}, "errors": []}


# ── Test runner ───────────────────────────────────────────────────────────────

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

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION A — Unit tests (no network)
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("Soundcharts Client — Unit Tests (offline)")
    print("=" * 60)

    # ── A1: fetch_by_platform — successful response ────────────────────────────
    print("\n[A1] fetch_by_platform — successful mocked response")
    mock_resp = _mock_urlopen(_platform_response("abc-123-uuid"))

    with tempfile.TemporaryDirectory() as tmpdir:
        orig = sc_module._cache_dir
        sc_module._cache_dir = Path(tmpdir)
        try:
            with patch.dict(sc_module.os.environ, {"SOUNDCHARTS_APP_ID": "fake-id", "SOUNDCHARTS_API_KEY": "fake-key"}):
                with patch("urllib.request.urlopen", return_value=mock_resp):
                    r = fetch_by_platform("spotify", "FAKE_TRACK_ID", use_cache=False)
            check("found=True",        r["found"])
            check("uuid extracted",    r["uuid"] == "abc-123-uuid",
                  got=r["uuid"], expected="abc-123-uuid")
            check("raw is non-empty",  bool(r["raw"]))
        finally:
            sc_module._cache_dir = orig

    # ── A2: fetch_by_platform — missing API credentials ───────────────────────
    print("\n[A2] fetch_by_platform — missing credentials returns found=False")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = sc_module._cache_dir
        sc_module._cache_dir = Path(tmpdir)
        orig_app_id  = sc_module.os.environ.pop("SOUNDCHARTS_APP_ID",  None)
        orig_api_key = sc_module.os.environ.pop("SOUNDCHARTS_API_KEY", None)
        try:
            r = fetch_by_platform("spotify", "FAKE_TRACK_ID", use_cache=False)
            check("found=False",  not r["found"])
            check("uuid is None", r["uuid"] is None)
            check("raw is empty", r["raw"] == {})
        finally:
            sc_module._cache_dir = orig
            if orig_app_id  is not None: sc_module.os.environ["SOUNDCHARTS_APP_ID"]  = orig_app_id
            if orig_api_key is not None: sc_module.os.environ["SOUNDCHARTS_API_KEY"] = orig_api_key

    # ── A3: fetch_by_platform — network error ─────────────────────────────────
    print("\n[A3] fetch_by_platform — network error returns found=False, no raise")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = sc_module._cache_dir
        sc_module._cache_dir = Path(tmpdir)
        try:
            with patch.dict(sc_module.os.environ, {"SOUNDCHARTS_APP_ID": "fake-id", "SOUNDCHARTS_API_KEY": "fake-key"}):
                with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
                    r = fetch_by_platform("spotify", "FAKE_TRACK_ID", use_cache=False)
            check("found=False on error",  not r["found"])
            check("uuid is None on error", r["uuid"] is None)
            check("no exception raised",   True)
        finally:
            sc_module._cache_dir = orig

    # ── A4: fetch_by_platform — cache round-trip ──────────────────────────────
    print("\n[A4] fetch_by_platform — cache written on first call, read on second")
    mock_resp2 = _mock_urlopen(_platform_response("cached-uuid"))

    with tempfile.TemporaryDirectory() as tmpdir:
        orig = sc_module._cache_dir
        sc_module._cache_dir = Path(tmpdir)
        try:
            with patch.dict(sc_module.os.environ, {"SOUNDCHARTS_APP_ID": "fake-id", "SOUNDCHARTS_API_KEY": "fake-key"}):
                with patch("urllib.request.urlopen", return_value=mock_resp2) as mock_open:
                    r1 = fetch_by_platform("spotify", "TRACK_XYZ")
                calls_first = mock_open.call_count

                with patch("urllib.request.urlopen",
                           side_effect=AssertionError("network called on cache hit")):
                    r2 = fetch_by_platform("spotify", "TRACK_XYZ")

            check("network called once on first fetch", calls_first == 1,
                  got=calls_first)
            check("cache hit returns same uuid",
                  r1["uuid"] == r2["uuid"], got=r2["uuid"], expected=r1["uuid"])
            check("cache file was written",
                  len(list(Path(tmpdir).glob("*.json"))) == 1)
        finally:
            sc_module._cache_dir = orig

    # ── A5: fetch_song — successful response ──────────────────────────────────
    print("\n[A5] fetch_song — successful mocked response")
    mock_song = _mock_urlopen(_song_response("abc-123-uuid", bpm=103.0, key="B minor"))

    with tempfile.TemporaryDirectory() as tmpdir:
        orig = sc_module._cache_dir
        sc_module._cache_dir = Path(tmpdir)
        try:
            with patch.dict(sc_module.os.environ, {"SOUNDCHARTS_APP_ID": "fake-id", "SOUNDCHARTS_API_KEY": "fake-key"}):
                with patch("urllib.request.urlopen", return_value=mock_song):
                    r = fetch_song("abc-123-uuid", use_cache=False)
            check("found=True",              r["found"])
            check("raw is non-empty",        bool(r["raw"]))
            check("bpm in raw",              r["raw"].get("bpm") == 103.0,
                  got=r["raw"].get("bpm"), expected=103.0)
            check("key in raw",              r["raw"].get("key") == "B minor",
                  got=r["raw"].get("key"), expected="B minor")
        finally:
            sc_module._cache_dir = orig

    # ── A6: fetch_song — missing credentials ──────────────────────────────────
    print("\n[A6] fetch_song — missing credentials returns found=False")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = sc_module._cache_dir
        sc_module._cache_dir = Path(tmpdir)
        orig_app_id  = sc_module.os.environ.pop("SOUNDCHARTS_APP_ID",  None)
        orig_api_key = sc_module.os.environ.pop("SOUNDCHARTS_API_KEY", None)
        try:
            r = fetch_song("some-uuid", use_cache=False)
            check("found=False",  not r["found"])
            check("raw is empty", r["raw"] == {})
        finally:
            sc_module._cache_dir = orig
            if orig_app_id  is not None: sc_module.os.environ["SOUNDCHARTS_APP_ID"]  = orig_app_id
            if orig_api_key is not None: sc_module.os.environ["SOUNDCHARTS_API_KEY"] = orig_api_key

    # ── A7: fetch_song — network error ────────────────────────────────────────
    print("\n[A7] fetch_song — network error returns found=False, no raise")
    with tempfile.TemporaryDirectory() as tmpdir:
        orig = sc_module._cache_dir
        sc_module._cache_dir = Path(tmpdir)
        try:
            with patch.dict(sc_module.os.environ, {"SOUNDCHARTS_APP_ID": "fake-id", "SOUNDCHARTS_API_KEY": "fake-key"}):
                with patch("urllib.request.urlopen", side_effect=OSError("timeout")):
                    r = fetch_song("some-uuid", use_cache=False)
            check("found=False on error",  not r["found"])
            check("no exception raised",   True)
        finally:
            sc_module._cache_dir = orig

    # ── A8: fetch_song — cache round-trip ─────────────────────────────────────
    print("\n[A8] fetch_song — cache written on first call, read on second")
    mock_song2 = _mock_urlopen(_song_response("cached-song-uuid", bpm=148.0))

    with tempfile.TemporaryDirectory() as tmpdir:
        orig = sc_module._cache_dir
        sc_module._cache_dir = Path(tmpdir)
        try:
            with patch.dict(sc_module.os.environ, {"SOUNDCHARTS_APP_ID": "fake-id", "SOUNDCHARTS_API_KEY": "fake-key"}):
                with patch("urllib.request.urlopen", return_value=mock_song2) as mock_open:
                    r1 = fetch_song("cached-song-uuid")
                calls_first = mock_open.call_count

                with patch("urllib.request.urlopen",
                           side_effect=AssertionError("network called on cache hit")):
                    r2 = fetch_song("cached-song-uuid")

            check("network called once on first fetch", calls_first == 1,
                  got=calls_first)
            check("cache hit returns same bpm",
                  r1["raw"].get("bpm") == r2["raw"].get("bpm"),
                  got=r2["raw"].get("bpm"), expected=r1["raw"].get("bpm"))
            check("cache file was written",
                  len(list(Path(tmpdir).glob("*.json"))) == 1)
        finally:
            sc_module._cache_dir = orig

    # ── A10: fetch_track_info — success, BPM and Camelot extracted ───────────
    print("\n[A10] fetch_track_info — success (key=1/mode=1 → Db major → 3B)")
    with patch("agentic_dj.music.soundcharts_client.fetch_by_platform") as mock_fbp:
        mock_fbp.return_value = {
            "found": True,
            "uuid": "test-uuid",
            "raw": {"audio": {"tempo": 148.11, "key": 1, "mode": 1}},
            "error": None,
        }
        r = fetch_track_info("FAKE_SPOTIFY_ID")
    check("found=True",                    r["found"])
    check("bpm == 148.11",                 r["bpm"] == 148.11,
          got=r["bpm"], expected=148.11)
    check("camelot_position == '3B'",      r["camelot_position"] == "3B",
          got=r["camelot_position"], expected="3B")

    # ── A11: fetch_track_info — platform miss ────────────────────────────────
    print("\n[A11] fetch_track_info — platform miss → all None")
    with patch("agentic_dj.music.soundcharts_client.fetch_by_platform") as mock_fbp:
        mock_fbp.return_value = {"found": False, "uuid": None, "raw": {}, "error": "404"}
        r = fetch_track_info("FAKE_SPOTIFY_ID")
    check("found=False",               not r["found"])
    check("bpm is None",               r["bpm"] is None)
    check("camelot_position is None",  r["camelot_position"] is None)

    # ── A12: fetch_track_info — missing audio block ──────────────────────────
    print("\n[A12] fetch_track_info — raw has no 'audio' key → graceful None")
    with patch("agentic_dj.music.soundcharts_client.fetch_by_platform") as mock_fbp:
        mock_fbp.return_value = {
            "found": True,
            "uuid": "test-uuid",
            "raw": {"name": "Test Song"},  # no 'audio' key
            "error": None,
        }
        r = fetch_track_info("FAKE_SPOTIFY_ID")
    check("found=True",                r["found"])
    check("bpm is None",               r["bpm"] is None)
    check("camelot_position is None",  r["camelot_position"] is None)

    # ── A9: cache keys are stable and distinct ────────────────────────────────
    print("\n[A9] Cache key stability and uniqueness")
    from agentic_dj.music.soundcharts_client import _cache_key
    k1 = _cache_key("spotify|TRACK_A")
    k2 = _cache_key("spotify|TRACK_A")
    k3 = _cache_key("spotify|TRACK_B")
    k4 = _cache_key("song|TRACK_A")
    check("same input → same key",                   k1 == k2)
    check("different identifier → different key",    k1 != k3)
    check("by-platform vs fetch_song → different key", k1 != k4)
    check("key is 16 hex chars",                     len(k1) == 16 and k1.isalnum())

    # ══════════════════════════════════════════════════════════════════════════
    #  SECTION B — Integration tests (live API + cache)
    # ══════════════════════════════════════════════════════════════════════════

    print("\n" + "=" * 60)
    print("Soundcharts Client — Integration Tests (live API)")
    print("=" * 60)
    print("  (first run hits customer.api.soundcharts.com; subsequent runs use cache)")

    import os as _os
    _app_id  = _os.getenv("SOUNDCHARTS_APP_ID")
    _api_key = _os.getenv("SOUNDCHARTS_API_KEY")
    print(f"  SOUNDCHARTS_APP_ID  : {'SET' if _app_id  else 'MISSING'}")
    print(f"  SOUNDCHARTS_API_KEY : {'SET' if _api_key else 'MISSING'}")

    # M83 "Midnight City" Spotify track ID
    MIDNIGHT_CITY_SPOTIFY_ID = "3n3Ppam7vgaVa1iaRUc9Lp"

    # ── B1: resolve Spotify ID → Soundcharts UUID ─────────────────────────────
    print("\n[B1] fetch_by_platform — M83 'Midnight City' (Spotify)")
    r = fetch_by_platform("spotify", MIDNIGHT_CITY_SPOTIFY_ID)
    check("found=True",            r["found"])
    check("uuid is a string",      isinstance(r.get("uuid"), str))
    check("uuid is non-empty",     bool(r.get("uuid")))
    print(f"      uuid:  {r.get('uuid')}")
    print(f"      error: {r.get('error')}")

    # ── B2: fetch full song metadata by UUID ──────────────────────────────────
    if r["found"] and r.get("uuid"):
        print("\n[B2] fetch_song — using UUID from B1")
        uuid = r["uuid"]
        r2 = fetch_song(uuid)
        check("found=True",          r2["found"])
        check("raw is non-empty",    bool(r2["raw"]))
        print(f"      error:    {r2.get('error')}")
        print(f"      raw keys: {list(r2['raw'].keys())}")
        if "bpm" in r2["raw"]:
            print(f"      bpm: {r2['raw']['bpm']}")
        if "key" in r2["raw"]:
            print(f"      key: {r2['raw']['key']}")
    else:
        print("\n[B2] skipped — B1 did not return a UUID")

    # ── B3: cache hit on repeated calls ───────────────────────────────────────
    print("\n[B3] Repeated calls use cache (no extra network traffic)")
    r_a = fetch_by_platform("spotify", MIDNIGHT_CITY_SPOTIFY_ID)
    r_b = fetch_by_platform("spotify", MIDNIGHT_CITY_SPOTIFY_ID)
    check("both calls agree on uuid",
          r_a.get("uuid") == r_b.get("uuid"),
          got=r_b.get("uuid"), expected=r_a.get("uuid"))

    # ── B4: unknown identifier handled gracefully ─────────────────────────────
    print("\n[B4] Unknown identifier — graceful miss")
    r = fetch_by_platform("spotify", "000000000000000000FAKE")
    check("found=False on unknown ID", not r["found"])
    check("uuid is None",              r["uuid"] is None)
    check("no exception",              True)

    # ── B5: fetch_track_info — end-to-end via Soundcharts ────────────────────
    print("\n[B5] fetch_track_info — end-to-end (Mr. Brightside)")
    r = fetch_track_info(MIDNIGHT_CITY_SPOTIFY_ID)
    check("found=True",                    r["found"])
    check("bpm is a float",                isinstance(r.get("bpm"), float))
    check("bpm in plausible range",        80 <= (r.get("bpm") or 0) <= 220,
          got=r.get("bpm"))
    check("camelot_position is a string",  isinstance(r.get("camelot_position"), str))
    print(f"      bpm: {r.get('bpm')}  camelot: {r.get('camelot_position')}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed  {failed} failed")
    print(f"{'=' * 60}\n")
    return failed == 0


if __name__ == "__main__":
    run_tests()
