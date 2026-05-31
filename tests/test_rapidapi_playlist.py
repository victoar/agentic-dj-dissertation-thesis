"""
Standalone smoke test for the RapidAPI Spotify playlist endpoint.

Run this BEFORE integrating into the project to verify the API key works,
the response shape is what we expect, and we can extract the fields we need.

Usage:
    python tests/test_rapidapi_playlist.py

Requires .env to contain RAPID_API_KEY.
No other project dependencies are used — this is intentionally isolated.
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL  = "https://spotify81.p.rapidapi.com"
API_KEY   = os.getenv("RAPID_API_KEY")
HEADERS   = {
    "x-rapidapi-key":  API_KEY,
    "x-rapidapi-host": "spotify81.p.rapidapi.com",
    "Content-Type":    "application/json",
}

# A well-known public playlist used as a fixture: "Today's Top Hits"
TEST_PLAYLIST_ID = "3CdfoioAm3TCoieKyYPPys"


def check(label: str, condition: bool, got=None) -> bool:
    if condition:
        print(f"  ✓  {label}")
    else:
        print(f"  ✗  {label}  (got: {got!r})")
    return condition


def run_tests() -> bool:
    passed = failed = 0

    def c(label, condition, got=None):
        nonlocal passed, failed
        ok = check(label, condition, got)
        passed += ok
        failed += (not ok)
        return ok

    print("\n" + "=" * 60)
    print("RapidAPI Spotify Playlist — Smoke Tests")
    print("=" * 60)

    # ── Pre-flight: key loaded ────────────────────────────────
    print("\n[0] Pre-flight")
    if not c("RAPID_API_KEY is set in .env", bool(API_KEY), got=API_KEY):
        print("\n  Cannot continue — set RAPID_API_KEY in .env first.")
        return False

    # ── Test 1: basic fetch ───────────────────────────────────
    print(f"\n[1] GET /playlist_tracks  (id={TEST_PLAYLIST_ID}, limit=5)")
    resp = requests.get(
        f"{BASE_URL}/playlist_tracks",
        headers=HEADERS,
        params={"id": TEST_PLAYLIST_ID, "limit": 5, "offset": 0},
        timeout=10,
    )

    c("HTTP status is 200", resp.status_code == 200, got=resp.status_code)
    if resp.status_code != 200:
        print(f"\n  Response body: {resp.text[:400]}")
        print(f"\n{'='*60}\nResults: {passed} passed  {failed} failed\n{'='*60}")
        return False

    data = resp.json()
    print(f"  Raw top-level keys: {list(data.keys())}")

    # ── Test 2: response shape ────────────────────────────────
    print("\n[2] Response shape")

    # The API may wrap items in different keys — detect which one is present
    items = None
    for key in ("items", "tracks", "data", "results"):
        if isinstance(data.get(key), list):
            items = data[key]
            print(f"  Track list found under key: '{key}'  ({len(items)} items)")
            break

    if items is None:
        # Some wrappers nest under tracks.items
        nested = data.get("tracks") or {}
        if isinstance(nested, dict):
            items = nested.get("items", [])
            if items:
                print(f"  Track list found under 'tracks.items'  ({len(items)} items)")

    c("Response contains a track list", items is not None and len(items) > 0,
      got=f"keys={list(data.keys())}")
    if not items:
        print(f"\n  Full response (first 800 chars):\n  {json.dumps(data)[:800]}")
        print(f"\n{'='*60}\nResults: {passed} passed  {failed} failed\n{'='*60}")
        return False

    # ── Test 3: item structure ────────────────────────────────
    print("\n[3] Item structure  (first item)")
    first = items[0]
    print(f"  First item keys: {list(first.keys())[:12]}")

    # Tracks may be nested under item["track"] (Spotify-style) or flat
    track = first.get("track") or first
    print(f"  Track keys: {list(track.keys())[:12]}")

    c("track has 'id'",          bool(track.get("id")),           got=track.get("id"))
    c("track has 'name'",        bool(track.get("name")),         got=track.get("name"))
    c("track has 'artists'",     bool(track.get("artists")),      got=type(track.get("artists")))
    c("track has 'duration_ms'", track.get("duration_ms") is not None,
      got=track.get("duration_ms"))
    c("track has 'uri'",         bool(track.get("uri")),          got=track.get("uri"))

    # Artist extraction
    artists = track.get("artists") or []
    artist_name = None
    if isinstance(artists, list) and artists:
        first_artist = artists[0]
        if isinstance(first_artist, dict):
            artist_name = first_artist.get("name")
        elif isinstance(first_artist, str):
            artist_name = first_artist
    c("artist name is extractable", bool(artist_name), got=artist_name)

    # ── Test 4: data quality spot-check ──────────────────────
    print("\n[4] Data quality  (first 5 tracks)")
    for item in items[:5]:
        t = item.get("track") or item
        name     = t.get("name", "?")
        artists  = t.get("artists") or []
        artist   = (artists[0].get("name") if isinstance(artists[0], dict)
                    else artists[0]) if artists else "?"
        dur_ms   = t.get("duration_ms", 0)
        track_id = t.get("id", "?")
        print(f"    {name} — {artist}  ({dur_ms // 1000}s)  id={track_id[:8]}…")

    # ── Test 5: offset / pagination ───────────────────────────
    print("\n[5] Pagination  (offset=5, limit=5)")
    resp2 = requests.get(
        f"{BASE_URL}/playlist_tracks",
        headers=HEADERS,
        params={"id": TEST_PLAYLIST_ID, "limit": 5, "offset": 5},
        timeout=10,
    )
    c("paginated request returns 200", resp2.status_code == 200,
      got=resp2.status_code)
    if resp2.status_code == 200:
        data2  = resp2.json()
        items2 = None
        for key in ("items", "tracks", "data", "results"):
            if isinstance(data2.get(key), list):
                items2 = data2[key]
                break
        if items2 is None:
            nested2 = data2.get("tracks") or {}
            if isinstance(nested2, dict):
                items2 = nested2.get("items", [])
        if items2:
            first_id_p1 = (items[0].get("track") or items[0]).get("id")
            first_id_p2 = (items2[0].get("track") or items2[0]).get("id")
            c("page 2 starts at a different track", first_id_p1 != first_id_p2,
              got=f"p1={first_id_p1[:8]}  p2={first_id_p2[:8]}" if first_id_p1 and first_id_p2 else None)

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Results: {passed} passed  {failed} failed")
    print(f"{'='*60}\n")
    return failed == 0


if __name__ == "__main__":
    import sys
    sys.exit(0 if run_tests() else 1)
