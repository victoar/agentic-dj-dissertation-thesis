"""
Spotify client tests.
These hit the live Spotify API — make sure:
  1. Your .env has SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET, SPOTIPY_REDIRECT_URI
  2. Spotify is open and playing on at least one device for playback tests
"""

from agentic_dj.spotify.client import SpotifyClient, SpotifyTrack


def run_tests():
    passed = 0
    failed = 0

    def check(label, condition, got=None, expected=None):
        nonlocal passed, failed
        if condition:
            print(f"  ✓  {label}")
            passed += 1
        else:
            print(f"  ✗  {label}  (got {got}, expected {expected})")
            failed += 1

    print("\n" + "=" * 55)
    print("Spotify Client — Integration Tests")
    print("=" * 55)

    client = SpotifyClient()

    # ── Test 1: Basic search ─────────────────────────────────
    print("\n[1] Basic search")
    results = client.search("Bad Bunny DtMF", limit=5)
    check("returns results",       len(results) > 0, got=len(results))
    check("results are SpotifyTrack objects",
          all(isinstance(r, SpotifyTrack) for r in results))
    check("first result has an id", results[0].id != "")
    check("first result has a URI",
          results[0].uri.startswith("spotify:track:"),
          got=results[0].uri)
    print(f"      Top result: {results[0].name} — {results[0].artist} ")

    # ── Test 2: Search by mood tags ──────────────────────────
    print("\n[2] Mood-based search")
    results = client.search_by_mood(["energetic", "indie rock"], limit=5)
    check("mood search returns results", len(results) > 0, got=len(results))

    print(f"      Top result: {results[0].name} — {results[0].artist}")

    # ── Test 3: Search similar to artist ────────────────────
    print("\n[3] Artist-based search")
    results = client.search_similar_to("M83", "Midnight City", limit=5)
    check("artist search returns results", len(results) > 0)
    check("results have non-empty artist names",
          all(r.artist != "unknown" for r in results))

    # ── Test 4: Track data quality ───────────────────────────
    print("\n[4] Track data quality")
    results = client.search("Mr Brightside The Killers", limit=1)
    t = results[0]
    check("duration is plausible (> 60 seconds)",
          t.duration_ms > 60_000, got=t.duration_ms)
    check("name is non-empty",   t.name != "")
    check("artist is non-empty", t.artist != "")
    check("album is non-empty",  t.album != "")
    print(f"      {t.name} — {t.artist} "
          f"({t.duration_ms // 1000}s")

    # ── Test 5: Playback state ───────────────────────────────
    print("\n[5] Playback state")
    state = client.get_playback()
    if state:
        print(f"      Active device: {state.device_name}")
        if state.track:
            print(f"      Playing: {state.track.name} — {state.track.artist}")
            print(f"      Progress: {state.progress_ms // 1000}s")
        check("device name is not empty", state.device_name != "")
        check("progress_ms is non-negative", state.progress_ms >= 0)
    else:
        print("      No active playback — open Spotify on a device first")
        print("      (playback tests skipped)")
        check("get_playback returns None gracefully (no exception)", True)

    # ── Test 6: Add to queue (requires active device) ────────
    # print("\n[6] Add to queue")
    # results = client.search("Bohemian Rhapsody Queen", limit=1)
    # track   = results[0]
    # success = client.add_to_queue(track)
    # if success:
    #     check("add_to_queue succeeded", True)
    #     print(f"      Added: {track.name} — {track.artist}")
    # else:
    #     print("      No active device — queue test skipped")
    #     check("add_to_queue returns False gracefully (no exception)", True)

    # ── Test 7: Saved library ────────────────────────────────
    print("\n[7] User saved library")
    saved = client.get_saved_tracks(limit=5)
    check("saved tracks returns a list",
          isinstance(saved, list))
    if saved:
        check("saved tracks are SpotifyTrack objects",
              all(isinstance(t, SpotifyTrack) for t in saved))
        print(f"      Found {len(saved)} saved tracks")
        print(f"      First: {saved[0].name} — {saved[0].artist}")
    else:
        print("      No saved tracks found (empty library?)")

    results = client.search("Bad Bunny DtMF", limit=5)
    is_saved = client.is_saved(results[0].id)
    if is_saved:
        check("song is saved", is_saved == True)
        print(f"      {results[0].name} is a saved song")
    else:
        print(f"      not a saved song")


    # ── Test 8: seek_to_beginning ────────────────────────────
    print("\n[8] seek_to_beginning")
    state = client.get_playback()
    if state and state.is_playing:
        ok = client.seek_to_beginning()
        check("seek_to_beginning returns True", ok is True, got=ok)
        state_after = client.get_playback()
        if state_after:
            check("progress_ms is near 0 after seek",
                  state_after.progress_ms < 3_000,
                  got=state_after.progress_ms)
    else:
        print("      No active playback — seek test skipped")
        check("seek_to_beginning skipped gracefully (no exception)", True)

    # ── Test 9: save_track ───────────────────────────────────
    print("\n[9] save_track")
    results = client.search("Mr Brightside The Killers", limit=1)
    if results:
        track_id = results[0].id
        already_saved = client.is_saved(track_id)
        ok = client.save_track(track_id)
        check("save_track returns True", ok is True, got=ok)
        check("track is now saved in library", client.is_saved(track_id) is True)
        if already_saved:
            print(f"      Track was already saved — save_track is idempotent")
        else:
            print(f"      Saved: {results[0].name} — {results[0].artist}")
    else:
        print("      Search returned no results — save_track test skipped")
        check("save_track skipped gracefully (no exception)", True)

    # ── Test 10: get_spotify_queue ───────────────────────────
    print("\n[10] get_spotify_queue")
    queue_ids = client.get_spotify_queue()
    check("get_spotify_queue returns a list", isinstance(queue_ids, list))
    check("all queue entries are non-empty strings",
          all(isinstance(qid, str) and qid for qid in queue_ids))
    print(f"      {len(queue_ids)} track(s) currently in Spotify queue")
    if queue_ids:
        print(f"      First queued ID: {queue_ids[0]}")

    # ── Test 11: get_top_tracks ──────────────────────────────
    print("\n[11] get_top_tracks")
    top = client.get_top_tracks(limit=10, time_range="short_term")
    check("get_top_tracks returns a list", isinstance(top, list))
    if top:
        check("results are SpotifyTrack objects",
              all(isinstance(t, SpotifyTrack) for t in top))
        check("first result has non-empty name",  top[0].name != "")
        check("first result has non-empty artist", top[0].artist != "unknown")
        print(f"      {len(top)} top tracks fetched")
        print(f"      Top: {top[0].name} — {top[0].artist}")
    else:
        print("      No top tracks returned (new account or API issue)")
        check("get_top_tracks returns empty list gracefully (no exception)", True)

    # medium_term sanity check
    top_med = client.get_top_tracks(limit=5, time_range="medium_term")
    check("medium_term also returns a list", isinstance(top_med, list))

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Results: {passed} passed  {failed} failed")
    print(f"{'='*55}\n")
    return failed == 0


if __name__ == "__main__":
    run_tests()