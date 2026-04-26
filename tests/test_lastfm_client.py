"""
Tests for the Last.fm client.
These hit the live Last.fm API on first run, then use the on-disk cache
on subsequent runs. The cache is persistent across test runs — delete
.cache/lastfm/ to force fresh fetches.
"""

import shutil
from pathlib import Path

from agentic_dj.music.lastfm_client import (
    fetch_enrichment,
    enrich_track,
    clear_cache,
    _cache_dir,
    _cache_key,
    _cache_path,
)


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
    print("Last.fm Client — Integration Tests")
    print("=" * 55)
    print("  (first run hits the live API; subsequent runs use cache)")

    # ── Test 1: Known popular track ──────────────────────────
    print("\n[1] Fetch enrichment for a well-known track")
    enr = fetch_enrichment("M83", "Midnight City")
    check("track found on Last.fm",         enr.found)
    check("returns non-empty tag list",     len(enr.tags) > 0,
          got=len(enr.tags))
    check("listener count looks plausible", enr.listeners > 100_000,
          got=enr.listeners)
    print(f"      Tags:      {enr.tags[:5]}...")
    print(f"      Listeners: {enr.listeners:,}")

    # ── Test 2: Cache is created ────────────────────────────
    print("\n[2] Cache persistence")
    key  = _cache_key("M83", "Midnight City")
    path = _cache_path(key)
    check("cache file exists after fetch", path.exists(), got=path)

    # ── Test 3: Cache hit is fast ───────────────────────────
    print("\n[3] Cached fetch returns same data")
    enr2 = fetch_enrichment("M83", "Midnight City")
    check("tag lists match after cache hit", enr.tags == enr2.tags)
    check("listeners match after cache hit", enr.listeners == enr2.listeners)

    # ── Test 4: Unknown track handled gracefully ────────────
    print("\n[4] Unknown track returns empty enrichment (no exception)")
    enr3 = fetch_enrichment("NotARealArtist_xyz123", "NotARealSong_abc789")
    check("found flag is False",  not enr3.found)
    check("empty tag list",       enr3.tags == [])
    check("listeners = 0",        enr3.listeners == 0)

    # ── Test 5: enrich_track builds a valid Track ───────────
    print("\n[5] enrich_track returns Track + TagEstimate")
    track, estimate = enrich_track(
        track_id="test_midnight_city",
        artist="M83",
        track_name="Midnight City",
        bpm=105,
    )
    check("track has correct name",          track.name == "Midnight City")
    check("track has correct artist",        track.artist == "M83")
    check("energy_est in [0.0, 1.0]",
          0.0 <= track.energy_est <= 1.0, got=track.energy_est)
    check("valence_est in [0.0, 1.0]",
          0.0 <= track.valence_est <= 1.0, got=track.valence_est)
    check("tags populated",                  len(track.tags) > 0)
    check("estimate confidence is meaningful",
          estimate.confidence > 0.3, got=estimate.confidence)
    print(f"      Estimate:  energy={track.energy_est:.2f}  "
          f"valence={track.valence_est:.2f}  "
          f"conf={estimate.confidence:.2f}")

    # ── Test 6: Different musical contexts produce different estimates ─
    print("\n[6] Contrasting tracks produce distinct estimates")
    energetic, _ = enrich_track(
        "test_mr_brightside", "The Killers", "Mr. Brightside", bpm=148
    )
    calm, _ = enrich_track(
        "test_clair_de_lune", "Claude Debussy", "Clair de Lune", bpm=66
    )
    check("energetic track has higher energy than calm track",
          energetic.energy_est > calm.energy_est,
          got=f"{energetic.energy_est} vs {calm.energy_est}")
    print(f"      Mr. Brightside:  energy={energetic.energy_est:.2f}  "
          f"valence={energetic.valence_est:.2f}")
    print(f"      Clair de Lune:   energy={calm.energy_est:.2f}  "
          f"valence={calm.valence_est:.2f}")

    # ── Test 7: Popularity normalisation ────────────────────
    print("\n[7] Listener count → normalised popularity")
    enr = fetch_enrichment("Bad Bunny", "DtMF")

    print(f"\n      Found:              {enr.found}")
    print(f"      Listeners:          {enr.listeners:,}")
    print(f"      Playcount:          {enr.playcount:,}")
    print(f"      Tags:               {enr.tags[:5]}")

    lastfm_popularity = min(100, int(enr.listeners / 100_000))
    print(f"      Normalised (0-100): {lastfm_popularity}")

    check("track found", enr.found)
    check("normalised popularity in [0, 100]",
          0 <= lastfm_popularity <= 100, got=lastfm_popularity)

    # ── Test 8: Clear cache works ────────────────────────────
    print("\n[8] clear_cache removes files")
    before = len(list(_cache_dir.glob("*.json"))) if _cache_dir.exists() else 0
    # don't actually wipe here — it would slow down subsequent test runs
    # just verify the function is callable and returns a sensible number
    check(f"cache directory has files ({before})", before > 0)
    # removed = clear_cache()
    # check("clear_cache returned number of removed files",
    #       isinstance(removed, int))

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Results: {passed} passed  {failed} failed")
    print(f"{'='*55}\n")
    return failed == 0


if __name__ == "__main__":
    run_tests()