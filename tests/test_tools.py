"""
Agent tools tests.
Tests each tool function in isolation before wiring them into the agent loop.
"""

from agentic_dj.agent.tools import (
    get_listener_state,
    update_listener_state,
    get_session_arc,
    get_compatible_keys,
    check_transition,
    estimate_bpm_compatibility,
    search_tracks,
    get_track_details,
    get_current_playback,
    get_queue_state,
    get_session_history,
    reset_session,
)


def run_tests():
    passed = 0
    failed = 0

    def check(label, condition, got=None):
        nonlocal passed, failed
        if condition:
            print(f"  ✓  {label}")
            passed += 1
        else:
            print(f"  ✗  {label}  (got {got})")
            failed += 1

    print("\n" + "=" * 55)
    print("Agent Tools — Unit + Integration Tests")
    print("=" * 55)

    # ── 1. reset_session ────────────────────────────────────
    print("\n[1] reset_session")
    r = reset_session("workout")
    check("returns reset=True",        r["reset"] is True)
    check("context is set",            r["context"] == "workout")
    check("state has energy key",      "energy" in r["state"])
    check("workout energy starts high",r["state"]["energy"] >= 0.6,
          got=r["state"]["energy"])

    # ── 2. get_listener_state ────────────────────────────────
    print("\n[2] get_listener_state")
    state = get_listener_state()
    for dim in ["energy", "valence", "focus", "openness", "social", "arc_phase"]:
        check(f"state has '{dim}' key", dim in state)
    check("arc starts at warmup", state["arc_phase"] == "warmup")

    # ── 3. get_session_arc ───────────────────────────────────
    print("\n[3] get_session_arc")
    arc = get_session_arc()
    check("arc_phase returned",     "arc_phase" in arc)
    check("description returned",   "description" in arc and len(arc["description"]) > 0)
    check("tracks_played returned", "tracks_played" in arc)

    # ── 4. update_listener_state ─────────────────────────────
    print("\n[4] update_listener_state")
    reset_session("general")
    before = get_listener_state()["energy"]
    r = update_listener_state("skip", "Loud Track", "Some Artist")
    check("returns updated=True", r.get("updated") is True)
    check("state returned",       "state" in r)

    r_bad = update_listener_state("invalid_event", "Track", "Artist")
    check("invalid event returns error key", "error" in r_bad)

    # ── 5. get_compatible_keys ───────────────────────────────
    print("\n[5] get_compatible_keys")
    r = get_compatible_keys("8B")
    check("returns compatible list",  "compatible" in r)
    check("returns 4 positions",      len(r["compatible"]) == 4,
          got=len(r["compatible"]))
    check("includes the key itself",
          any(c["position"] == "8B" for c in r["compatible"]))

    r_bad = get_compatible_keys("99Z")
    check("invalid position returns error", "error" in r_bad)

    # ── 6. check_transition ──────────────────────────────────
    print("\n[6] check_transition")
    r = check_transition("8B", "8B")
    check("identical keys → score 1.0",   r["score"] == 1.0)
    check("identical keys → smooth",      "smooth" in r["verdict"])

    r = check_transition("8B", "9B")
    check("adjacent keys → high score",   r["score"] >= 0.75)

    r = check_transition("8B", "2B")
    check("tritone → low score",          r["score"] < 0.4, got=r["score"])
    check("tritone → avoid in verdict",   "avoid" in r["verdict"])

    # ── 7. estimate_bpm_compatibility ───────────────────────
    print("\n[7] estimate_bpm_compatibility")
    r = estimate_bpm_compatibility(105, 111, "build")
    check("6 BPM jump acceptable in build",  r["acceptable"] is True)
    check("direction is 'up'",               r["direction"] == "up")

    r = estimate_bpm_compatibility(105, 148, "warmup")
    check("43 BPM jump rejected in warmup",  r["acceptable"] is False,
        got=r["difference"])

    r = estimate_bpm_compatibility(105, 148, "peak")
    check("43 BPM jump rejected even at peak", r["acceptable"] is False,
        got=r["difference"])

    r = estimate_bpm_compatibility(105, 120, "peak")
    check("15 BPM jump acceptable at peak",  r["acceptable"] is True)

    r = estimate_bpm_compatibility(120, 120, "build")
    check("same BPM → direction is 'same'",  r["direction"] == "same")

    r = estimate_bpm_compatibility(120, 108, "cooldown")
    check("12 BPM jump rejected in cooldown",r["acceptable"] is False,
        got=r["difference"])

    r = estimate_bpm_compatibility(120, 112, "cooldown")
    check("8 BPM jump acceptable in cooldown", r["acceptable"] is True)

    # ── 8. search_tracks ────────────────────────────────────
    print("\n[8] search_tracks (live API)")
    r = search_tracks("Midnight City M83", limit=3)
    check("returns count",          "count" in r and r["count"] > 0)
    check("returns candidates list","candidates" in r)
    c = r["candidates"][0]
    check("candidate has name",     "name" in c)
    check("candidate has artist",   "artist" in c)
    check("candidate has energy",   "energy_est" in c)
    check("candidate has tags",     "tags" in c)
    print(f"      {c['name']} — {c['artist']}")
    print(f"      energy={c['energy_est']:.2f}  "
          f"valence={c['valence_est']:.2f}  "
          f"tags={c['tags'][:3]}")

    # ── 9. get_track_details ────────────────────────────────
    print("\n[9] get_track_details (live API)")
    r = get_track_details("Mr. Brightside", "The Killers")
    check("name returned",      r.get("name") != "")
    check("artist returned",    r.get("artist") != "")
    check("energy estimated",   0.0 <= r.get("energy_est", -1) <= 1.0)
    check("valence estimated",  0.0 <= r.get("valence_est", -1) <= 1.0)
    print(f"      energy={r.get('energy_est'):.2f}  "
          f"valence={r.get('valence_est'):.2f}  "
          f"listeners={r.get('listeners'):,}")

    # ── 10. get_current_playback ─────────────────────────────
    print("\n[10] get_current_playback (live API)")
    r = get_current_playback()
    check("returns a dict", isinstance(r, dict))
    check("has playing key", "playing" in r)
    if r.get("playing"):
        check("has track_name", "track_name" in r)
        print(f"      Playing: {r.get('track_name')} — {r.get('artist')}")
        print(f"      Progress: {r.get('progress_pct')}%")
    else:
        print("      No active playback")

    # ── 11. get_queue_state + get_session_history ────────────
    print("\n[11] get_queue_state + get_session_history")
    reset_session("general")
    q = get_queue_state()
    check("queue_size starts at 0", q["queue_size"] == 0)
    h = get_session_history()
    check("history starts empty",   h["tracks_played"] == 0)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Results: {passed} passed  {failed} failed")
    print(f"{'='*55}\n")
    return failed == 0


if __name__ == "__main__":
    run_tests()