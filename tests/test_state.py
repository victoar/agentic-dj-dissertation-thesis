from agentic_dj.agent.state import (
    ListenerState, Track, FeedbackEvent, ArcPhase,
    update_state, advance_track, init_state
)

def make_track(name, energy=0.7, valence=0.6, familiar=False):
    return Track(
        id=name.lower().replace(" ", "_"),
        name=name, artist="Test Artist",
        energy_est=energy, valence_est=valence,
        familiar=familiar
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

    print("\n" + "=" * 50)
    print("Listener State Model — Unit Tests")
    print("=" * 50)

    # ── Test 1: Context presets ───────────────────────────────
    print("\n[1] Context presets")
    s = init_state("workout")
    check("workout energy starts high",  s.energy >= 0.6)
    check("workout social starts low",   s.social <= 0.3)
    s = init_state("study")
    check("study focus starts high",     s.focus >= 0.7)
    check("study energy starts low",     s.energy <= 0.4)
    s = init_state("party")
    check("party social starts high",    s.social >= 0.8)

    # ── Test 2: Skip on high-energy track lowers energy ───────
    print("\n[2] Skip on high-energy track")
    s = init_state("general")
    energy_before = s.energy
    high_track = make_track("Loud Song", energy=0.9)
    s = update_state(s, FeedbackEvent.SKIP, high_track)
    check("energy drops after skip",     s.energy < energy_before,
          got=s.energy, expected=f"< {energy_before}")
    check("openness drops after skip",   s.openness < 0.7)

    # ── Test 3: Full listen on high-energy track raises energy ─
    print("\n[3] Full listen on high-energy track")
    s = init_state("general")
    energy_before = s.energy
    s = update_state(s, FeedbackEvent.FULL_LISTEN, high_track)
    check("energy rises after full listen", s.energy > energy_before,
          got=s.energy, expected=f"> {energy_before}")
    check("openness rises after full listen", s.openness > 0.7)

    # ── Test 4: Early skip is stronger than regular skip ──────
    print("\n[4] Early skip vs regular skip")
    s1 = init_state("general")
    s2 = init_state("general")
    track = make_track("Test", energy=0.8)
    s1 = update_state(s1, FeedbackEvent.EARLY_SKIP, track)
    s2 = update_state(s2, FeedbackEvent.SKIP, track)
    check("early skip drops energy more", s1.energy < s2.energy,
          got=s1.energy, expected=f"< {s2.energy}")

    # ── Test 5: Replay raises valence ─────────────────────────
    print("\n[5] Replay signal")
    s = init_state("general")
    valence_before = s.valence
    happy_track = make_track("Happy Song", valence=0.9)
    s = update_state(s, FeedbackEvent.REPLAY, happy_track)
    check("valence rises after replay",  s.valence > valence_before)
    check("openness drops after replay", s.openness < 0.7)
    check("focus rises after replay",    s.focus > 0.5)

    # ── Test 6: Skip on LOW-energy track raises energy ────────
    print("\n[6] Skip on low-energy track (direction reversal)")
    s = init_state("general")
    energy_before = s.energy
    low_track = make_track("Quiet Song", energy=0.2)
    s = update_state(s, FeedbackEvent.SKIP, low_track)
    check("energy RISES after skipping low-energy track",
          s.energy > energy_before,
          got=s.energy, expected=f"> {energy_before}")

    # ── Test 7: Values always stay in [0, 1] ──────────────────
    print("\n[7] Clamping — values stay in [0.0, 1.0]")
    s = ListenerState(energy=0.98, valence=0.02)
    track = make_track("Edge", energy=0.9, valence=0.1)
    for _ in range(10):
        s = update_state(s, FeedbackEvent.THUMBS_UP, track)
    check("energy never exceeds 1.0", s.energy <= 1.0, got=s.energy)
    s2 = ListenerState(energy=0.02)
    for _ in range(10):
        s2 = update_state(s2, FeedbackEvent.THUMBS_DOWN, track)
    check("energy never goes below 0.0", s2.energy >= 0.0, got=s2.energy)

    # ── Test 8: Arc phase advances correctly ──────────────────
    print("\n[8] Arc phase transitions")
    s = init_state("general")
    track = make_track("Any Track")
    check("starts in WARMUP", s.arc_phase == ArcPhase.WARMUP)
    for _ in range(4):
        s = advance_track(s, track)
    check("moves to BUILD after 4 tracks", s.arc_phase == ArcPhase.BUILD,
          got=s.arc_phase)
    for _ in range(8):
        s = advance_track(s, track)
    check("moves to PEAK after 8 more",   s.arc_phase == ArcPhase.PEAK,
          got=s.arc_phase)
    for _ in range(4):
        s = advance_track(s, track)
    check("moves to COOLDOWN after 4 more", s.arc_phase == ArcPhase.COOLDOWN,
          got=s.arc_phase)

    # ── Test 9: Full session simulation ───────────────────────
    print("\n[9] Simulated session — 6 events")
    s = init_state("workout")
    events = [
        (FeedbackEvent.FULL_LISTEN, make_track("T1", energy=0.8)),
        (FeedbackEvent.FULL_LISTEN, make_track("T2", energy=0.7)),
        (FeedbackEvent.SKIP,        make_track("T3", energy=0.3)),  # too calm
        (FeedbackEvent.SKIP,        make_track("T4", energy=0.3)),  # still too calm
        (FeedbackEvent.THUMBS_UP,   make_track("T5", energy=0.9)),
        (FeedbackEvent.REPLAY,      make_track("T6", energy=0.8)),
    ]
    print(f"  Start: {s.summary()}")
    for ev, tr in events:
        s = update_state(s, ev, tr)
    print(f"  End:   {s.summary()}")
    check("energy stays high in workout session", s.energy >= 0.6,
          got=s.energy)
    check("history logged correctly", len(s._history) == 6,
          got=len(s._history))

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed  {failed} failed")
    print(f"{'='*50}\n")
    return failed == 0


if __name__ == "__main__":
    run_tests()
