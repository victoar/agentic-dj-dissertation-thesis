from agentic_dj.music.camelot import (
    CamelotKey, from_spotify, parse, compatible,
    compatibility_strength, compatible_positions, tracks_compatible,
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
    print("Camelot Wheel — Unit Tests")
    print("=" * 50)

    # ── Test 1: Spotify → Camelot conversion ────────────────
    print("\n[1] Spotify (key, mode) → Camelot position")
    # C major → 8B
    k = from_spotify(0, 1)
    check("C major → 8B", k.position == "8B", got=k.position)
    # A minor → 8A
    k = from_spotify(9, 0)
    check("A minor → 8A", k.position == "8A", got=k.position)
    # Ab major → 4B (this is M83 Midnight City)
    k = from_spotify(8, 1)
    check("Ab major → 4B", k.position == "4B", got=k.position)

    # ── Test 2: Parse string positions ───────────────────────
    print("\n[2] Parse string positions")
    k = parse("8B")
    check("parse '8B'", k.number == 8 and k.letter == "B")
    k = parse("12A")
    check("parse '12A'", k.number == 12 and k.letter == "A")
    check("parse invalid '13A'", parse("13A") is None)
    check("parse invalid '8C'", parse("8C") is None)

    # ── Test 3: Identical keys are compatible ───────────────
    print("\n[3] Compatibility — identical keys")
    k = parse("8B")
    check("8B → 8B compatible", compatible(k, k))
    check("8B → 8B strength 1.0", compatibility_strength(k, k) == 1.0)

    # ── Test 4: Adjacent positions, same letter ─────────────
    print("\n[4] Compatibility — adjacent on wheel")
    k1, k2 = parse("8B"), parse("9B")
    check("8B → 9B compatible", compatible(k1, k2))
    k1, k2 = parse("8B"), parse("7B")
    check("8B → 7B compatible (other direction)", compatible(k1, k2))
    # Wheel wrap: 12B → 1B
    k1, k2 = parse("12B"), parse("1B")
    check("12B → 1B compatible (wheel wraps)", compatible(k1, k2))
    k1, k2 = parse("1B"), parse("12B")
    check("1B → 12B compatible (wheel wraps)", compatible(k1, k2))

    # ── Test 5: Relative major/minor (same number) ──────────
    print("\n[5] Compatibility — relative major/minor")
    k1, k2 = parse("8B"), parse("8A")
    check("8B → 8A compatible (C major ↔ A minor)", compatible(k1, k2))
    check("strength around 0.75", abs(compatibility_strength(k1, k2) - 0.75) < 0.01)

    # ── Test 6: Incompatible jumps ──────────────────────────
    print("\n[6] Compatibility — incompatible jumps")
    k1, k2 = parse("8B"), parse("2B")   # C major → F# major (tritone)
    check("8B → 2B incompatible (tritone)", not compatible(k1, k2))
    k1, k2 = parse("8B"), parse("11B")  # 3 steps apart
    check("8B → 11B incompatible (3 steps apart)", not compatible(k1, k2))

    # ── Test 7: compatible_positions returns correct set ────
    print("\n[7] compatible_positions — filter for candidate search")
    k = parse("8B")
    result = compatible_positions(k)
    expected_set = {"8B", "7B", "9B", "8A"}
    check(f"8B compatible set = {expected_set}",
          set(result) == expected_set, got=set(result), expected=expected_set)

    # Wheel wrap test
    k = parse("1B")
    result = compatible_positions(k)
    expected_set = {"1B", "12B", "2B", "1A"}
    check(f"1B compatible set wraps around wheel",
          set(result) == expected_set, got=set(result), expected=expected_set)

    # ── Test 8: Strength ordering ──────────────────────────
    print("\n[8] Compatibility strength ordering")
    k1 = parse("8B")
    s_same      = compatibility_strength(k1, parse("8B"))   # identical
    s_adjacent  = compatibility_strength(k1, parse("9B"))   # adjacent
    s_relative  = compatibility_strength(k1, parse("8A"))   # relative minor
    s_two_steps = compatibility_strength(k1, parse("10B"))  # two steps
    s_tritone   = compatibility_strength(k1, parse("2B"))   # incompatible

    check("identical > adjacent",  s_same > s_adjacent)
    check("adjacent > relative",   s_adjacent > s_relative)
    check("relative > two_steps",  s_relative > s_two_steps)
    check("two_steps > tritone",   s_two_steps > s_tritone)

    # ── Test 9: Direct Spotify-format check ────────────────
    print("\n[9] tracks_compatible — end-to-end Spotify-format")
    # C major (0,1) → G major (7,1) — adjacent, compatible
    check("C major → G major compatible",
          tracks_compatible(0, 1, 7, 1))
    # C major (0,1) → F# major (6,1) — tritone, incompatible
    check("C major → F# major incompatible (tritone)",
          not tracks_compatible(0, 1, 6, 1))
    # Ab major (8,1) → F minor (5,0) — relative, compatible
    check("Ab major → F minor compatible (relative)",
          tracks_compatible(8, 1, 5, 0))

    # ── Test 10: Real-world scenario (M83 → Foster the People) ─
    print("\n[10] Real transition — M83 → Foster the People")
    # Midnight City is in Ab major → 4B
    # Pumped Up Kicks is in F major → 7B
    # That's 3 positions apart — should NOT be compatible
    k1 = from_spotify(8, 1)  # Ab major
    k2 = from_spotify(5, 1)  # F major
    check(f"Midnight City (4B) → Pumped Up Kicks (7B) distance check",
          not compatible(k1, k2))
    # But Ab major → Eb major (9B→5B... wait, let's check adjacency)
    k2 = from_spotify(3, 1)  # Eb major → 5B
    check(f"Midnight City (4B) → Eb major (5B) compatible",
          compatible(k1, k2))

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"Results: {passed} passed  {failed} failed")
    print(f"{'='*50}\n")
    return failed == 0


if __name__ == "__main__":
    run_tests()
