from agentic_dj.music.tags import estimate_features, TAG_LEXICON


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
    print("Tag-to-Feature Translator — Unit Tests")
    print("=" * 55)

    # ── Test 1: Empty input returns neutral ────────────────
    print("\n[1] Edge cases")
    r = estimate_features([])
    check("empty tags returns neutral 0.5/0.5",
          r.energy == 0.5 and r.valence == 0.5 and r.confidence == 0.0)

    r = estimate_features(["completelyunknowntag123"])
    check("unknown tags return neutral",
          r.energy == 0.5 and r.valence == 0.5 and r.confidence == 0.0)

    # ── Test 2: Single clear tags ──────────────────────────
    print("\n[2] Single tags produce correct direction")
    r = estimate_features(["energetic"])
    check("'energetic' → high energy", r.energy > 0.7, got=r.energy)
    check("'energetic' → positive-ish valence", r.valence > 0.5, got=r.valence)

    r = estimate_features(["melancholic"])
    check("'melancholic' → low valence", r.valence < 0.3, got=r.valence)
    check("'melancholic' → low-ish energy", r.energy < 0.5, got=r.energy)

    r = estimate_features(["chill"])
    check("'chill' → low energy",  r.energy < 0.4,  got=r.energy)
    check("'chill' → positive valence", r.valence > 0.55, got=r.valence)

    # ── Test 3: Confidence scales with tag count ───────────
    print("\n[3] Confidence scales with matched tag count")
    r1 = estimate_features(["energetic"])
    r2 = estimate_features(["energetic", "upbeat", "dance", "party", "fun"])
    check("confidence rises with more matching tags", r2.confidence > r1.confidence,
          got=f"{r1.confidence} vs {r2.confidence}")
    check("5+ matches → confidence 1.0", r2.confidence == 1.0, got=r2.confidence)

    # ── Test 4: Normalisation handles real-world variations ─
    print("\n[4] Normalisation (case, spacing, dashes)")
    r1 = estimate_features(["Chillout"])
    r2 = estimate_features(["chillout"])
    r3 = estimate_features(["chill-out"])
    r4 = estimate_features(["chill_out"])
    check("'Chillout' matches",  r1.energy == r2.energy)
    check("'chill-out' matches", r3.energy == r2.energy)
    check("'chill_out' matches", r4.energy == r2.energy)

    # ── Test 5: Weighted aggregation ───────────────────────
    print("\n[5] Tag popularity weights")
    # If 'energetic' has weight 100 and 'chill' has weight 1,
    # the estimate should lean strongly toward energetic.
    r = estimate_features(["energetic", "chill"], tag_weights=[100, 1])
    check("heavily weighted 'energetic' dominates", r.energy > 0.7, got=r.energy)

    # Equal weights → averaged
    r = estimate_features(["energetic", "chill"], tag_weights=[1, 1])
    check("equal weights produce middle-ish energy",
          0.4 < r.energy < 0.6, got=r.energy)

    # ── Test 6: Realistic M83 Midnight City tags ──────────
    print("\n[6] Real-world scenario — M83 Midnight City")
    # These are typical Last.fm tags for this track
    tags = ["synthpop", "electronic", "dream pop", "indie", "atmospheric"]
    r = estimate_features(tags)
    check("synthpop+dream pop → moderate energy",
          0.4 < r.energy < 0.75, got=r.energy)
    check("moderate, slightly positive valence",
          0.4 < r.valence < 0.7, got=r.valence)
    check("confidence is meaningful (>0.6)", r.confidence > 0.6,
          got=r.confidence)

    # ── Test 7: Sad classical track ────────────────────────
    print("\n[7] Real-world scenario — Sad classical")
    tags = ["classical", "melancholic", "sad", "emotional", "instrumental"]
    r = estimate_features(tags)
    check("low valence expected",  r.valence < 0.35, got=r.valence)
    check("low-ish energy expected", r.energy < 0.45, got=r.energy)

    # ── Test 8: High-energy dance track ────────────────────
    print("\n[8] Real-world scenario — Dance banger")
    tags = ["dance", "edm", "party", "high energy", "banger", "upbeat"]
    r = estimate_features(tags)
    check("very high energy", r.energy > 0.8, got=r.energy)
    check("very high valence", r.valence > 0.7, got=r.valence)
    check("full confidence", r.confidence == 1.0)

    # ── Test 9: Values always in [0, 1] ───────────────────
    print("\n[9] Output bounds")
    # Throw everything positive at it
    extreme = ["energetic", "high energy", "party", "banger", "upbeat",
               "happy", "joyful", "uplifting"]
    r = estimate_features(extreme)
    check("energy stays <= 1.0",  r.energy  <= 1.0, got=r.energy)
    check("valence stays <= 1.0", r.valence <= 1.0, got=r.valence)

    # Everything negative
    extreme = ["depressing", "sad", "melancholic", "dark", "lonely", "gloomy"]
    r = estimate_features(extreme)
    check("energy stays >= 0.0",  r.energy  >= 0.0, got=r.energy)
    check("valence stays >= 0.0", r.valence >= 0.0, got=r.valence)

    # ── Test 10: Lexicon sanity checks ────────────────────
    print("\n[10] Lexicon coverage")
    required = ["energetic", "chill", "sad", "happy", "dance",
                "melancholic", "uplifting", "rock", "electronic"]
    missing = [t for t in required if t not in TAG_LEXICON]
    check(f"core tags all present (missing: {missing})", not missing)
    check(f"lexicon has at least 80 entries ({len(TAG_LEXICON)})",
          len(TAG_LEXICON) >= 80)

    # ── Summary ───────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Results: {passed} passed  {failed} failed")
    print(f"{'='*55}\n")
    return failed == 0


if __name__ == "__main__":
    run_tests()
