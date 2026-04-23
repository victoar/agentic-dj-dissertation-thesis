from agentic_dj.music.tags import estimate_features_with_fallback, TAG_LEXICON


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
    print("Tag Fallback — Unknown Tag Handling Tests")
    print("=" * 55)

    # ── Test 1: Known tags work identically ────────────────
    print("\n[1] Known tags behave the same as before")
    r = estimate_features_with_fallback(["energetic", "dance"])
    check("known tags produce high energy", r.energy > 0.7, got=r.energy)
    check("no inferred tags when all match",
          all("~" not in t for t in r.matched_tags))

    # ── Test 2: Unknown tag near a known one ───────────────
    print("\n[2] Semantic fallback for near-synonyms")
    # "pumped" is not in the lexicon, but is semantically close to "energetic"
    r = estimate_features_with_fallback(["pumped"])
    check("'pumped' → high-ish energy via fallback",
          r.energy > 0.55, got=r.energy)
    check("fallback tag marked with ~",
          any("~" in t for t in r.matched_tags),
          got=r.matched_tags)

    # ── Test 3: Obscure genre → should find a neighbour ────
    print("\n[3] Obscure genre tags")
    # vaporwave is not in the lexicon
    r = estimate_features_with_fallback(["vaporwave"])
    check("'vaporwave' produces some estimate (not neutral)",
          r.energy != 0.5 or r.valence != 0.5,
          got=f"energy={r.energy}, valence={r.valence}")

    # ── Test 4: Pure nonsense → neutral ────────────────────
    print("\n[4] Total nonsense tags stay neutral")
    r = estimate_features_with_fallback(["asdfqwerty123", "xyzzy456"])
    check("nonsense tags → neutral energy",  r.energy == 0.5,  got=r.energy)
    check("nonsense tags → neutral valence", r.valence == 0.5, got=r.valence)
    check("low confidence", r.confidence < 0.3, got=r.confidence)

    # ── Test 5: Popularity fallback ────────────────────────
    print("\n[5] Popularity as last-resort valence signal")
    r_hi = estimate_features_with_fallback([], popularity=85)
    r_lo = estimate_features_with_fallback([], popularity=15)
    check("high popularity → slightly positive valence",
          r_hi.valence > 0.55, got=r_hi.valence)
    check("low popularity → slightly negative valence",
          r_lo.valence < 0.45, got=r_lo.valence)

    # ── Test 6: Mix of known and unknown ───────────────────
    print("\n[6] Mixed known + unknown tags")
    r = estimate_features_with_fallback(
        ["melancholic", "vaporwave", "nostalgic"],
    )
    check("known tags still dominate",  r.valence < 0.4, got=r.valence)
    check("at least one direct match",
          any("~" not in t for t in r.matched_tags),
          got=r.matched_tags)

    # ── Test 7: Fallback disabled → behaves like v1 ────────
    print("\n[7] Fallback disabled")
    r = estimate_features_with_fallback(
        ["vaporwave"], use_semantic=False
    )
    check("with fallback off, unknown → neutral", r.energy == 0.5)

    # ── Summary ────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Results: {passed} passed  {failed} failed")
    print(f"{'='*55}\n")
    return failed == 0


if __name__ == "__main__":
    run_tests()
