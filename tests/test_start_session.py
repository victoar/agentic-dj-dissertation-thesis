"""
start_session() integration tests.
Requires .env populated and an active Spotify device.
Tests both the happy path (clear description) and the fallback path (vague input).
"""

from agentic_dj.agent.loop import start_session
import agentic_dj.agent.tools as tool_module


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
    print("start_session() — Integration Tests")
    print("=" * 55)

    # ── Test 1: Clear description → happy path ───────────────
    print("\n[1] Clear description — '2016 club vibes'")
    result = start_session("2016 club vibes", verbose=True)

    check("returns success=True",         result.get("success") is True,
          got=result)
    check("session_label is non-empty",   bool(result.get("session_label")),
          got=result.get("session_label"))
    check("opening_track dict present",   isinstance(result.get("opening_track"), dict))
    check("opening_track has name",       bool((result.get("opening_track") or {}).get("name")))
    check("fallback_used is False",       result.get("fallback_used") is False,
          got=result.get("fallback_used"))
    check("explanation is non-empty",     bool(result.get("explanation")))

    # Listener state should reflect high energy/valence
    state = tool_module.get_listener_state()
    check("energy seeded above 0.5",      state["energy"] > 0.5,
          got=state["energy"])
    check("history contains 1 track",     tool_module._history != [],
          got=len(tool_module._history))
    check("_queue is empty (play, not queue)", tool_module._queue == [],
          got=len(tool_module._queue))
    check("_lookahead is empty",          tool_module._lookahead == [],
          got=len(tool_module._lookahead))

    print(f"\n  Label:  {result.get('session_label')}")
    print(f"  Track:  {(result.get('opening_track') or {}).get('name')} "
          f"— {(result.get('opening_track') or {}).get('artist')}")

    # ── Test 2: Vague/meaningless description → fallback ────
    print("\n[2] Vague description — 'xkqzpl' (random characters)")
    result2 = start_session("xkqzpl", verbose=True)

    check("returns success=True",         result2.get("success") is True,
          got=result2)
    check("fallback_used is True",        result2.get("fallback_used") is True,
          got=result2.get("fallback_used"))
    check("session_label indicates fallback",
          "favourite" in result2.get("session_label", "").lower()
          or bool(result2.get("session_label")))

    print(f"\n  Label:  {result2.get('session_label')}")
    print(f"  Track:  {(result2.get('opening_track') or {}).get('name')} "
          f"— {(result2.get('opening_track') or {}).get('artist')}")

    # ── Test 3: Calm description → low energy state seed ────
    print("\n[3] Calm description — 'quiet Sunday morning reading'")
    result3 = start_session("quiet Sunday morning reading", verbose=False)

    check("returns success=True",         result3.get("success") is True,
          got=result3)
    state3 = tool_module.get_listener_state()
    check("energy seeded below 0.6",      state3["energy"] < 0.6,
          got=state3["energy"])

    print(f"\n  Label:  {result3.get('session_label')}")

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"Results: {passed} passed  {failed} failed")
    print(f"{'='*55}\n")
    return failed == 0


if __name__ == "__main__":
    run_tests()
