"""
Agent loop tests.
These run a full ReAct cycle against live APIs.
Make sure Spotify is open on a device before running.
"""

from agentic_dj.agent.tools import reset_session
from agentic_dj.agent.loop import run_agent_cycle


def test_basic_cycle():
    """A full cycle from cold start — no feedback, general context."""
    print("\n" + "=" * 55)
    print("Agent Loop — Basic Cycle Test")
    print("=" * 55)

    reset_session("general")
    result = run_agent_cycle(verbose=True)

    assert isinstance(result, dict)
    assert "explanation"  in result
    assert "trace"        in result
    assert "steps"        in result
    assert result["steps"] > 0, "Agent made zero tool calls"
    assert len(result["explanation"]) > 20, "Explanation is too short"

    print(f"\n✓ Cycle completed in {result['steps']} steps")
    print(f"✓ Success: {result['success']}")
    print(f"✓ Explanation: {result['explanation'][:200]}")


def test_cycle_with_feedback():
    """Cycle triggered by a skip — agent should lower energy."""
    print("\n" + "=" * 55)
    print("Agent Loop — Feedback Cycle Test (skip)")
    print("=" * 55)

    reset_session("workout")
    result = run_agent_cycle(
        feedback_event="skip",
        feedback_track="Some Loud Track",
        feedback_artist="Some Artist",
        verbose=True,
    )

    assert result["steps"] > 0
    assert len(result["trace"]) > 0

    # Check that update_listener_state was called with the skip
    tool_calls = [t["tool_name"] for t in result["trace"] if t.get("tool_name")]
    assert "update_listener_state" in tool_calls, \
        f"Expected update_listener_state in tool calls, got: {tool_calls}"

    print(f"\n✓ Feedback applied and cycle completed")
    print(f"✓ Tool calls: {tool_calls}")


def test_trace_structure():
    """Verify the trace contains the expected reasoning steps."""
    print("\n" + "=" * 55)
    print("Agent Loop — Trace Structure Test")
    print("=" * 55)

    reset_session("study")
    result = run_agent_cycle(verbose=False)

    trace      = result["trace"]
    tool_names = [t["tool_name"] for t in trace if t.get("tool_name")]

    print(f"\n  Tool call sequence: {tool_names}")

    # Context tools are now called directly in Python and recorded in trace
    assert "get_listener_state"   in tool_names, \
        f"Expected get_listener_state in trace, got: {tool_names}"
    assert "get_session_arc"      in tool_names, \
        f"Expected get_session_arc in trace, got: {tool_names}"
    assert "get_current_playback" in tool_names, \
        f"Expected get_current_playback in trace, got: {tool_names}"
    assert "add_track_to_queue"   in tool_names, \
        f"Expected add_track_to_queue in trace, got: {tool_names}"

    # Every act entry should have a tool_result
    act_entries = [t for t in trace if t["kind"] == "act" and t.get("tool_name")]
    for entry in act_entries:
        assert entry["tool_result"] is not None, \
            f"Tool {entry['tool_name']} has no result in trace"

    print(f"✓ Trace has {len(trace)} entries")
    print(f"✓ Tool sequence: {tool_names}")
    print(f"✓ All act entries have results")


if __name__ == "__main__":
    test_basic_cycle()
    test_cycle_with_feedback()
    test_trace_structure()