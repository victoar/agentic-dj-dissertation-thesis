"""
Unit tests for the Groq ReAct agent loop.

No external API calls — the Groq client is mocked via unittest.mock.patch.
Run with: pytest tests/test_loop_unit.py -v
"""

import json
import pytest
from unittest.mock import patch, MagicMock

import agentic_dj.agent.loop as loop_module
import agentic_dj.agent.tools as tool_module
from agentic_dj.agent.loop import (
    GROQ_TOOLS,
    TOOL_DECLARATIONS,
    TOOL_REGISTRY,
    _best_fallback,
    _run_react_loop,
    run_agent_cycle,
)


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_mock_response(tool_name=None, tool_args=None, content="", tool_id="call_001"):
    """Build a MagicMock that mimics a Groq chat completion response."""
    mock_resp = MagicMock()
    msg = mock_resp.choices[0].message
    msg.content = content
    if tool_name:
        tc = MagicMock()
        tc.id = tool_id
        tc.function.name = tool_name
        tc.function.arguments = json.dumps(tool_args or {})
        msg.tool_calls = [tc]
    else:
        msg.tool_calls = None
    return mock_resp


# ── Group 1: Static structure ─────────────────────────────────────────────────

def test_groq_tools_count():
    """GROQ_TOOLS and TOOL_DECLARATIONS must have the same length (12)."""
    assert len(GROQ_TOOLS) == 12
    assert len(GROQ_TOOLS) == len(TOOL_DECLARATIONS)


def test_groq_tools_envelope():
    """Every entry in GROQ_TOOLS must have the correct Groq/OpenAI envelope."""
    for tool in GROQ_TOOLS:
        assert tool["type"] == "function", f"Missing type=function: {tool}"
        fn = tool["function"]
        assert "name"        in fn, f"Missing name in function: {fn}"
        assert "description" in fn, f"Missing description in function: {fn}"
        assert "parameters"  in fn, f"Missing parameters in function: {fn}"


def test_tool_registry_completeness():
    """Every tool declared in TOOL_DECLARATIONS must be callable via TOOL_REGISTRY."""
    for decl in TOOL_DECLARATIONS:
        name = decl["name"]
        assert name in TOOL_REGISTRY,       f"Missing from TOOL_REGISTRY: {name}"
        assert callable(TOOL_REGISTRY[name]), f"Not callable in TOOL_REGISTRY: {name}"


def test_tool_registry_no_extras():
    """TOOL_REGISTRY must not contain names absent from TOOL_DECLARATIONS."""
    declared = {d["name"] for d in TOOL_DECLARATIONS}
    for name in TOOL_REGISTRY:
        assert name in declared, f"Extra entry in TOOL_REGISTRY not in TOOL_DECLARATIONS: {name}"


# ── Group 2: _best_fallback() ─────────────────────────────────────────────────

def test_best_fallback_picks_closest():
    """Picks the candidate with the smallest energy+valence distance to target."""
    state = {"energy": 0.8, "valence": 0.7}
    candidates = [
        {"name": "Far Track",  "artist": "A", "energy_est": 0.2, "valence_est": 0.1},
        {"name": "Near Track", "artist": "B", "energy_est": 0.75, "valence_est": 0.65},
        {"name": "Mid Track",  "artist": "C", "energy_est": 0.5, "valence_est": 0.5},
    ]
    result = _best_fallback(candidates, state)
    assert result["name"] == "Near Track"


def test_best_fallback_empty():
    """Empty candidate list returns an empty dict."""
    assert _best_fallback([], {"energy": 0.5, "valence": 0.5}) == {}


def test_best_fallback_penalizes_bpm_camelot():
    """bpm_ok=False and camelot_ok=False each add +0.2 to the score (lower = better).

    A candidate with perfect energy/valence match but both flags False (score=0.4)
    should lose to one that is 0.1 off in each dimension but has no flags (score=0.2).
    """
    state = {"energy": 0.5, "valence": 0.5}
    candidates = [
        {"name": "Flag Track",    "artist": "A", "energy_est": 0.5, "valence_est": 0.5,
         "bpm_ok": False, "camelot_ok": False},
        {"name": "Healthy Track", "artist": "B", "energy_est": 0.6, "valence_est": 0.6},
    ]
    result = _best_fallback(candidates, state)
    assert result["name"] == "Healthy Track"


# ── Group 3: run_agent_cycle() return shape ───────────────────────────────────

def test_run_agent_cycle_return_keys():
    """run_agent_cycle() must always return the 4 required keys."""
    tool_module.reset_session("general")

    queue_response = _make_mock_response(
        tool_name="add_track_to_queue",
        tool_args={"track_name": "Test Track", "artist": "Test Artist"},
        tool_id="c1",
    )
    explain_response = _make_mock_response(content="Great pick!")

    with patch.object(loop_module, "client") as mock_client:
        mock_client.chat.completions.create.side_effect = [queue_response, explain_response]
        with patch.dict(loop_module.TOOL_REGISTRY, {
            "add_track_to_queue": lambda track_name, artist: {
                "success": True,
                "queued": {"name": track_name, "artist": artist},
            },
        }):
            result = run_agent_cycle(verbose=False)

    assert "explanation"  in result
    assert "queued_track" in result
    assert "trace"        in result
    assert "success"      in result


def test_feedback_applied_before_loop():
    """update_listener_state must appear at step 0 in the trace when feedback is given."""
    tool_module.reset_session("general")

    queue_response = _make_mock_response(
        tool_name="add_track_to_queue",
        tool_args={"track_name": "Next Track", "artist": "Artist"},
        tool_id="c1",
    )
    explain_response = _make_mock_response(content="Done.")

    with patch.object(loop_module, "client") as mock_client:
        mock_client.chat.completions.create.side_effect = [queue_response, explain_response]
        with patch.dict(loop_module.TOOL_REGISTRY, {
            "add_track_to_queue": lambda track_name, artist: {
                "success": True,
                "queued": {"name": track_name, "artist": artist},
            },
        }):
            result = run_agent_cycle(
                feedback_event="skip",
                feedback_track="Old Track",
                feedback_artist="Old Artist",
                verbose=False,
            )

    tool_names = [t["tool_name"] for t in result["trace"] if t.get("tool_name")]
    assert "update_listener_state" in tool_names, \
        f"update_listener_state missing from trace: {tool_names}"

    step0 = next((t for t in result["trace"] if t.get("step") == 0), None)
    assert step0 is not None, "No step-0 trace entry found"
    assert step0["tool_name"] == "update_listener_state"


# ── Group 4: _run_react_loop() edge cases ─────────────────────────────────────

def test_unknown_tool_graceful():
    """Groq calling an unknown tool name must not raise — loop returns normally."""
    tool_module.reset_session("general")

    unknown_response = _make_mock_response(
        tool_name="nonexistent_tool",
        tool_args={},
        tool_id="c1",
    )
    final_response = _make_mock_response(content="Recovered gracefully.")

    with patch.object(loop_module, "client") as mock_client:
        mock_client.chat.completions.create.side_effect = [unknown_response, final_response]
        result = _run_react_loop(
            messages=[{"role": "user", "content": "select a track"}],
            trace=[],
        )

    assert isinstance(result, dict)
    assert "queued"      in result
    assert "explanation" in result


def test_rate_limit_retries():
    """A 429/rate_limit exception triggers a retry and time.sleep is called once."""
    tool_module.reset_session("general")

    rate_err      = Exception("429 rate_limit exceeded")
    success_resp  = _make_mock_response(content="Succeeded after retry.")

    with patch.object(loop_module, "client") as mock_client:
        with patch.object(loop_module.time, "sleep") as mock_sleep:
            mock_client.chat.completions.create.side_effect = [rate_err, success_resp]
            result = _run_react_loop(
                messages=[{"role": "user", "content": "select a track"}],
                trace=[],
            )

    assert mock_client.chat.completions.create.call_count == 2
    assert mock_sleep.call_count == 1
    assert result["explanation"] == "Succeeded after retry."
