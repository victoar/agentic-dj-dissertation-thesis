"""
Graph-engine tests for the LangGraph migration (Phase 2).

Drives ``agentic_dj.agent.graph`` with a fake LLM and fake tools so the ReAct
control flow, stop-tool termination, reasoning capture, and trace shape are
verified deterministically — no network, no GROQ_API_KEY.
"""

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

import agentic_dj.agent.graph as graph_mod


# ── Fake ChatGroq: returns a scripted sequence of AIMessages ────────────────────

class FakeChatGroq:
    script: list = []
    idx: int = 0

    def __init__(self, **kwargs):
        pass

    def bind_tools(self, tools):
        return self                       # same object → shares the script

    def invoke(self, messages):
        msg = FakeChatGroq.script[FakeChatGroq.idx]
        FakeChatGroq.idx += 1
        return msg


@pytest.fixture(autouse=True)
def _patch_chatgroq(monkeypatch):
    monkeypatch.setattr(graph_mod, "ChatGroq", FakeChatGroq)
    FakeChatGroq.script = []
    FakeChatGroq.idx = 0
    yield


# ── Fake tools ──────────────────────────────────────────────────────────────────

@tool
def check_transition(from_camelot: str, to_camelot: str) -> dict:
    """Check harmonic smoothness."""
    return {"score": 0.8, "verdict": "smooth"}


@tool
def add_track_to_queue(track_name: str, artist: str) -> dict:
    """Terminal action: queue a track."""
    return {"success": True, "queued": {"name": track_name, "artist": artist}}


TOOLS = [check_transition, add_track_to_queue]


def _ai_tool_call(name, args, call_id, reasoning=""):
    return AIMessage(
        content="",
        additional_kwargs={"reasoning_content": reasoning} if reasoning else {},
        tool_calls=[{"name": name, "args": args, "id": call_id, "type": "tool_call"}],
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_react_loop_reaches_stop_tool_and_explains():
    FakeChatGroq.script = [
        _ai_tool_call("check_transition", {"from_camelot": "8B", "to_camelot": "9B"},
                      "c1", reasoning="Both minor, one step — smooth."),
        _ai_tool_call("add_track_to_queue", {"track_name": "Time", "artist": "Pink Floyd"},
                      "c2", reasoning="Commit the top candidate."),
        AIMessage(content="Queued 'Time' by Pink Floyd — smooth 8B→9B step keeps the build going."),
    ]
    graph = graph_mod.build_dj_graph(TOOLS, stop_tool="add_track_to_queue")
    out = graph_mod.run_graph(graph, [HumanMessage("Pick the next track.")])

    # Terminal result captured from the stop tool
    assert out["queued"] == {"success": True, "queued": {"name": "Time", "artist": "Pink Floyd"}}
    assert "Pink Floyd" in out["explanation"]

    kinds = [e["kind"] for e in out["trace"]]
    assert kinds.count("think") == 2          # reasoning_content captured on both action turns
    assert kinds.count("act") == 2            # check_transition + add_track_to_queue
    assert kinds.count("explain") == 1        # final tool-less turn

    # Trace entry shape matches what bridge.adapt_trace() consumes
    for e in out["trace"]:
        assert set(e) == {"step", "kind", "content", "tool_name", "tool_args", "tool_result"}

    # The act entries name the tools and carry their results
    acts = [e for e in out["trace"] if e["kind"] == "act"]
    assert {a["tool_name"] for a in acts} == {"check_transition", "add_track_to_queue"}
    assert acts[0]["tool_result"]["verdict"] == "smooth"


def test_no_tool_calls_ends_with_final_text():
    # A final turn carries reasoning (reasoning_content) AND the answer (content).
    # Parity with the original loop: both a `think` and an `explain` entry are emitted.
    FakeChatGroq.script = [AIMessage(
        content="I have nothing to queue right now.",
        additional_kwargs={"reasoning_content": "No suitable candidate; hold the queue."},
    )]
    graph = graph_mod.build_dj_graph(TOOLS, stop_tool="add_track_to_queue")
    out = graph_mod.run_graph(graph, [HumanMessage("Pick the next track.")])

    assert out["queued"] is None
    assert out["explanation"] == "I have nothing to queue right now."
    assert [e["kind"] for e in out["trace"]] == ["think", "explain"]
    assert out["trace"][0]["content"] == "No suitable candidate; hold the queue."
    assert out["trace"][1]["content"] == "I have nothing to queue right now."


# ── Opener graph (natural-language session start) ───────────────────────────────

@tool
def _opener_search(query: str, sample_size: int = 20) -> dict:
    """Fake playlist search."""
    return {"candidates": [{"name": "Closer", "artist": "The Chainsmokers"}]}


@tool
def select_opening_track(track_name: str, artist: str) -> dict:
    """Synthetic terminal — body never runs (node validates against state)."""
    return {"success": False, "error": "handled by node"}


def _opener_tools():
    # _opener_search must be named search_tracks_by_playlist for the registry.
    _opener_search.name = "search_tracks_by_playlist"
    return [_opener_search, select_opening_track]


def test_opener_commits_a_seen_track():
    FakeChatGroq.script = [
        _ai_tool_call("search_tracks_by_playlist", {"query": "2016 pop hits"}, "s1",
                      reasoning="Search 2016 pop playlists."),
        _ai_tool_call("select_opening_track",
                      {"track_name": "Closer", "artist": "The Chainsmokers"}, "s2"),
        AIMessage(content="Opening with 'Closer' — peak-2016 pop to set the vibe."),
    ]
    graph = graph_mod.build_opener_graph(_opener_tools(), stop_tool="select_opening_track")
    out = graph_mod.run_graph(graph, [HumanMessage("2016 clubbing vibes")], max_iterations=8)

    assert out["choice"] == {"name": "Closer", "artist": "The Chainsmokers"}
    assert out["queued"]["success"] is True
    assert "Closer" in out["explanation"]


def test_opener_rejects_unseen_track():
    # The model tries to commit a track it never searched for → guard rejects it,
    # the loop does not terminate, and (here) the model then gives up.
    FakeChatGroq.script = [
        _ai_tool_call("select_opening_track",
                      {"track_name": "Imaginary Hit", "artist": "Nobody"}, "s1"),
        AIMessage(content="No verified candidate found."),
    ]
    graph = graph_mod.build_opener_graph(_opener_tools(), stop_tool="select_opening_track")
    out = graph_mod.run_graph(graph, [HumanMessage("something obscure")], max_iterations=8)

    assert out["queued"] is None
    assert out["choice"] is None
    act = next(e for e in out["trace"] if e["kind"] == "act" and e["tool_name"] == "select_opening_track")
    assert act["tool_result"]["success"] is False
    assert "not in your search results" in act["tool_result"]["error"]


def test_unsuccessful_stop_tool_does_not_terminate():
    # add_track_to_queue returns success=False once, then the model gives final text.
    @tool
    def failing_queue(track_name: str, artist: str) -> dict:
        """Queue that fails."""
        return {"success": False, "error": "duplicate"}

    FakeChatGroq.script = [
        _ai_tool_call("failing_queue", {"track_name": "X", "artist": "Y"}, "c1"),
        AIMessage(content="Could not queue; stopping."),
    ]
    graph = graph_mod.build_dj_graph([failing_queue], stop_tool="failing_queue")
    out = graph_mod.run_graph(graph, [HumanMessage("go")])

    assert out["queued"] is None              # success=False must not set terminal
    assert out["explanation"] == "Could not queue; stopping."
