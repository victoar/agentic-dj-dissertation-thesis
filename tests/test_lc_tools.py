"""
Schema-parity test for the LangGraph migration (Phase 1).

Asserts that the StructuredTool schemas generated in ``agentic_dj.agent.lc_tools``
match — name, argument names, and required args — the hand-written
``TOOL_DECLARATIONS`` the LLM is tuned against today. This guards against the
``@tool`` signature inference silently drifting from what the model currently
sees (see plan_langgraph_migration.md §6.1, §7).

EXPECTED below is a frozen snapshot of loop.TOOL_DECLARATIONS. It is copied here
deliberately rather than imported, because importing loop.py instantiates a Groq
client at module load (needs GROQ_API_KEY). When TOOL_DECLARATIONS is deleted at
cutover (Phase 4), this snapshot becomes the standalone golden reference.
"""

import pytest

from agentic_dj.agent import lc_tools


# name -> (set of arg names, set of required arg names)
EXPECTED: dict[str, tuple[set[str], set[str]]] = {
    "get_listener_state":         (set(), set()),
    "update_listener_state":      ({"event", "track_name", "artist"},
                                   {"event", "track_name", "artist"}),
    "get_session_arc":            (set(), set()),
    "get_compatible_keys":        ({"camelot_position"}, {"camelot_position"}),
    "check_transition":           ({"from_camelot", "to_camelot"},
                                   {"from_camelot", "to_camelot"}),
    "estimate_bpm_compatibility": ({"current_bpm", "candidate_bpm", "arc_phase"},
                                   {"current_bpm", "candidate_bpm"}),
    "search_tracks_by_tag":       ({"tag", "limit"}, {"tag"}),
    "search_artist_tracks":       ({"artist", "limit"}, {"artist"}),
    "search_tracks":              ({"query", "limit"}, {"query"}),
    "search_tracks_by_playlist":  ({"query", "sample_size"}, {"query"}),
    "get_track_details":          ({"track_name", "artist"}, {"track_name", "artist"}),
    "get_current_playback":       (set(), set()),
    "get_queue_state":            (set(), set()),
    "get_session_history":        (set(), set()),
    "add_track_to_queue":         ({"track_name", "artist"}, {"track_name", "artist"}),
}


def _schema(tool):
    s = tool.get_input_schema().model_json_schema()
    props = set(s.get("properties", {}).keys())
    required = set(s.get("required", []))
    return props, required


def test_all_tools_present_and_no_extras():
    names = {t.name for t in lc_tools.ALL_TOOLS}
    assert names == set(EXPECTED), (
        f"Tool set drift. Missing: {set(EXPECTED) - names}; Extra: {names - set(EXPECTED)}"
    )
    assert len(lc_tools.ALL_TOOLS) == len(EXPECTED) == 15


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_tool_schema_matches_declaration(name):
    tool = lc_tools.TOOLS_BY_NAME[name]
    exp_args, exp_required = EXPECTED[name]
    props, required = _schema(tool)
    assert props == exp_args, f"{name}: arg names {props} != expected {exp_args}"
    assert required == exp_required, f"{name}: required {required} != expected {exp_required}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_every_tool_has_a_description(name):
    # The docstring becomes the description the LLM sees — it must be non-empty.
    assert lc_tools.TOOLS_BY_NAME[name].description.strip()


def test_cycle_tools_excludes_search():
    cycle_names = {t.name for t in lc_tools.CYCLE_TOOLS}
    search = {"search_tracks_by_tag", "search_artist_tracks",
              "search_tracks", "search_tracks_by_playlist"}
    assert search.isdisjoint(cycle_names), "cycle tools must not contain search tools"
    assert cycle_names == set(EXPECTED) - search
    assert len(lc_tools.CYCLE_TOOLS) == 11


def test_search_tracks_does_not_expose_enrich():
    # The LLM never controlled `enrich`; it must stay hidden (fixed True).
    props, _ = _schema(lc_tools.TOOLS_BY_NAME["search_tracks"])
    assert "enrich" not in props
