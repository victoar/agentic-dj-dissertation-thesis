"""
Unit tests for the agent loop + candidate selection (post-refactor).

No external API calls. The compiled graph is mocked so run_agent_cycle's
orchestration (feedback, trace, fallback, return shape) is tested in isolation.
Discovery + ranking now live in agent/tools.py (get_ranked_candidates and its
helpers); the ReAct graph engine is covered by tests/test_graph.py and the tool
schemas by tests/test_lc_tools.py.
"""

import pytest
from unittest.mock import patch

import agentic_dj.agent.loop as loop_module
import agentic_dj.agent.tools as tool_module
from agentic_dj.agent.tools import (
    _compatible_pool, _rank_candidates, _seed_tags, _gather_bucket, get_ranked_candidates,
)
from agentic_dj.agent.loop import run_agent_cycle


def _cand(name, key, bpm, energy=0.5, valence=0.5, familiar=False):
    return {"id": name, "name": name, "artist": "A", "camelot_position": key, "bpm": bpm,
            "energy_est": energy, "valence_est": valence, "familiar": familiar, "tags": []}


# ── _compatible_pool() — hard harmonic + tempo filter ───────────────────────────

def test_compatible_pool_drops_incompatible_key():
    state = {"arc_phase": "build"}
    cands = [_cand("Good", "1B", 120), _cand("Bad", "6B", 120)]  # from 12B: 1B=0.85, 6B=0.10
    names = {c["name"] for c in _compatible_pool(cands, state, "12B", 120)}
    assert "Good" in names and "Bad" not in names


def test_compatible_pool_drops_out_of_tempo():
    state = {"arc_phase": "warmup"}                              # tight ±8 BPM
    cands = [_cand("Close", "12B", 140), _cand("Far", "12B", 124)]
    names = {c["name"] for c in _compatible_pool(cands, state, "12B", 142)}
    assert "Close" in names and "Far" not in names


def test_compatible_pool_unknown_current_keeps_all():
    state = {"arc_phase": "build"}
    cands = [_cand("X", "1B", 120), _cand("Y", "6B", 200)]
    assert len(_compatible_pool(cands, state, None, None)) == 2


def test_compatible_pool_never_empty():
    state = {"arc_phase": "peak"}
    cands = [_cand("OnlyBad", "6B", 124)]
    assert _compatible_pool(cands, state, "12B", 142) == cands   # soft degrade


# ── _rank_candidates() — composite RANK_WEIGHTS ─────────────────────────────────

def test_rank_prefers_harmonic():
    state = {"energy": 0.5, "valence": 0.5, "arc_phase": "build", "openness": 0.5}
    cands = [_cand("Far", "6B", 120), _cand("Smooth", "1B", 120)]   # vs 12B: 0.10 vs 0.85
    ranked = _rank_candidates(cands, state, "12B", 120)
    assert ranked[0]["name"] == "Smooth"


def test_rank_uses_arc_target():
    # warmup band 0.3–0.5; with target energy 0.65 both are equidistant on ev,
    # so the arc term decides → the in-band (lower-energy) track wins.
    state = {"energy": 0.65, "valence": 0.5, "arc_phase": "warmup", "openness": 0.5}
    cands = [_cand("Calm", None, None, energy=0.5), _cand("Hype", None, None, energy=0.8)]
    ranked = _rank_candidates(cands, state, None, None)
    assert ranked[0]["name"] == "Calm"


def test_rank_openness_flips_familiarity():
    base = {"energy": 0.5, "valence": 0.5, "arc_phase": "build"}
    cands = [_cand("Known", None, None, familiar=True), _cand("New", None, None, familiar=False)]
    high = _rank_candidates(cands, {**base, "openness": 0.9}, None, None)
    low  = _rank_candidates(cands, {**base, "openness": 0.1}, None, None)
    assert high[0]["name"] == "New"      # high openness rewards novelty
    assert low[0]["name"] == "Known"     # low openness rewards familiarity


# ── _seed_tags() — on-genre, from the current track ─────────────────────────────

def test_seed_tags_from_current_track():
    state = {"energy": 0.5, "valence": 0.5, "arc_phase": "build"}
    current = {"artist": "Bad Bunny", "tags": ["reggaeton", "Bad Bunny", "latin"]}
    assert _seed_tags(state, current) == ["reggaeton", "latin"]   # artist tag skipped


def test_seed_tags_fall_back_to_state():
    state = {"energy": 0.5, "valence": 0.5, "arc_phase": "build"}
    assert _seed_tags(state, None) == tool_module._state_to_tags(state)


# ── get_ranked_candidates() + _gather_bucket() ──────────────────────────────────

def test_get_ranked_candidates_filters_and_formats():
    current = {"id": "cur", "name": "Cur", "artist": "C", "camelot_position": "12B", "bpm": 120}
    bucket = [_cand("Smooth", "1B", 120), _cand("Bad", "6B", 120)]   # Bad is out of key
    with patch.object(tool_module, "_current_track_profile", return_value=current), \
         patch.object(tool_module, "_gather_bucket", return_value=bucket):
        out = get_ranked_candidates(limit=5)
    names = [c["name"] for c in out["candidates"]]
    assert names == ["Smooth"]                       # incompatible dropped
    assert out["candidates"][0]["harmonic_score"] == 0.85
    assert out["from"]["key"] == "12B"


def test_gather_bucket_dedups_and_caps():
    current = {"id": "cur", "name": "Cur", "artist": "C"}
    many = {"candidates": [_cand(f"t{i}", "8B", 120) for i in range(30)]}
    with patch.object(tool_module, "search_similar_tracks", return_value=many), \
         patch.object(tool_module, "get_similar_artists", return_value=[]), \
         patch.object(tool_module, "search_tracks_by_tag", return_value={"candidates": []}):
        bucket = _gather_bucket(current, {"arc_phase": "build"})
    ids = [c["id"] for c in bucket]
    assert len(bucket) <= tool_module._BUCKET_SIZE
    assert len(ids) == len(set(ids))                 # de-duped


def test_search_similar_tracks_shape():
    from agentic_dj.spotify.client import SpotifyTrack
    tool_module.reset_session("general")
    sp = SpotifyTrack(id="s1", name="Sim", artist="SimArtist", album="",
                      duration_ms=1000, uri="spotify:track:s1")
    with patch.object(tool_module, "get_similar_tracks",
                      return_value=[{"name": "Sim", "artist": "SimArtist"}]), \
         patch.object(tool_module, "_resolve", return_value=[sp]), \
         patch.object(tool_module, "_spotify_to_candidate",
                      return_value=_cand("Sim", "8B", 120)):
        res = tool_module.search_similar_tracks("SimArtist", "Sim", limit=5)
    assert res["count"] == 1 and res["candidates"][0]["name"] == "Sim"


# ── start_session opener: interpret robustness + gate ───────────────────────────

class _FakeMsg:
    def __init__(self, content="", reasoning=""):
        self.content = content
        self.additional_kwargs = {"reasoning_content": reasoning} if reasoning else {}


class _FakeLLM:
    def __init__(self, msg):
        self._msg = msg

    def invoke(self, messages):
        return self._msg


def test_interpret_reads_json_from_reasoning_channel():
    msg = _FakeMsg(content="", reasoning='{"energy":0.8,"search_queries":["x"],'
                                         '"session_label":"L","confident":true}')
    with patch.object(loop_module, "_get_interpret_llm", return_value=_FakeLLM(msg)):
        vibe = loop_module._interpret_description("clubbing", trace=[])
    assert vibe and vibe["session_label"] == "L" and vibe["search_queries"] == ["x"]


def test_interpret_extracts_json_from_surrounding_prose():
    msg = _FakeMsg(content='Sure! {"energy":0.3,"search_queries":["calm"],'
                           '"session_label":"Calm","confident":false} hope that helps')
    with patch.object(loop_module, "_get_interpret_llm", return_value=_FakeLLM(msg)):
        vibe = loop_module._interpret_description("calm sunday", trace=[])
    assert vibe and vibe["search_queries"] == ["calm"]


def test_start_session_runs_opener_even_when_not_confident():
    from agentic_dj.spotify.client import SpotifyTrack
    vibe = {"confident": False, "search_queries": ["calm focus"], "session_label": "Calm",
            "energy": 0.3, "valence": 0.6, "focus": 0.7, "openness": 0.7, "social": 0.3}
    track = SpotifyTrack(id="t1", name="Weightless", artist="Marconi Union",
                         album="", duration_ms=1000, uri="spotify:track:t1")
    with patch.object(loop_module, "_interpret_description", return_value=vibe), \
         patch.object(loop_module, "_react_pick_opener",
                      return_value={"name": "Weightless", "artist": "Marconi Union"}) as mock_open, \
         patch.object(loop_module, "prewarm_embedding_model"), \
         patch.object(tool_module._spotify, "search", return_value=[track]), \
         patch.object(tool_module._spotify, "play", return_value=True), \
         patch.object(tool_module, "_record_played_track", return_value={"name": "Weightless"}):
        result = loop_module.start_session("calm focus music")
    assert mock_open.called
    assert result["success"] is True and result["fallback_used"] is False


# ── Session caches ──────────────────────────────────────────────────────────────

def test_resolve_cache_dedupes_searches():
    tool_module.reset_session("general")
    with patch.object(tool_module._spotify, "search", return_value=[]) as mock_search:
        for _ in range(3):
            tool_module._resolve("Amber 311", limit=1)
    assert mock_search.call_count == 1


def test_candidate_cache_dedupes_enrichment():
    from agentic_dj.spotify.client import SpotifyTrack
    tool_module.reset_session("general")
    track = SpotifyTrack(id="abc", name="X", artist="Y", album="",
                         duration_ms=1000, uri="spotify:track:abc")
    with patch.object(tool_module._spotify, "is_saved", return_value=False), \
         patch.object(tool_module, "fetch_enrichment") as mock_enr, \
         patch.object(tool_module, "fetch_track_info",
                      return_value={"bpm": None, "camelot_position": None, "found": False}) as mock_sc:
        mock_enr.return_value.found = False
        for _ in range(3):
            tool_module._spotify_to_candidate(track, enrich=True)
    assert mock_sc.call_count == 1 and mock_enr.call_count == 1


# ── run_agent_cycle() orchestration — graph mocked ──────────────────────────────

def test_run_agent_cycle_return_keys():
    tool_module.reset_session("general")
    graph_result = {"queued": {"success": True, "queued": {"name": "T", "artist": "A"}},
                    "explanation": "Great pick!", "trace": []}
    with patch.object(loop_module, "_get_cycle_graph", return_value=object()), \
         patch.object(loop_module, "run_graph", return_value=graph_result):
        result = run_agent_cycle(verbose=False)
    assert set(result) >= {"explanation", "queued_track", "trace", "success"}
    assert result["success"] is True
    assert result["queued_track"] == {"name": "T", "artist": "A"}


def test_feedback_applied_before_loop():
    tool_module.reset_session("general")
    graph_result = {"queued": {"success": True, "queued": {"name": "N", "artist": "A"}},
                    "explanation": "Done.", "trace": []}
    with patch.object(loop_module, "_get_cycle_graph", return_value=object()), \
         patch.object(loop_module, "run_graph", return_value=graph_result):
        result = run_agent_cycle(feedback_event="skip", feedback_track="Old",
                                 feedback_artist="Old A", verbose=False)
    step0 = next((t for t in result["trace"] if t.get("step") == 0
                  and t.get("tool_name") == "update_listener_state"), None)
    assert step0 is not None


def test_run_agent_cycle_fallback_uses_ranked_candidate():
    tool_module.reset_session("general")
    graph_result = {"queued": None, "explanation": "", "trace": []}
    with patch.object(loop_module, "_get_cycle_graph", return_value=object()), \
         patch.object(loop_module, "run_graph", return_value=graph_result), \
         patch.object(tool_module, "get_ranked_candidates",
                      return_value={"candidates": [{"name": "Fallback Pick", "artist": "FB"}]}), \
         patch.object(tool_module, "add_track_to_queue",
                      return_value={"success": True, "queued": {"name": "Fallback Pick", "artist": "FB"}}):
        result = run_agent_cycle(verbose=False)
    assert result["success"] is True
    assert result["queued_track"]["name"] == "Fallback Pick"
