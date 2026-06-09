# Plan — Restore Explanations + Cut Cycle Latency (without breaking ReAct)

*Draft for review. No implementation yet. Same branch (`feature/ADJ-16-add-lang-graph`).*

---

## Goal & guardrails

Fix two problems: (1) the agent stopped reliably explaining its choice, and (2) a cycle is slow (3–4 gpt-oss turns + heavy enrichment). Do this **without weakening the ReAct claim** the dissertation rests on.

**ReAct guardrails (non-negotiable for these changes):**
- The LLM still **chooses the track** by reasoning over the candidates — it is not forced to "take #1".
- The music-theory tools (`check_transition`, `get_compatible_keys`, `estimate_bpm_compatibility`) stay **available** in the cycle toolset — we discourage redundant calls but never remove them, so the agent *can* apply music theory itself.
- Reasoning stays rich enough to analyse for the failure taxonomy → reasoning effort tuned to **medium**, not minimal.
- The loop remains Thought → Action → Observation, LLM-driven.

---

## Change A — Explanation at commit (remove the separate explain turn)

**Problem:** the explanation comes from a fragile extra tool-less LLM turn (`explain` node) that gpt-oss often returns empty, leaving the generic "Up next: X by Y".

**Fix:**
- Add an optional `reason: str` parameter to `add_track_to_queue` (tools.py + the lc_tools wrapper). The model passes its one-sentence justification *when it commits*. `tools.add_track_to_queue` records it and returns it in the result dict.
- The graph's terminal result then carries the reason; `run_graph` surfaces it as the explanation.
- **Remove the `explain` node** from the shared graph assembly (both cycle and opener). Control flow becomes:
  ```
  START → agent
  agent → tool_calls? tools : END        (END → explanation = final content, if any)
  tools → stop-tool fired? END : agent   (END → explanation = terminal reason)
  ```
- `run_agent_cycle` normalisation keeps its existing fallbacks (last `think` entry, then a synthesised line) for the rare case the reason is missing.

**Effect:** explanation is now deterministic and grounded in the committing turn, AND one LLM round-trip is eliminated. More ReAct-faithful, not less (the explanation is the Thought attached to the Action).

## Change B — Lower reasoning effort on the cycle (medium)

`_make_llm` (graph.py) gains a `reasoning_effort` parameter, threaded through `build_dj_graph`. The mid-session cycle uses a named constant `CYCLE_REASONING_EFFORT = "medium"`. The opener/interpret keep their current behaviour. This is a decoding knob only — the ReAct loop is unchanged — and is the single biggest latency lever because gpt-oss "thinks" in reasoning tokens.

## Change C — Keep selection genuinely agentic (prompt)

Rewrite `CYCLE_SYSTEM_PROMPT` so the agent:
- calls `get_ranked_candidates` once,
- **reasons about which of the top 5 best fits** the listener state, arc phase, and openness (not "always #1") — usually #1, but justify the choice,
- *may* confirm a borderline pick with `check_transition` / `get_compatible_keys` (optional, since candidates are pre-screened),
- commits with `add_track_to_queue(track_name, artist, reason)` where `reason` references one listener-state/arc signal and one musical property.

This preserves the LLM's selection reasoning and its access to the music-theory tools — the substance of the ReAct claim — while removing the *forced* redundant `check_transition` call.

## Change D — Parallelise enrichment in get_ranked_candidates

The non-LLM cost is enriching ~20 candidates (each = Spotify resolve + Last.fm + Soundcharts), done serially. Refactor `_gather_bucket` to:
1. gather candidate **identities** from the sources (the Last.fm calls),
2. resolve + enrich them **in parallel** via a `ThreadPoolExecutor` (I/O-bound; ~20 serial calls → a few parallel batches),
3. de-dupe, cap at `_BUCKET_SIZE`, then filter + rank as today.

Thread-safety: enrichment of distinct tracks is independent; the disk caches and the `_resolve_cache`/`_candidate_cache` dicts may do a rare redundant fetch under a race but never corrupt (worst case = one extra lookup). Session state is not mutated during enrichment.

*Optional complement (decide later):* a two-phase narrow — gather ~20 identities, cheap pre-rank by source priority/listener count, enrich only the top ~8 — to cut external calls further. Keep as a follow-up if parallelism alone isn't enough.

---

## Round-trip math

| | Before | After |
|---|---|---|
| decide → `get_ranked_candidates` | 1 | 1 |
| (optional) `check_transition` | often 1 | 0–1 (agent's choice) |
| `add_track_to_queue` | 1 | 1 (now carries `reason`) |
| separate `explain` turn | 1 | **0 (removed)** |
| **typical total** | **3–4** | **2** |

Plus: each remaining turn is faster (reasoning_effort medium), and the tool's enrichment is parallel.

## Files touched

| File | Change |
|---|---|
| `agent/tools.py` | `add_track_to_queue(reason="")` records/returns reason; `_gather_bucket` parallel enrichment |
| `agent/lc_tools.py` | `add_track_to_queue` wrapper gains `reason`; update its docstring |
| `agent/graph.py` | `_make_llm(reasoning_effort=…)`; drop the `explain` node from `_assemble`; tools node stores the terminal `reason`; `run_graph` returns explanation from the reason |
| `agent/loop.py` | `CYCLE_REASONING_EFFORT` constant; rewrite `CYCLE_SYSTEM_PROMPT`; `run_agent_cycle` reads explanation from `graph_out` (fallbacks unchanged) |
| `tests/` | update graph tests (no explain node; reason→explanation), `test_lc_tools` EXPECTED for `add_track_to_queue` (+optional `reason`), `test_loop_unit` explanation source |

## Tests (offline)

- `add_track_to_queue` returns the `reason`; schema exposes `reason` as optional (not required).
- Graph: a scripted commit with a `reason` ends the graph and `run_graph` returns that reason as the explanation; no `explain` entry is emitted.
- Existing retry/sanitize/opener tests still pass.
- `_gather_bucket` parallel path returns the same de-duped, capped bucket as the serial version (determinism of the *set*, ordering may differ — assert on the set).
- Existing 63 tests retargeted where they referenced the explain node.

## Risks

- **Reason quality**: the model might write a thin reason. Mitigated by the prompt requirement (one signal + one musical property) and the `think`/synth fallbacks.
- **Thinner traces** if effort were set too low — hence *medium*, not *low*.
- **Thread-safety** of the caches under parallel enrichment — benign races only (documented above); can pin a lock around cache writes if we ever see duplicates.
- **ReAct integrity**: the only behavioural risk is the agent degenerating into "always #1". The prompt keeps real selection reasoning, and `check_transition` stays available — we should spot-check traces after, to confirm the agent still reasons over the set.

## Open questions for you

1. **Reasoning effort** — `medium` (my lean, keeps traces rich) or `low` (faster, thinner reasoning)? Worth a quick A/B on latency vs trace quality.
2. **Two-phase enrichment** — include now, or ship parallelism first and add only if still slow? *Lean: parallel first.*
3. **`reason` required vs optional** on `add_track_to_queue` — optional (with fallbacks) is safer for tool-call reliability; required guarantees an explanation but risks a malformed call. *Lean: optional + fallbacks.*
