# Plan — Migrating the AgDJ Agent Loop to LangGraph

*Draft for review. No implementation has started. The goal is to make the code match the dissertation's claim that the agent is "implemented using LangGraph, with each reasoning step structured as a node in a directed graph" (§3.1, §3.4), while preserving 100% of current behaviour.*

---

## 1. Objective and guiding principle

Replace the hand-rolled `_run_react_loop` driver in `agent/loop.py` with a compiled LangGraph `StateGraph`, **without changing**:

- the public function contracts `run_agent_cycle(...)` and `start_session(...)` (same arguments, same return dict shape),
- the trace entry format that `app/bridge.py:adapt_trace()` and the Streamlit trace panel consume (`{kind, content, tool_name, tool_args, tool_result}`),
- the model (`openai/gpt-oss-120b` on Groq) and the rate-limit backoff behaviour,
- the music/Spotify/Last.fm/Soundcharts tools in `tools.py` (untouched),
- the listener-state model in `state.py` (untouched).

**Guiding principle: behaviour-preserving refactor.** The agent should produce the same selections, traces, and explanations it does today; only the *control-flow substrate* changes. If a user-facing behaviour changes, that's a bug, not a goal.

---

## 2. What changes and what doesn't

| Area | Status | Notes |
|---|---|---|
| `agent/loop.py` `_run_react_loop` | **Replaced** | Becomes a compiled graph + node functions |
| `agent/loop.py` `run_agent_cycle` / `start_session` | **Rewritten internally, same signature** | Build initial state → `graph.invoke()` → normalise return |
| `agent/loop.py` Python pre-fetch + `_best_fallback` | **Kept** | Pre-fetch stays a Python pre-step; fallback stays a post-graph safety net |
| `agent/loop.py` `_groq_call` backoff | **Removed** | Replaced by `ChatGroq(max_retries=4)` native retries (decision settled, §6.4) |
| `TOOL_DECLARATIONS` / `GROQ_TOOLS` / `TOOL_REGISTRY` | **Removed** | Collapsed into a single list of LangChain tool objects (decision settled, §6.1) |
| `CYCLE_TOOLS` | **Becomes a filtered list** | Same tool objects minus the four search tools |
| `agent/lc_tools.py` | **New** | `StructuredTool` objects wrapping the `tools.py` functions — the single source of truth for both `bind_tools(...)` and `ToolNode(...)` |
| `agent/tools.py` | **Unchanged (business logic)** | Functions stay plain/callable; wrapping happens in `lc_tools.py`. Only minor docstring edits to fold in guidance currently living in `TOOL_DECLARATIONS`. |
| `agent/state.py`, `music/*`, `spotify/*` | **Unchanged** | |
| `app/bridge.py`, `app/components/*` | **Unchanged** (target) | Only true if return/trace shapes are preserved — this is a hard acceptance test |
| `pyproject.toml` / `requirements.txt` | **Updated** | Add `langchain-groq`; `langgraph`/`langchain` already declared but unused |

---

## 3. Target architecture

### 3.1 Graph state

A `TypedDict` carrying the message history plus the bookkeeping the current loop tracks in locals:

```
class DJState(TypedDict):
    messages:    Annotated[list, add_messages]   # LangChain message objects
    trace:       list[dict]                       # _trace_entry-compatible dicts
    step:        int                              # trace step counter
    terminal:    dict | None                      # result of the stop tool
    explanation: str
```

### 3.2 Nodes

1. **`agent` node** — calls `ChatGroq(...).bind_tools(tools)` on `state["messages"]`, appends the assistant message, and records a `think` trace entry if the model produced reasoning text. (Mirrors steps 1–3 of today's loop.)
2. **`tools` node** — a **custom** tool node (not the stock `ToolNode`) because it must additionally: record one `act` trace entry per call (with `tool_name`/`tool_args`/`tool_result`), detect when the stop tool succeeded and set `state["terminal"]`, and host the `select_opening_track` tool for the bootstrap graph. Executes the LangChain tool objects from `lc_tools.py`. `select_opening_track` is a stateful tool that reads `seen_tracks` from `DJState` via LangGraph's `InjectedState` (replaces today's closure).
3. **`explain` node** — after the stop tool fires, one LLM call with **tools disabled** to get the plain-language explanation (mirrors today's `tools=None` final call), recorded as an `explain` entry.

### 3.3 Edges (the "directed graph")

```
START → agent
agent → (router):
          • no tool_calls            → END        (model gave final text)
          • stop tool already fired  → explain → END
          • otherwise                → tools
tools → agent
```

The router replaces today's `if not msg.tool_calls: break` and the `stop_tool` success check. LangGraph's prebuilt `tools_condition` is *not* sufficient because of our custom terminal-tool rule, so we write a small router function. `recursion_limit` replaces `MAX_ITERATIONS = 15`.

### 3.4 Two graphs, one builder

`run_agent_cycle` and `start_session` differ only in (a) the tool set and (b) the stop tool. So a single factory builds both:

```
build_dj_graph(tools, registry, stop_tool, max_iterations) -> CompiledGraph
```

- **cycle graph** — `tools = CYCLE_TOOLS` (no search; pre-fetch already ran), `stop_tool = "add_track_to_queue"`.
- **opener graph** — the 3-tool subset + synthetic `select_opening_track`, `stop_tool = "select_opening_track"`. The `seen_tracks` anti-hallucination guard moves into the custom tool node's closure (it is graph-scoped state, so it can live in `DJState` or in a per-invocation closure — see §6 decision 5).

Compile once at import time; invoke per cycle.

### 3.5 What stays in plain Python (not in the graph)

To keep the diff small and the behaviour identical, these remain procedural and wrap the graph call, exactly as today:

- the **feedback pre-application** (step 0) in `run_agent_cycle`,
- the **Last.fm pre-fetch + shortlist scoring** that builds the injected candidate list,
- the **scored fallback** (`_best_fallback`) when the graph ends without a queued track,
- in `start_session`: the **structured-JSON vibe extraction** (a single non-loop LLM call — stays a plain `invoke`), the **`reset_session` + state seeding**, the Spotify **`play()`**, and **`_record_played_track`**.

---

## 4. Migration phases (incremental, always-green)

**Phase 0 — Spike / de-risk — ✅ DONE, PASSED.** Confirmed via standalone spike:
- `ChatGroq(model="openai/gpt-oss-120b")` round-trips the model string and does clean native tool-calling; `tool_calls` come back with `args` already as a dict (no `json.loads` needed, unlike the current loop).
- Reasoning text **is** available, on a separate channel: `ChatGroq` → `msg.additional_kwargs["reasoning_content"]`; raw SDK → `message.reasoning`. `response_metadata.token_usage.reasoning_tokens` confirms it numerically.
- Multi-turn (feed `ToolMessage` back) works and the model returns a final text answer with no tool calls.

**Important finding:** on tool-call turns, `.content` is empty/None in both SDKs — the reasoning lives only on the reasoning channel. The current `loop.py` reads `reasoning = (msg.content or "")`, so it is **silently dropping action-turn reasoning today** (its `think` traces only capture the final no-tool turn). The `agent` node must read `additional_kwargs.get("reasoning_content")` (fallback `.content`). This also means the migration will *improve* trace fidelity — see the parity-test note in §5.

**Phase 1 — Tools + graph module beside the old loop (1.5 days).** New file `agent/lc_tools.py` wrapping each `tools.py` function as a `StructuredTool` (schema inferred from signature + docstring; original callable preserved as `.func`), with the richer `TOOL_DECLARATIONS` guidance folded into the relevant docstrings. New file `agent/graph.py` containing `DJState`, the node functions, the router, and `build_dj_graph`. No call sites change yet; `loop.py` still drives everything. Add a unit test asserting every generated tool's name/args-schema matches the current `TOOL_DECLARATIONS` (so the `@tool` inference doesn't silently drift from what the LLM sees today).

**Phase 2 — Port `run_agent_cycle` (1 day).** Keep the pre-fetch and fallback; replace the `_run_react_loop(...)` call with a `cycle_graph.invoke(initial_state)` and normalise the returned `DJState` into today's return dict.

**Phase 3 — Port `start_session`** (1 day). Move the `select_opening_track` guard into the opener graph's tool node; replace `_react_pick_opener`'s `_run_react_loop` call with `opener_graph.invoke(...)`.

**Phase 4 — Parity validation + cutover (1 day).** Run the parity tests (§5). All migration work happens on a **dedicated branch** with a **hard cut** — no feature flag, no dual-run path (decision settled, §6.7). Once parity tests are green on the branch, delete `_run_react_loop`, `_groq_call`, and dead schema plumbing, update `CLAUDE.md` and the dissertation §3 text, then merge. If the implementation fails, the branch is simply abandoned with no impact on `main`.

Estimated total: ~4–5 focused days including testing. Tools, state model, and UI are untouched, which keeps the blast radius small.

---

## 5. Verification strategy (this is also dissertation evidence)

1. **Contract tests** — `run_agent_cycle` and `start_session` must return dicts with identical keys/types to today. Extend `tests/test_loop_unit.py`.
2. **Trace-shape test** — assert every emitted entry has `{kind ∈ think|act|observe|explain, content, tool_name, tool_args, tool_result}` so `bridge.adapt_trace()` keeps working. The Streamlit UI is the real acceptance test — load it and confirm the Agent Trace tab renders unchanged.
3. **Parity harness** — with a stubbed/mocked LLM that returns a fixed tool-call script, run the old loop and the new graph over the same script and assert **behavioural** parity: same tool calls, same queued track, same final explanation. Do **not** demand byte-identical traces — the new graph reads `reasoning_content` and will therefore emit *more* `think` entries than the old loop (which drops action-turn reasoning). Assert the new trace has ≥ the think entries of the old.
4. **Router unit tests** — no-tool-calls → END; stop-tool-success → explain → END; otherwise → tools; recursion cap honoured.
5. **Backoff test** — assert `ChatGroq` is constructed with `max_retries=4` and that a simulated 429 is retried (not surfaced) by the native retry layer. Exact inter-attempt timing is no longer asserted (native retries use their own exponential-backoff-with-jitter schedule, not the old 2/4/8/16s).
6. **Full pytest** must stay green (`test_state`, `test_camelot`, `test_tags*` are unaffected; `test_loop*` updated).

---

## 6. Decisions (settled)

1. **Tool schema source — `@tool` / `StructuredTool`. ✅ settled.** Wrap the `tools.py` functions as LangChain tool objects in a new `agent/lc_tools.py`; this single list drives both `bind_tools(...)` and the `ToolNode`, removing the `TOOL_DECLARATIONS`/`GROQ_TOOLS`/`TOOL_REGISTRY` triplet. **Wrap, don't decorate in place** — the `tools.py` functions must stay plain callables because `bridge.py` and the Python pre-step/fallback call them directly; decorating in place would break those callers. Fold the richer guidance from the current `TOOL_DECLARATIONS` into the docstrings, and add a schema-parity test (Phase 1) so the inferred schemas match what the LLM sees today.
2. **Graph construction — ONE FACTORY.** A single `build_dj_graph(tools, registry, stop_tool, max_iterations)` called twice (cycle graph + opener graph). *(Adopted lean.)*
3. **Pre-fetch shortlist — PYTHON PRE-STEP.** Stays procedural and wraps the graph invoke; not modelled as a node. *(Adopted lean.)*
4. **Rate-limit backoff — NATIVE RETRIES. ✅ settled.** Use `ChatGroq(max_retries=4)`; delete `_groq_call`. Native retries are exponential-backoff-with-jitter, so the dissertation's "exponential backoff" claim (Risk 1) still holds — just update any prose that quotes the literal 2/4/8/16s schedule.
5. **`select_opening_track` `seen_tracks` guard — IN `DJState`.** Lives in graph state so it's inspectable and testable. *(Adopted lean.)*
6. **Checkpointing / SQLite — DEFERRED. ✅ settled.** This migration explicitly **excludes** `SqliteSaver` and the analytical trace tables. LangGraph lands first; SQLite is a separate follow-up workstream tied to the evaluation chapter.
7. **Migration safety — HARD CUT ON A BRANCH. ✅ settled.** All work on a dedicated branch; no `AGDJ_USE_LANGGRAPH` flag, no dual-run. Parity tests must pass on the branch before merge. If it fails, abandon the branch — `main` is untouched.

---

## 7. Risks

- **Tool-calling parity on `gpt-oss-120b` via `ChatGroq`** (Phase 0 de-risks this). If `ChatGroq` mangles the tool format or drops reasoning text, the trace fidelity the dissertation's failure taxonomy relies on degrades.
- **Trace-shape drift** silently breaking the Streamlit panel — caught by test #2/#3.
- **Backoff timing change (accepted).** Native retries replace the literal 2/4/8/16s schedule (decision 4). Behaviourally equivalent (still exponential, still recovers from 429s); only the inter-attempt timing differs. Update any dissertation prose that quotes the old numbers.
- **`@tool` schema-inference drift.** `StructuredTool.from_function` infers the args schema from signatures/type hints; it could differ subtly from the hand-written `TOOL_DECLARATIONS` the LLM is tuned against (e.g. a missing default, an `int` vs `number`, lost description nuance). Mitigated by the Phase 1 schema-parity test and by folding declaration guidance into docstrings.
- **Decorating `tools.py` in place would break direct callers** in `bridge.py` and the Python pre-step. Mitigation is structural: wrap in `lc_tools.py`, keep `tools.py` functions plain (decision 6.1).
- **Scope creep** — the temptation to also do SQLite and checkpointing in one go. Keep them out (decision 6).
- **Dependency weight** — `langchain-groq` + `langgraph` pull in a non-trivial dependency tree; confirm it doesn't conflict with the pinned Python 3.11 / sentence-transformers stack.

---

## 8. Definition of done

- `run_agent_cycle` and `start_session` run entirely through compiled LangGraph graphs.
- All existing tests pass; new contract/parity/router tests added.
- Streamlit Agent Trace, Queue, and Now Playing tabs behave identically.
- `_run_react_loop` and dead schema plumbing removed.
- `pyproject.toml`/`requirements.txt` reflect actually-used dependencies.
- `CLAUDE.md` and dissertation §3.1/§3.4/§3.7 updated to describe the real LangGraph architecture.
