"""
LangGraph agent graph for AgDJ (migration).

Compiled ReAct graphs that replace the hand-rolled ``_run_react_loop`` in
``loop.py``. Two builders share the same agent/explain nodes and routers:

  • build_dj_graph     — mid-session cycle. Stops on add_track_to_queue.
  • build_opener_graph — natural-language session start. Stops on
                         select_opening_track, with an anti-hallucination guard:
                         the model may only commit a track it has actually seen
                         in a prior search result. The ``seen_tracks`` set lives
                         in DJState (not a closure) so it is inspectable/testable.

Control flow (mirrors the original loop):

    START → agent
    agent → tool_calls?  yes → tools      no → END   (final text = explanation)
    tools → stop tool fired?  yes → explain → END     no → agent

Reasoning is read from additional_kwargs["reasoning_content"] (gpt-oss puts it
on a separate channel; .content is empty on tool-call turns). Backoff is
delegated to ChatGroq(max_retries=4) — there is no custom _groq_call here.
"""

import json
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq


# Defined locally so importing this module does not import loop.py (which
# instantiates a Groq client at import time and needs GROQ_API_KEY).
MODEL: str = "openai/gpt-oss-120b"
MAX_ITERATIONS: int = 15
# gpt-oss on Groq intermittently emits a malformed tool call (harmony channel
# token leaked into the name) that Groq rejects with a 400 `tool_use_failed`.
# It is sampling-dependent, so re-invoking usually returns a clean call.
TOOL_RETRY_ATTEMPTS: int = 3


def _is_tool_validation_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "tool_use_failed" in msg
        or "tool call validation failed" in msg
        or "<|channel|>" in msg
    )


# ── Graph state ───────────────────────────────────────────────────────────────

class DJState(TypedDict, total=False):
    """State carried through the graph.

    ``messages`` uses the add_messages reducer (append semantics). All other
    keys are plain — a node returns the new value and LangGraph overwrites.
    """
    messages:    Annotated[list, add_messages]
    trace:       list[dict]          # _trace_entry-compatible dicts
    step:        int                 # trace step counter
    terminal:    Optional[dict]      # result dict of the stop tool, once it succeeds
    explanation: str
    seen_tracks: dict                # opener anti-hallucination set: "name|artist" -> {name, artist}
    choice:      Optional[dict]      # opener: the committed {name, artist}


# ── Trace helper (same shape as the original loop._trace_entry) ─────────────────

def _trace_entry(
    step:        int,
    kind:        str,                 # "think" | "act" | "observe" | "explain"
    content:     str,
    tool_name:   Optional[str] = None,
    tool_args:   Optional[dict] = None,
    tool_result: Optional[dict] = None,
) -> dict:
    return {
        "step":        step,
        "kind":        kind,
        "content":     content,
        "tool_name":   tool_name,
        "tool_args":   tool_args,
        "tool_result": tool_result,
    }


def _extract_reasoning(message: Any) -> str:
    """
    Pull the model's reasoning text. gpt-oss on Groq returns it on a separate
    channel — additional_kwargs['reasoning_content'] via ChatGroq — and leaves
    .content empty on tool-call turns. Fall back to .content for the final turn.
    """
    extra = getattr(message, "additional_kwargs", {}) or {}
    reasoning = extra.get("reasoning_content") or extra.get("reasoning") or ""
    if not reasoning:
        content = getattr(message, "content", "") or ""
        reasoning = content if isinstance(content, str) else ""
    return reasoning.strip()


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "") or ""
    return content.strip() if isinstance(content, str) else ""


def _clean_tool_name(name: str) -> str:
    """
    Strip gpt-oss "harmony" channel tokens / namespace prefixes that Groq
    sometimes leaks into a tool name, e.g. 'search_tracks<|channel|>commentary'
    or 'functions.search_tracks'. Without this the tool name does not match the
    registry (and Groq 400s the call unless disable_tool_validation is set).
    """
    if not name:
        return name
    name = name.split("<|")[0].strip()        # drop harmony channel suffix
    if name.startswith("functions."):
        name = name[len("functions."):]
    return name


def _sanitize_tool_calls(message: Any) -> None:
    """
    In-place: clean leaked tokens from tool-call names on both the parsed
    ``.tool_calls`` and the raw ``additional_kwargs['tool_calls']`` so that
    execution, the trace, AND the message history sent on the next turn all
    see the real tool name.
    """
    for tc in getattr(message, "tool_calls", None) or []:
        if isinstance(tc, dict) and tc.get("name"):
            tc["name"] = _clean_tool_name(tc["name"])
    raw = (getattr(message, "additional_kwargs", {}) or {}).get("tool_calls")
    for tc in raw or []:
        fn = tc.get("function") if isinstance(tc, dict) else None
        if fn and fn.get("name"):
            fn["name"] = _clean_tool_name(fn["name"])


# ── Shared nodes & routers ──────────────────────────────────────────────────────

def _invoke_with_tool_retry(llm_with_tools, messages):
    """
    Invoke the tool-bound LLM, retrying on Groq's `tool_use_failed` 400 (a
    malformed/corrupted tool call). If every attempt fails, return a tool-less
    AIMessage so the graph ends gracefully and the Python fallback selects a
    track — far better than crashing the cycle.
    """
    last_exc = None
    for _ in range(TOOL_RETRY_ATTEMPTS):
        try:
            return llm_with_tools.invoke(messages)
        except Exception as exc:                       # noqa: BLE001
            if not _is_tool_validation_error(exc):
                raise
            last_exc = exc                             # re-sample on next loop
    return AIMessage(
        content="",
        additional_kwargs={"reasoning_content":
                           f"Tool-call generation failed after retries: {last_exc}"},
    )


def _make_agent_node(llm_with_tools):
    def agent_node(state: DJState) -> dict:
        response = _invoke_with_tool_retry(llm_with_tools, state["messages"])
        _sanitize_tool_calls(response)   # gpt-oss leaks harmony tokens into tool names
        trace = list(state.get("trace", []))
        step  = state.get("step", 1)

        reasoning = _extract_reasoning(response)
        if reasoning:
            trace.append(_trace_entry(step, "think", reasoning))
            step += 1

        update: dict = {"messages": [response], "trace": trace, "step": step}

        # No tool calls → the assistant has given its final answer.
        if not getattr(response, "tool_calls", None):
            explanation = _message_text(response) or reasoning
            if explanation:
                trace.append(_trace_entry(step, "explain", explanation))
                step += 1
                update["trace"] = trace
                update["step"]  = step
            update["explanation"] = explanation
        return update
    return agent_node


def _route_after_agent(state: DJState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def _route_after_tools(state: DJState) -> str:
    # Terminal (stop tool succeeded) → END. The explanation is the `reason` the
    # model passed at commit (no separate explain turn). Otherwise loop back.
    return END if state.get("terminal") else "agent"


def _assemble(llm_with_tools, tools_node):
    """Wire the shared agent node + routers around a given tools node."""
    g = StateGraph(DJState)
    g.add_node("agent", _make_agent_node(llm_with_tools))
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", END: END})
    g.add_conditional_edges("tools", _route_after_tools, {"agent": "agent", END: END})
    return g.compile()


# ── Standard tools node (mid-session cycle) ─────────────────────────────────────

def _make_standard_tools_node(registry: dict, stop_tool: str):
    def tools_node(state: DJState) -> dict:
        last  = state["messages"][-1]
        trace = list(state.get("trace", []))
        step  = state.get("step", 1)
        terminal    = state.get("terminal")
        explanation = state.get("explanation", "")
        out_messages = []

        for tc in last.tool_calls:
            name = _clean_tool_name(tc["name"])
            args = tc.get("args", {}) or {}
            tool = registry.get(name)
            if tool is None:
                result = {"error": f"Unknown tool: {name}"}
            else:
                try:
                    result = tool.invoke(args)
                except Exception as exc:                       # noqa: BLE001
                    result = {"error": f"Tool '{name}' raised: {exc}"}

            trace.append(_trace_entry(
                step=step, kind="act", content=name,
                tool_name=name, tool_args=args,
                tool_result=result if isinstance(result, dict) else {"value": result},
            ))
            step += 1
            out_messages.append(ToolMessage(content=json.dumps(result, default=str),
                                            tool_call_id=tc["id"]))

            if name == stop_tool and isinstance(result, dict) and result.get("success"):
                terminal = result
                # The committing turn carries its own explanation via `reason`.
                if result.get("reason"):
                    explanation = result["reason"]
                    trace.append(_trace_entry(step, "explain", explanation))
                    step += 1

        return {"messages": out_messages, "trace": trace, "step": step,
                "terminal": terminal, "explanation": explanation}
    return tools_node


# ── Opener tools node (natural-language session start) ──────────────────────────

def _register_candidates(seen: dict, candidates: list) -> None:
    for c in candidates:
        name = c.get("name", "")
        if not name:
            continue
        key = f"{name.lower()}|{c.get('artist', '').lower()}"
        seen[key] = {"name": name, "artist": c.get("artist", "")}


def _validate_opening_choice(track_name: str, artist: str, seen: dict):
    """
    Port of the original select_opening_track guard. Returns (result_dict, picked
    or None). A track may only be committed if it was seen in a prior search.
    """
    key = f"{track_name.lower()}|{artist.lower()}"
    if key in seen:
        picked = {"name": track_name, "artist": artist}
        return {"success": True, "selected": picked}, picked

    # Partial name match (handles minor title punctuation differences)
    name_lower = track_name.lower()
    for k, v in seen.items():
        if k.split("|")[0] == name_lower:
            picked = {"name": v["name"], "artist": v["artist"]}
            return {"success": True, "selected": picked}, picked

    available = ", ".join(
        f"'{v['name']}' by {v['artist']}" for v in list(seen.values())[:6]
    ) or "none yet — call search_tracks_by_playlist first"
    return {
        "success": False,
        "error": (
            f"'{track_name}' by {artist} was not in your search results and cannot "
            f"be committed. You must select a track you have actually seen returned "
            f"by a search tool this session. Available candidates include: {available}."
        ),
    }, None


def _make_opener_tools_node(registry: dict, stop_tool: str):
    def tools_node(state: DJState) -> dict:
        last  = state["messages"][-1]
        trace = list(state.get("trace", []))
        step  = state.get("step", 1)
        terminal = state.get("terminal")
        choice   = state.get("choice")
        seen     = dict(state.get("seen_tracks", {}))
        out_messages = []

        for tc in last.tool_calls:
            name = _clean_tool_name(tc["name"])
            args = tc.get("args", {}) or {}

            if name == stop_tool:
                # Validated against graph state — the tool body is never run.
                result, picked = _validate_opening_choice(
                    args.get("track_name", ""), args.get("artist", ""), seen
                )
                if result.get("success"):
                    terminal = result
                    choice   = picked
            else:
                tool = registry.get(name)
                if tool is None:
                    result = {"error": f"Unknown tool: {name}"}
                else:
                    try:
                        result = tool.invoke(args)
                    except Exception as exc:                   # noqa: BLE001
                        result = {"error": f"Tool '{name}' raised: {exc}"}
                # Register everything the model has now seen.
                if isinstance(result, dict):
                    if isinstance(result.get("candidates"), list):
                        _register_candidates(seen, result["candidates"])
                    elif result.get("name") and not result.get("error"):
                        _register_candidates(seen, [result])

            trace.append(_trace_entry(
                step=step, kind="act", content=name,
                tool_name=name, tool_args=args,
                tool_result=result if isinstance(result, dict) else {"value": result},
            ))
            step += 1
            out_messages.append(ToolMessage(content=json.dumps(result, default=str),
                                            tool_call_id=tc["id"]))

        return {
            "messages": out_messages, "trace": trace, "step": step,
            "terminal": terminal, "choice": choice, "seen_tracks": seen,
        }
    return tools_node


# ── Graph factories ─────────────────────────────────────────────────────────────

def _make_llm(model: str, temperature: float, max_retries: int,
              reasoning_effort: str | None = None):
    # disable_tool_validation: gpt-oss-120b on Groq intermittently leaks a harmony
    # channel token into the tool name (e.g. 'search_tracks<|channel|>commentary'),
    # which Groq's server otherwise rejects with a 400 mid-generation. With
    # validation off the call is returned and we clean the name (_sanitize_tool_calls);
    # unknown tools are still handled gracefully by the tools node.
    # reasoning_effort (low/medium/high) tunes how long gpt-oss "thinks" per turn.
    # It is an explicit ChatGroq field, so it must be passed directly — NOT via
    # model_kwargs (which is only for non-field passthrough like disable_tool_validation).
    kwargs: dict = dict(
        model=model,
        temperature=temperature,
        max_retries=max_retries,
        model_kwargs={"disable_tool_validation": True},
    )
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    return ChatGroq(**kwargs)


def build_dj_graph(
    tools:            list,
    stop_tool:        str,
    model:            str = MODEL,
    temperature:      float = 0.3,
    max_retries:      int = 4,
    reasoning_effort: str | None = None,
):
    """Compile the mid-session cycle graph (stops on ``stop_tool`` success)."""
    llm = _make_llm(model, temperature, max_retries, reasoning_effort)
    registry = {t.name: t for t in tools}
    return _assemble(llm.bind_tools(tools),
                     _make_standard_tools_node(registry, stop_tool))


def build_opener_graph(
    tools:            list,
    stop_tool:        str = "select_opening_track",
    model:            str = MODEL,
    temperature:      float = 0.3,
    max_retries:      int = 4,
    reasoning_effort: str | None = None,
):
    """
    Compile the natural-language session-start graph. ``tools`` should be the
    opener toolset (search tools + the synthetic stop tool object). The stop
    tool's body is never executed — the opener tools node validates the choice
    against ``seen_tracks`` in graph state.
    """
    llm = _make_llm(model, temperature, max_retries, reasoning_effort)
    registry = {t.name: t for t in tools}
    return _assemble(llm.bind_tools(tools),
                     _make_opener_tools_node(registry, stop_tool))


# ── Convenience runner (used by loop.py) ────────────────────────────────────────

def run_graph(
    graph,
    messages:        list,
    seen_tracks:     Optional[dict] = None,
    max_iterations:  int = MAX_ITERATIONS,
) -> dict:
    """
    Invoke a compiled graph and normalise the final state into the shape the
    public loop functions expect: {"queued", "explanation", "trace", "choice"}.
    ``recursion_limit`` is derived from max_iterations (agent+tools = 2 steps per
    ReAct turn, plus margin for the explain node).
    """
    initial: DJState = {
        "messages":    messages,
        "trace":       [],
        "step":        1,
        "terminal":    None,
        "explanation": "",
        "seen_tracks": seen_tracks if seen_tracks is not None else {},
        "choice":      None,
    }
    final = graph.invoke(initial, config={"recursion_limit": max_iterations * 2 + 2})
    return {
        "queued":      final.get("terminal"),
        "explanation": final.get("explanation", ""),
        "trace":       final.get("trace", []),
        "choice":      final.get("choice"),
    }
