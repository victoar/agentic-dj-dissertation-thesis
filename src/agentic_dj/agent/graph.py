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

from langchain_core.messages import ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq


# Defined locally so importing this module does not import loop.py (which
# instantiates a Groq client at import time and needs GROQ_API_KEY).
MODEL: str = "openai/gpt-oss-120b"
MAX_ITERATIONS: int = 15


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


# ── Shared nodes & routers ──────────────────────────────────────────────────────

def _make_agent_node(llm_with_tools):
    def agent_node(state: DJState) -> dict:
        response = llm_with_tools.invoke(state["messages"])
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


def _make_explain_node(llm):
    def explain_node(state: DJState) -> dict:
        trace = list(state.get("trace", []))
        step  = state.get("step", 1)
        try:
            response = llm.invoke(state["messages"])        # no tools bound
            explanation = _message_text(response) or _extract_reasoning(response)
        except Exception:                                    # noqa: BLE001
            explanation = ""                                 # caller synthesises a fallback
        if explanation:
            trace.append(_trace_entry(step, "explain", explanation))
            step += 1
        return {"trace": trace, "step": step, "explanation": explanation}
    return explain_node


def _route_after_agent(state: DJState) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END


def _route_after_tools(state: DJState) -> str:
    return "explain" if state.get("terminal") else "agent"


def _assemble(llm, llm_with_tools, tools_node):
    """Wire the shared agent/explain nodes + routers around a given tools node."""
    g = StateGraph(DJState)
    g.add_node("agent", _make_agent_node(llm_with_tools))
    g.add_node("tools", tools_node)
    g.add_node("explain", _make_explain_node(llm))
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", _route_after_agent, {"tools": "tools", END: END})
    g.add_conditional_edges("tools", _route_after_tools, {"explain": "explain", "agent": "agent"})
    g.add_edge("explain", END)
    return g.compile()


# ── Standard tools node (mid-session cycle) ─────────────────────────────────────

def _make_standard_tools_node(registry: dict, stop_tool: str):
    def tools_node(state: DJState) -> dict:
        last  = state["messages"][-1]
        trace = list(state.get("trace", []))
        step  = state.get("step", 1)
        terminal = state.get("terminal")
        out_messages = []

        for tc in last.tool_calls:
            name = tc["name"]
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

        return {"messages": out_messages, "trace": trace, "step": step, "terminal": terminal}
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
            name = tc["name"]
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

def _make_llm(model: str, temperature: float, max_retries: int):
    return ChatGroq(model=model, temperature=temperature, max_retries=max_retries)


def build_dj_graph(
    tools:       list,
    stop_tool:   str,
    model:       str = MODEL,
    temperature: float = 0.3,
    max_retries: int = 4,
):
    """Compile the mid-session cycle graph (stops on ``stop_tool`` success)."""
    llm = _make_llm(model, temperature, max_retries)
    registry = {t.name: t for t in tools}
    return _assemble(llm, llm.bind_tools(tools),
                     _make_standard_tools_node(registry, stop_tool))


def build_opener_graph(
    tools:       list,
    stop_tool:   str = "select_opening_track",
    model:       str = MODEL,
    temperature: float = 0.3,
    max_retries: int = 4,
):
    """
    Compile the natural-language session-start graph. ``tools`` should be the
    opener toolset (search tools + the synthetic stop tool object). The stop
    tool's body is never executed — the opener tools node validates the choice
    against ``seen_tracks`` in graph state.
    """
    llm = _make_llm(model, temperature, max_retries)
    registry = {t.name: t for t in tools}
    return _assemble(llm, llm.bind_tools(tools),
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
