import streamlit as st

MOCK_TRACE = [
    {
        "kind":    "think",
        "content": (
            "Listener state: energy 0.78, valence 0.62, session arc = building toward peak. "
            "Both previous tracks were listened to in full — no skips. Openness is moderate "
            "(0.70) so I can introduce a slightly less familiar track."
        ),
    },
    {
        "kind":    "act",
        "content": (
            "key_compatible(key=8, mode=1) → compatible_keys([1, 3, 5, 8, 10, 13, 15, 17, 18])\n"
            "target_tempo_range = [100, 125]"
        ),
    },
    {
        "kind":    "observe",
        "content": (
            "Compatible keys identified. Target profile: energy 0.70–0.85, BPM 100–125, "
            "valence 0.55–0.75, harmonic with A♭ major."
        ),
    },
    {
        "kind":    "act",
        "content": (
            "search_tracks(energy=[0.70, 0.85], tempo=[100, 125], key_filter=[1, 3, 5, 8])\n"
            "→ 34 candidates returned"
        ),
    },
    {
        "kind":    "observe",
        "content": (
            "Top candidates by combined score: Foster the People (0.89), "
            "Two Door Cinema Club (0.84), MGMT (0.16). "
            "Checking for recent play history to avoid repetition."
        ),
    },
    {
        "kind":    "act",
        "content": (
            "Foster the People — Pumped Up Kicks selected. BPM 111 (harmonically adjacent "
            "to A♭), energy 0.74 — slightly lower to avoid fatigue before the peak. "
            "Not played in this session. Openness permits it. Selecting."
        ),
    },
    {
        "kind":    "act",
        "content": "add_to_queue(track_id='spotify:track:7MbLCl21...', position=1)",
    },
]

_KIND_STYLE = {
    "think":   ("Think",   "#f0f0eb", "#666"),
    "act":     ("Act",     "#eeedf9", "#5c5bd6"),
    "observe": ("Observe", "#ebf5ee", "#3aaa6e"),
}


def _trace_row(entry: dict) -> str:
    label, bg, colour = _KIND_STYLE.get(entry["kind"], ("—", "#f5f5f0", "#888"))
    # Use monospace for act entries (tool calls), normal font otherwise
    font  = "font-family:monospace;font-size:0.82rem;" if entry["kind"] == "act" else "font-size:0.88rem;"
    # Preserve newlines
    text  = entry["content"].replace("\n", "<br>")

    return f"""
    <div style="display:flex;align-items:flex-start;gap:12px;padding:11px 4px;">
      <span style="
          background:{bg};color:{colour};
          border-radius:6px;
          padding:2px 9px;
          font-size:0.75rem;
          font-weight:600;
          white-space:nowrap;
          margin-top:2px;
          min-width:58px;
          text-align:center;
          display:inline-block;
      ">{label}</span>
      <div style="{font}color:#333;line-height:1.55;flex:1;">{text}</div>
    </div>
    """


def _divider() -> str:
    return "<hr style='border:none;border-top:1px solid #e8e8e3;margin:0 4px;'>"


def render(trace: list | None = None, current_track: str = "Midnight City"):
    """
    Render the Agent Trace tab.
    trace: list of dicts with 'kind' and 'content' keys, or None for mock data.
    current_track: name of the track the agent was selecting for (used in the heading).
    """
    items = trace or MOCK_TRACE

    with st.container(border=True):
        st.markdown(
            f"<p style='font-size:0.9rem;color:#444;margin-bottom:4px;'>"
            f"Agent reasoning trace — selecting track after {current_track}</p>",
            unsafe_allow_html=True,
        )

        rows_html = ""
        for i, entry in enumerate(items):
            if i > 0:
                rows_html += _divider()
            rows_html += _trace_row(entry)

        st.markdown(rows_html, unsafe_allow_html=True)
