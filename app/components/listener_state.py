import streamlit as st
import plotly.graph_objects as go

MOCK_STATE = {
    "energy":        0.78,
    "valence":       0.62,
    "focus":         0.55,
    "openness":      0.70,
    "social":        0.35,
    "arc_phase":     "build",
    "tracks_played": 8,
}

# Planned arc: energy targets per track index across a ~20-track session
_PLANNED_X = [0,  2,   4,   6,   8,   10,  12,  14,  16,  18,  20]
_PLANNED_Y = [0.25, 0.32, 0.45, 0.60, 0.72, 0.82, 0.88, 0.88, 0.80, 0.55, 0.30]

# Simulated actual energy values up to "now" (track 8)
_ACTUAL_X = [0,    1,    2,    3,    4,    5,    6,    7,    8]
_ACTUAL_Y = [0.28, 0.30, 0.40, 0.52, 0.65, 0.68, 0.74, 0.76, 0.78]

# Metric card colour per dimension
_COLOURS = {
    "energy":   "#e05252",
    "valence":  "#7c6fe0",
    "focus":    "#4a90d9",
    "openness": "#3aaa6e",
    "social":   "#e0993a",
    "arc":      "#d95f8a",
}

_ARC_ARROWS = {"warmup": "→", "build": "↑", "peak": "▲", "cooldown": "↓"}


def _metric_card(label: str, value, colour: str) -> str:
    if isinstance(value, float):
        display = f"{value:.2f}"
        pct     = value * 100
    else:
        display = value
        pct     = 50  # arc phase bar always shown at half

    return f"""
    <div style="
        background:#f5f5f0;
        border-radius:10px;
        padding:14px 16px 12px;
        height:90px;
        box-sizing:border-box;
    ">
        <div style="font-size:0.82rem;color:#666;margin-bottom:6px;">{label}</div>
        <div style="
            background:#e0e0da;
            border-radius:4px;
            height:5px;
            width:100%;
            margin-bottom:7px;
            overflow:hidden;
        ">
            <div style="
                background:{colour};
                height:100%;
                width:{pct:.1f}%;
                border-radius:4px;
            "></div>
        </div>
        <div style="font-size:1.05rem;font-weight:600;color:#222;">{display}</div>
    </div>
    """


def _arc_chart(state: dict) -> go.Figure:
    now_x = state.get("tracks_played", 8)

    # Interpolate actual_y at now_x for the "Now" marker
    now_y = _ACTUAL_Y[-1]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=_PLANNED_X, y=_PLANNED_Y,
        mode="lines",
        line=dict(color="#b0b0b0", dash="dash", width=2),
        name="Planned arc",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=_ACTUAL_X, y=_ACTUAL_Y,
        mode="lines",
        line=dict(color="#5c5bd6", width=2.5),
        name="Actual",
        hoverinfo="skip",
    ))

    fig.add_trace(go.Scatter(
        x=[now_x], y=[now_y],
        mode="markers",
        marker=dict(color="#5c5bd6", size=12, line=dict(color="#fff", width=2)),
        name="Now",
        hoverinfo="skip",
    ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=8, b=0),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        xaxis=dict(
            tickvals=[0, now_x, 20],
            ticktext=["Start", "Now", "End"],
            tickfont=dict(color="#888", size=11),
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            tickvals=[0.1, 0.9],
            ticktext=["Low", "High"],
            tickfont=dict(color="#888", size=11),
            showgrid=False,
            zeroline=False,
            range=[0, 1],
        ),
    )
    return fig


def render(state: dict | None = None):
    """
    Render the Listener State tab.
    state: dict from get_listener_state() or None to show mock data.
    """
    data = state or MOCK_STATE
    arc  = data.get("arc_phase", "build")
    arrow = _ARC_ARROWS.get(arc, "")

    with st.container(border=True):
        st.markdown(
            "<p style='font-size:0.9rem;color:#444;margin-bottom:10px;'>"
            "Current listener state vector</p>",
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3, gap="small")

        with col1:
            st.markdown(
                _metric_card("Energy",   data["energy"],   _COLOURS["energy"]),
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                _metric_card("Valence",  data["valence"],  _COLOURS["valence"]),
                unsafe_allow_html=True,
            )
        with col3:
            st.markdown(
                _metric_card("Focus",    data["focus"],    _COLOURS["focus"]),
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        col4, col5, col6 = st.columns(3, gap="small")

        with col4:
            st.markdown(
                _metric_card("Openness", data["openness"], _COLOURS["openness"]),
                unsafe_allow_html=True,
            )
        with col5:
            st.markdown(
                _metric_card("Social",   data["social"],   _COLOURS["social"]),
                unsafe_allow_html=True,
            )
        with col6:
            st.markdown(
                _metric_card("Session arc", f"{arc.capitalize()} {arrow}", _COLOURS["arc"]),
                unsafe_allow_html=True,
            )

        st.markdown(
            "<p style='font-size:0.9rem;color:#444;margin-top:16px;margin-bottom:0;'>"
            "Energy arc — session trajectory</p>",
            unsafe_allow_html=True,
        )
        st.plotly_chart(_arc_chart(data), use_container_width=True, config={"displayModeBar": False})

        st.markdown(
            "<p style='font-size:0.8rem;color:#888;margin-top:-10px;'>"
            "Purple line = actual session. Dashed = planned arc. "
            "Currently tracking the plan closely.</p>",
            unsafe_allow_html=True,
        )
