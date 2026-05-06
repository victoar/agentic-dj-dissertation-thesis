import streamlit as st

MOCK = {
    "track_name":  "Midnight City",
    "artist":      "M83",
    "album":       "Hurry Up, We're Dreaming",
    "energy_est":  0.82,
    "valence_est": 0.62,
    "bpm":         105,
    "key":         "A♭ maj",
    "progress_ms": 45_000,
    "duration_ms": 243_000,
    "reasoning": (
        "Your last two tracks were fast and energetic. You listened to both all the way "
        "through, so I’m keeping the energy high — but shifting toward something "
        "slightly more melodic to avoid fatigue."
    ),
}


def _fmt_ms(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


def render(track: dict | None = None, on_feedback=None):
    """
    Render the Now Playing tab.
    track: enriched candidate dict from tools.py (or None to show mock data).
    on_feedback: callable(event: str) triggered by feedback buttons.
    """
    data = track or MOCK

    with st.container(border=True):
        col_art, col_info = st.columns([1, 5], gap="large")

        with col_art:
            st.markdown(
                """
                <div style="
                    background:#ede8f5;
                    border-radius:12px;
                    width:110px;
                    height:110px;
                    display:flex;
                    align-items:center;
                    justify-content:center;
                    font-size:2.8rem;
                ">🎵</div>
                """,
                unsafe_allow_html=True,
            )

        with col_info:
            st.markdown(f"### {data['track_name']}")
            st.markdown(
                f"<span style='color:#555;font-size:0.95rem;'>"
                f"<b>{data['artist']}</b> &middot; {data['album']}</span>",
                unsafe_allow_html=True,
            )

            # ── Badges ────────────────────────────────────────────────────────
            badge_style = (
                "display:inline-block;"
                "border:1px solid #d0d0d0;"
                "border-radius:20px;"
                "padding:2px 10px;"
                "font-size:0.82rem;"
                "margin-right:6px;"
                "margin-top:8px;"
                "background:#fff;"
                "color:#222;"
            )
            badges = [
                f"Energy {data['energy_est']:.2f}",
                f"Valence {data['valence_est']:.2f}",
                f"BPM {data.get('bpm') or '—'}",
                f"Key {data.get('key', '—')}",
            ]
            badge_html = "".join(
                f'<span style="{badge_style}">{b}</span>' for b in badges
            )
            st.markdown(badge_html, unsafe_allow_html=True)

        # ── Progress bar ──────────────────────────────────────────────────────
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

        progress = data["progress_ms"] / max(data["duration_ms"], 1)
        elapsed  = _fmt_ms(data["progress_ms"])
        total    = _fmt_ms(data["duration_ms"])

        st.markdown(
            f"""
            <div style="margin-bottom:2px;">
              <div style="
                  background:#e0e0e0;
                  border-radius:4px;
                  height:5px;
                  width:100%;
                  overflow:hidden;
              ">
                <div style="
                    background:#5c5bd6;
                    height:100%;
                    width:{progress*100:.1f}%;
                    border-radius:4px;
                "></div>
              </div>
              <div style="display:flex;justify-content:space-between;
                          font-size:0.78rem;color:#888;margin-top:3px;">
                <span>{elapsed}</span><span>{total}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Feedback buttons ──────────────────────────────────────────────────
        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
        b1, b2, b3, b4 = st.columns(4)

        with b1:
            if st.button("↩ Skip", use_container_width=True):
                if on_feedback:
                    on_feedback("skip")

        with b2:
            if st.button("↺ Replay", use_container_width=True):
                if on_feedback:
                    on_feedback("replay")

        with b3:
            if st.button("♥ Love", use_container_width=True):
                if on_feedback:
                    on_feedback("thumbs_up")

        with b4:
            if st.button("Next →", use_container_width=True):
                if on_feedback:
                    on_feedback("full_listen")

    # ── Agent reasoning ───────────────────────────────────────────────────────
    st.markdown(
        "<p style='font-size:0.88rem;color:#555;margin-top:12px;"
        "margin-bottom:4px;'>Agent reasoning</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div style="
            background:#f0f0eb;
            border-radius:10px;
            padding:14px 16px;
            font-size:0.92rem;
            color:#333;
            line-height:1.55;
        ">{data['reasoning']}</div>
        """,
        unsafe_allow_html=True,
    )
