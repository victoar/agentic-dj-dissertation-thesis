import streamlit as st

import bridge
from components.now_playing    import render as render_now_playing
from components.listener_state import render as render_listener_state
from components.queue          import render as render_queue
from components.agent_trace    import render as render_agent_trace

st.set_page_config(
    page_title="Agentic DJ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.sidebar.markdown(
    "BPM & key data: [GetSongKey](https://getsongkey.com)",
    unsafe_allow_html=True,
)

st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stApp { background-color: #f5f5f0; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 800px; }
</style>
""", unsafe_allow_html=True)

bridge.init_session()

st.markdown("## The practical application")

if not st.session_state.get("initialised"):
    st.info(
        "Spotify is not connected. Run the app locally with valid credentials in `.env` to start a session.",
        icon="ℹ️",
    )
    st.stop()

tab_now, tab_state, tab_queue, tab_trace = st.tabs([
    "Now playing", "Listener state", "Queue", "Agent trace"
])

# ── Now Playing — polls Spotify every 5 s ─────────────────────────────────────
with tab_now:
    @st.fragment(run_every=5)
    def now_playing_fragment():
        changed = bridge.detect_and_handle_track_change()
        if changed:
            bridge.ensure_buffer(2)

        if st.session_state.current_playback.get("playing"):
            render_now_playing(
                track=bridge.adapt_now_playing(),
                on_feedback=bridge.handle_feedback,
            )
        else:
            st.info("No track currently playing. Open Spotify on any device to get started.")

    now_playing_fragment()

# ── Listener State — updated after every feedback event ───────────────────────
with tab_state:
    render_listener_state(state=st.session_state.listener_state)

with tab_queue:
    now_playing_data, queue_data = bridge.adapt_queue()
    render_queue(now_playing=now_playing_data, queue=queue_data)

with tab_trace:
    current_name = st.session_state.current_playback.get("track_name", "current track")
    render_agent_trace(
        trace=bridge.adapt_trace(),
        current_track=current_name,
    )
