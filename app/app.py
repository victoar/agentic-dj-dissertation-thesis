import streamlit as st

from components.now_playing    import render as render_now_playing
from components.listener_state import render as render_listener_state
from components.queue          import render as render_queue
from components.agent_trace    import render as render_agent_trace

st.set_page_config(
    page_title="Agentic DJ",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none; }
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    .stApp { background-color: #f5f5f0; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 800px; }
</style>
""", unsafe_allow_html=True)

st.markdown("## The practical application")

tab_now, tab_state, tab_queue, tab_trace = st.tabs([
    "Now playing", "Listener state", "Queue", "Agent trace"
])

with tab_now:
    render_now_playing()

with tab_state:
    render_listener_state()

with tab_queue:
    render_queue()

with tab_trace:
    render_agent_trace()
