import streamlit as st

MOCK_NOW_PLAYING = {
    "name":   "Midnight City",
    "artist": "M83",
    "bpm":    105,
    "key":    "A♭ maj",
}

MOCK_QUEUE = [
    {
        "name":   "Pumped Up Kicks",
        "artist": "Foster the People",
        "bpm":    111,
        "key":    "F maj",
        "energy": 0.74,
        "note":   "Compatible key (F→A♭ +3 semitones) · tempo close · maintaining build arc",
    },
    {
        "name":   "Time",
        "artist": "Pink Floyd",
        "bpm":    123,
        "key":    "D maj",
        "energy": 0.85,
        "note":   "Tempo step up toward arc peak · high energy valence matches state",
    },
    {
        "name":   "Mr. Brightside",
        "artist": "The Killers",
        "bpm":    148,
        "key":    "G maj",
        "energy": 0.91,
        "note":   "Arc peak approaching · familiar track (openness allows) · G compatible with D",
    },
]

_BADGE_BASE = (
    "display:inline-block;"
    "border-radius:20px;"
    "padding:2px 10px;"
    "font-size:0.78rem;"
    "font-weight:500;"
    "white-space:nowrap;"
)
_BADGE_PLAYING = _BADGE_BASE + "background:#eeedf9;color:#5c5bd6;"
_BADGE_AGENT   = _BADGE_BASE + "background:#eeedf9;color:#5c5bd6;"

_META_STYLE = "font-size:0.78rem;color:#888;"
_NOTE_STYLE = "font-size:0.78rem;color:#aaa;font-style:italic;margin-top:2px;"


def _now_playing_row(track: dict) -> str:
    return f"""
    <div style="display:flex;align-items:center;gap:12px;padding:12px 4px;">
      <div style="font-size:1.1rem;color:#555;min-width:20px;">▶</div>
      <div style="
          background:#ede8f5;border-radius:8px;
          width:38px;height:38px;
          display:flex;align-items:center;justify-content:center;
          font-size:1.1rem;flex-shrink:0;
      ">🎵</div>
      <div style="flex:1;min-width:0;">
        <div style="font-size:0.95rem;font-weight:600;color:#222;">
          {track['name']} — {track['artist']}
        </div>
        <div style="{_META_STYLE}">
          Now playing &middot; {track['bpm']} BPM &middot; {track['key']}
        </div>
      </div>
      <span style="{_BADGE_PLAYING}">Playing</span>
    </div>
    """


def _queue_row(idx: int, track: dict) -> str:
    return f"""
    <div style="display:flex;align-items:flex-start;gap:12px;padding:12px 4px;">
      <div style="font-size:0.85rem;color:#aaa;min-width:20px;padding-top:2px;text-align:center;">
        {idx}
      </div>
      <div style="
          background:#f0f0eb;border-radius:8px;
          width:38px;height:38px;
          display:flex;align-items:center;justify-content:center;
          font-size:1.1rem;flex-shrink:0;
      ">🎵</div>
      <div style="flex:1;min-width:0;">
        <div style="font-size:0.95rem;font-weight:600;color:#222;">
          {track['name']} — {track['artist']}
        </div>
        <div style="{_META_STYLE}">
          {track['bpm']} BPM &middot; {track['key']} &middot; Energy {track['energy']:.2f}
        </div>
        <div style="{_NOTE_STYLE}">{track['note']}</div>
      </div>
      <span style="{_BADGE_AGENT}">Agent</span>
    </div>
    """


def _divider() -> str:
    return "<hr style='border:none;border-top:1px solid #e8e8e3;margin:0 4px;'>"


def render(now_playing: dict | None = None, queue: list | None = None):
    """
    Render the Queue tab.
    now_playing: dict with name/artist/bpm/key or None for mock data.
    queue: list of enriched track dicts or None for mock data.
    """
    current = now_playing or MOCK_NOW_PLAYING
    items   = queue       or MOCK_QUEUE

    with st.container(border=True):
        st.markdown(
            "<p style='font-size:0.9rem;color:#444;margin-bottom:0;'>"
            "Upcoming queue — agent-managed</p>",
            unsafe_allow_html=True,
        )

        rows_html = _now_playing_row(current)
        for i, track in enumerate(items, start=1):
            rows_html += _divider() + _queue_row(i, track)

        st.markdown(rows_html, unsafe_allow_html=True)
