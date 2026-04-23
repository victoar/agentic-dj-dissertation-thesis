import os
import pylast
from dotenv import load_dotenv

load_dotenv()

print("=" * 45)
print("TEST 2 — Last.fm API")
print("=" * 45)

network = pylast.LastFMNetwork(
    api_key=os.getenv("LASTFM_API_KEY")
)

# 1. Track tags — this is our replacement for audio features
track = network.get_track("M83", "Midnight City")
tags  = track.get_top_tags(limit=8)

print(f"\n[1] Track tags   ✓  M83 — Midnight City")
print("      Tags:", ", ".join(t.item.name for t in tags))

# 2. Similar tracks — this is our replacement for recommendations
similar = track.get_similar(limit=5)
print(f"\n[2] Similar      ✓  Top 5 similar tracks:")
for s in similar:
    print(f"      {s.item.title} — {s.item.artist.name}")

# 3. Artist tags — useful for genre and mood context
artist = network.get_artist("M83")
artist_tags = artist.get_top_tags(limit=5)
print(f"\n[3] Artist tags  ✓  M83 tags:")
print("      Tags:", ", ".join(t.item.name for t in artist_tags))

# 4. User top tracks — personalisation signal
user = network.get_user(os.getenv("LASTFM_USERNAME", ""))
if os.getenv("LASTFM_USERNAME"):
    top = user.get_top_tracks(limit=3, period=pylast.PERIOD_1MONTH)
    print(f"\n[4] User top     ✓  Your top tracks this month:")
    for t in top:
        print(f"      {t.item.title} — {t.item.artist.name}")
else:
    print("\n[4] User top     —  Add LASTFM_USERNAME to .env to enable")

print("\n✓ Test 2 passed — Last.fm API working\n")