import os
import spotipy
from spotipy.oauth2 import SpotifyPKCE
from dotenv import load_dotenv

load_dotenv()

print("=" * 45)
print("TEST 1 — Spotify API")
print("=" * 45)

sp = spotipy.Spotify(auth_manager=SpotifyPKCE(
    client_id=os.getenv("SPOTIPY_CLIENT_ID"),
    redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
    scope=(
        "user-read-playback-state "
        "user-modify-playback-state "
        "user-read-currently-playing"
    )
))

# 1. Auth
me = sp.current_user()
print(f"\n[1] Auth         ✓  {me['display_name']} ({me['id']})")

# 2. Search
results = sp.search(q="Midnight City M83", limit=3, type="track")
tracks = results["tracks"]["items"]
print(f"[2] Search       ✓  Found {len(tracks)} tracks")
print(tracks)
for t in tracks:
    print(f"      {t['name']} — {t['artists'][0]['name']}")

# 3. Playback state
playback = sp.current_playback()
if playback and playback.get("item"):
    t = playback["item"]
    print(f"[3] Playback     ✓  {t['name']} — {t['artists'][0]['name']}")
else:
    print("[3] Playback     —  No active device (open Spotify and play something)")

print("\n✓ Test 1 passed — Spotify core API working\n")