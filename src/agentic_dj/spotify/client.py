"""
Spotify client — wraps Spotipy into clean methods the agent can call.

Responsibilities:
  - Search for candidate tracks by query string
  - Get track metadata (name, artist, popularity, duration)
  - Control playback (play, pause, skip, add to queue)
  - Read current playback state
  - Manage the session queue

Note: audio-features and recommendations endpoints are unavailable for
new apps (deprecated Nov 2024). Candidate discovery uses search + Last.fm
enrichment instead.
"""

import os
from dataclasses import dataclass
from typing import Optional

import spotipy
from spotipy.oauth2 import SpotifyPKCE
from dotenv import load_dotenv

load_dotenv()

# ── Scopes we need ───────────────────────────────────────────────────────────
# user-read-playback-state  → read current track + device
# user-modify-playback-state → play, pause, skip, queue
# user-read-currently-playing → currently playing track
# user-library-read           → user's saved tracks (familiarity signal)
SCOPES = " ".join([
    "user-read-playback-state",
    "user-modify-playback-state",
    "user-read-currently-playing",
    "user-library-read",
    "user-library-modify",
    "user-top-read",
])


@dataclass
class SpotifyTrack:
    """
    Raw Spotify track data — before Last.fm enrichment.
    Returned by search and playback queries.
    """
    id:          str
    name:        str
    artist:      str
    album:       str
    duration_ms: int
    uri:         str           # spotify:track:xxx — needed for playback


@dataclass
class PlaybackState:
    """Current playback snapshot."""
    is_playing:   bool
    track:        Optional[SpotifyTrack]
    progress_ms:  int           # how far through the current track
    device_name:  str
    device_id:    str


# ── Client ───────────────────────────────────────────────────────────────────

class SpotifyClient:

    def __init__(self):
        self._sp: Optional[spotipy.Spotify] = None

    def _get_sp(self) -> spotipy.Spotify:
        """Lazy-init the authenticated Spotipy client."""
        if self._sp is None:
            self._sp = spotipy.Spotify(auth_manager=SpotifyPKCE(
                client_id=os.getenv("SPOTIPY_CLIENT_ID"),
                redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
                scope=SCOPES,
            ))
        return self._sp

    # ── Search ───────────────────────────────────────────────────────────

    def search(
        self,
        query:  str,
        limit:  int = 10,
    ) -> list[SpotifyTrack]:
        """
        Search the Spotify catalogue.
        Query can be anything: track name, artist, genre keywords.
        Returns up to `limit` results sorted by Spotify's relevance.
        """
        sp = self._get_sp()
        results = sp.search(q=query, limit=limit, type="track")
        print(query)
        tracks  = results.get("tracks", {}).get("items", [])
        return [self._parse_track(t) for t in tracks if t]

    def search_by_mood(
        self,
        mood_tags:  list[str],
        limit:      int = 10,
    ) -> list[SpotifyTrack]:
        """
        Search using mood/genre tags as the query string.
        Joins tags into a natural language query Spotify understands well.
        Example: ["energetic", "indie rock"] → "energetic indie rock"
        """
        query = " ".join(mood_tags[:4])   # Spotify handles ~4 terms well
        return self.search(query, limit)

    def search_similar_to(
        self,
        artist:    str,
        track:     str,
        limit:     int = 10,
    ) -> list[SpotifyTrack]:
        """
        Find tracks stylistically similar to a given artist/track.
        Uses Spotify's search with artist name as the seed — simple and reliable.
        """
        query = f"artist:{artist}"
        return self.search(query, limit)

    # ── Playback state ───────────────────────────────────────────────────

    def get_playback(self) -> Optional[PlaybackState]:
        """
        Return the current playback state, or None if nothing is playing.
        Requires an active Spotify device (phone, desktop app, web player).
        """
        sp = self._get_sp()
        state = sp.current_playback()

        if not state:
            return None

        item = state.get("item")
        device = state.get("device", {})

        track = self._parse_track(item) if item else None

        return PlaybackState(
            is_playing=state.get("is_playing", False),
            track=track,
            progress_ms=state.get("progress_ms", 0),
            device_name=device.get("name", "unknown"),
            device_id=device.get("id", ""),
        )

    def get_current_track(self) -> Optional[SpotifyTrack]:
        """Convenience wrapper — just the current track, no extra state."""
        state = self.get_playback()
        return state.track if state else None

    # ── Playback control ─────────────────────────────────────────────────

    def play(self, track: SpotifyTrack, device_id: str = "") -> bool:
        """
        Start playing a specific track.
        Returns True on success, False if no active device found.
        """
        sp = self._get_sp()
        try:
            kwargs = {"uris": [track.uri]}
            if device_id:
                kwargs["device_id"] = device_id
            sp.start_playback(**kwargs)
            return True
        except spotipy.SpotifyException as e:
            if "NO_ACTIVE_DEVICE" in str(e):
                return False
            raise

    def add_to_queue(self, track: SpotifyTrack, device_id: str = "") -> bool:
        """
        Add a track to the end of the playback queue.
        This is the primary way the agent schedules upcoming tracks.
        """
        sp = self._get_sp()
        try:
            kwargs = {"uri": track.uri}
            if device_id:
                kwargs["device_id"] = device_id
            sp.add_to_queue(**kwargs)
            return True
        except spotipy.SpotifyException as e:
            if "NO_ACTIVE_DEVICE" in str(e):
                return False
            raise

    def skip(self) -> bool:
        """Skip to the next track. Returns True on success."""
        sp = self._get_sp()
        try:
            sp.next_track()
            return True
        except spotipy.SpotifyException:
            return False

    def pause(self) -> bool:
        """Pause playback. Returns True on success."""
        sp = self._get_sp()
        try:
            sp.pause_playback()
            return True
        except spotipy.SpotifyException:
            return False

    def resume(self) -> bool:
        """Resume playback. Returns True on success."""
        sp = self._get_sp()
        try:
            sp.start_playback()
            return True
        except spotipy.SpotifyException:
            return False

    def seek_to_beginning(self) -> bool:
        """Restart the current track from position 0. Returns True on success."""
        sp = self._get_sp()
        try:
            sp.seek_track(0)
            return True
        except spotipy.SpotifyException:
            return False

    def save_track(self, track_id: str) -> bool:
        """Save a track to the user's Spotify library. Returns True on success."""
        sp = self._get_sp()
        try:
            sp.current_user_saved_tracks_add([track_id])
            return True
        except spotipy.SpotifyException:
            return False

    def get_spotify_queue(self) -> list[str]:
        """Return track IDs currently in Spotify's actual queue (excludes currently playing)."""
        sp = self._get_sp()
        try:
            data = sp.queue()
            return [t["id"] for t in data.get("queue", []) if t.get("id")]
        except spotipy.SpotifyException:
            return []

    # ── User library ─────────────────────────────────────────────────────

    def is_saved(self, track_id: str) -> bool:
        """
        Check if a track is in the user's saved library.
        Used to set the `familiar` flag on Track objects — saved tracks
        are assumed to be familiar to the listener.
        """
        sp = self._get_sp()
        try:
            result = sp.current_user_saved_tracks_contains([track_id])
            return result[0] if result else False
        except spotipy.SpotifyException:
            return False

    def get_top_tracks(self, limit: int = 20, time_range: str = "short_term") -> list[SpotifyTrack]:
        """Fetch the user's most-played tracks. time_range: short_term (4w), medium_term (6m), long_term (all time)."""
        sp = self._get_sp()
        try:
            results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
            return [self._parse_track(t) for t in results.get("items", []) if t]
        except spotipy.SpotifyException:
            return []

    def get_saved_tracks(self, limit: int = 50) -> list[SpotifyTrack]:
        """
        Fetch the user's most recently saved tracks.
        Useful for seeding the familiarity signal at session start.
        """
        sp = self._get_sp()
        results = sp.current_user_saved_tracks(limit=limit)
        items   = results.get("items", [])
        return [self._parse_track(item["track"])
                for item in items if item.get("track")]
    
    def get_audio_analysis(self, track_id):
        sp = self._get_sp()
        result = sp.audio_analysis(track_id)
        print(result)
        return result

    # ── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _parse_track(raw: dict) -> SpotifyTrack:
        """Convert a raw Spotify API track dict into a SpotifyTrack."""
        artists = raw.get("artists", [{}])
        return SpotifyTrack(
            id=raw.get("id", ""),
            name=raw.get("name", "unknown"),
            artist=artists[0].get("name", "unknown") if artists else "unknown",
            album=raw.get("album", {}).get("name", ""),
            duration_ms=raw.get("duration_ms", 0),
            uri=raw.get("uri", ""),
        )