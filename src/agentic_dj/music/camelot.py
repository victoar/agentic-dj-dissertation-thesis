"""
Camelot Wheel implementation for harmonic compatibility checking.

The Camelot Wheel maps all 24 musical keys onto a 12-position clock face
with two letters (A = minor, B = major). Harmonically compatible transitions
are:
    - Same position (e.g. 8B → 8B)
    - Adjacent positions, same letter (e.g. 8B → 9B or 8B → 7B)
    - Same number, different letter — relative major/minor (8B → 8A)

Spotify's audio-features endpoint returns a numeric 'key' (0-11) and 'mode'
(0 = minor, 1 = major). We provide conversion helpers for both directions.
"""

from dataclasses import dataclass


# ── Key representation ────────────────────────────────────────────────────
# Spotify key encoding: 0 = C, 1 = C#/Db, 2 = D ... 11 = B
# Mode encoding: 0 = minor, 1 = major

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Camelot Wheel mapping — (spotify_key, spotify_mode) → Camelot position
CAMELOT_FROM_SPOTIFY = {
    # Major keys (mode = 1) — B suffix on the wheel
    (0, 1):  "8B",   # C major
    (7, 1):  "9B",   # G major
    (2, 1):  "10B",  # D major
    (9, 1):  "11B",  # A major
    (4, 1):  "12B",  # E major
    (11, 1): "1B",   # B major
    (6, 1):  "2B",   # F# major
    (1, 1):  "3B",   # Db major
    (8, 1):  "4B",   # Ab major
    (3, 1):  "5B",   # Eb major
    (10, 1): "6B",   # Bb major
    (5, 1):  "7B",   # F major
    # Minor keys (mode = 0) — A suffix on the wheel
    (9, 0):  "8A",   # A minor
    (4, 0):  "9A",   # E minor
    (11, 0): "10A",  # B minor
    (6, 0):  "11A",  # F# minor
    (1, 0):  "12A",  # C# minor
    (8, 0):  "1A",   # G# minor
    (3, 0):  "2A",   # D# minor
    (10, 0): "3A",   # A# minor
    (5, 0):  "4A",   # F minor
    (0, 0):  "5A",   # C minor
    (7, 0):  "6A",   # G minor
    (2, 0):  "7A",   # D minor
}

# Reverse lookup — Camelot position → human-readable key name
CAMELOT_TO_NAME = {
    "8B": "C major",   "9B": "G major",   "10B": "D major",   "11B": "A major",
    "12B": "E major",  "1B": "B major",   "2B": "F# major",   "3B": "Db major",
    "4B": "Ab major",  "5B": "Eb major",  "6B": "Bb major",   "7B": "F major",
    "8A": "A minor",   "9A": "E minor",   "10A": "B minor",   "11A": "F# minor",
    "12A": "C# minor", "1A": "G# minor",  "2A": "D# minor",   "3A": "A# minor",
    "4A": "F minor",   "5A": "C minor",   "6A": "G minor",    "7A": "D minor",
}


@dataclass
class CamelotKey:
    """A parsed Camelot position — e.g. 8B = C major."""
    number: int   # 1-12
    letter: str   # "A" or "B"

    @property
    def position(self) -> str:
        return f"{self.number}{self.letter}"

    @property
    def key_name(self) -> str:
        return CAMELOT_TO_NAME.get(self.position, "unknown")


# ── Conversion helpers ────────────────────────────────────────────────────

def from_spotify(spotify_key: int, spotify_mode: int) -> CamelotKey | None:
    """Convert Spotify (key, mode) pair to Camelot position."""
    position = CAMELOT_FROM_SPOTIFY.get((spotify_key, spotify_mode))
    if position is None:
        return None
    return CamelotKey(number=int(position[:-1]), letter=position[-1])


def parse(camelot_string: str) -> CamelotKey | None:
    """Parse a Camelot string like '8B' or '12A' into a CamelotKey."""
    s = camelot_string.strip().upper()
    if len(s) < 2 or s[-1] not in ("A", "B"):
        return None
    try:
        number = int(s[:-1])
        if not 1 <= number <= 12:
            return None
        return CamelotKey(number=number, letter=s[-1])
    except ValueError:
        return None


# ── Compatibility logic ───────────────────────────────────────────────────

def _wheel_distance(a: int, b: int) -> int:
    """Shortest distance between two positions on the 12-position wheel."""
    diff = abs(a - b)
    return min(diff, 12 - diff)


def compatible(k1: CamelotKey, k2: CamelotKey) -> bool:
    """
    True if a transition from k1 to k2 is harmonically compatible.

    Rules (in order of strength):
      1. Same position (identical key)                          — perfect
      2. Adjacent position, same letter (e.g. 8B → 9B or 7B)    — smooth
      3. Same number, different letter (e.g. 8B → 8A)           — mode switch
    """
    if k1.position == k2.position:
        return True

    if k1.letter == k2.letter and _wheel_distance(k1.number, k2.number) == 1:
        return True

    if k1.number == k2.number and k1.letter != k2.letter:
        return True

    return False


def compatibility_strength(k1: CamelotKey, k2: CamelotKey) -> float:
    """
    Quantify how smooth a transition is, from 0.0 (jarring) to 1.0 (perfect).
    Useful for ranking multiple candidate tracks.
    """
    if k1.position == k2.position:
        return 1.0

    # Same letter (both major or both minor)
    if k1.letter == k2.letter:
        d = _wheel_distance(k1.number, k2.number)
        if d == 1: return 0.85   # adjacent
        if d == 2: return 0.55   # one step removed
        if d == 3: return 0.30   # noticeable
        return 0.10              # far apart — jarring

    # Different letter (major ↔ minor)
    if k1.number == k2.number:
        return 0.75              # relative major/minor — smooth

    # Different letter and different number — only compatible if
    # the relative key is also adjacent
    d = _wheel_distance(k1.number, k2.number)
    if d == 1: return 0.40       # adjacent relative
    return 0.05                  # essentially incompatible


def compatible_positions(k: CamelotKey) -> list[str]:
    """
    Return all Camelot positions that form a compatible transition from k.
    Used by the agent to filter candidate tracks before LLM evaluation.
    """
    compatible = [k.position]

    # Adjacent positions, same letter
    prev_num = 12 if k.number == 1  else k.number - 1
    next_num = 1  if k.number == 12 else k.number + 1
    compatible.append(f"{prev_num}{k.letter}")
    compatible.append(f"{next_num}{k.letter}")

    # Same number, different letter (relative major/minor)
    other_letter = "A" if k.letter == "B" else "B"
    compatible.append(f"{k.number}{other_letter}")

    return compatible


# ── Convenience: work directly with Spotify-format tracks ─────────────────

def tracks_compatible(
    spotify_key_1: int, spotify_mode_1: int,
    spotify_key_2: int, spotify_mode_2: int,
) -> bool:
    """Check compatibility directly from Spotify-format key/mode pairs."""
    k1 = from_spotify(spotify_key_1, spotify_mode_1)
    k2 = from_spotify(spotify_key_2, spotify_mode_2)
    if k1 is None or k2 is None:
        return False
    return compatible(k1, k2)
