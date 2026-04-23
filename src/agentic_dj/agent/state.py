from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime


class ArcPhase(Enum):
    WARMUP   = "warmup"
    BUILD    = "build"
    PEAK     = "peak"
    COOLDOWN = "cooldown"


class FeedbackEvent(Enum):
    SKIP            = "skip"
    EARLY_SKIP      = "early_skip"      # skipped before 30% of track
    REPLAY          = "replay"
    FULL_LISTEN     = "full_listen"     # listened to >85% of track
    PARTIAL_LISTEN  = "partial_listen"  # listened to 30–85%
    THUMBS_UP       = "thumbs_up"
    THUMBS_DOWN     = "thumbs_down"


@dataclass
class Track:
    """Minimal track representation with the features the state model needs."""
    id:           str
    name:         str
    artist:       str
    tags:         list[str]   = field(default_factory=list)   # from Last.fm
    bpm:          float       = 120.0
    energy_est:   float       = 0.5    # estimated 0-1 from tags
    valence_est:  float       = 0.5    # estimated 0-1 from tags
    familiar:     bool        = False  # has user heard this artist before?


@dataclass
class ListenerState:
    """
    Six-dimensional dynamic representation of the listener's current state.
    All dimensions are floats in [0.0, 1.0].
    """
    energy:    float = 0.5   # physical arousal — high = fast/intense, low = calm
    valence:   float = 0.5   # emotional tone  — high = happy, low = melancholic
    focus:     float = 0.5   # engagement      — high = background, low = active listening
    openness:  float = 0.7   # novelty welcome — high = try new things, low = familiar only
    social:    float = 0.3   # context         — high = group/party, low = solo
    arc_phase: ArcPhase = ArcPhase.WARMUP

    # Internal tracking — not exposed to the agent directly
    _tracks_played:   int   = field(default=0,   repr=False)
    _session_minutes: float = field(default=0.0, repr=False)
    _history:         list  = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        return {
            "energy":    round(self.energy,   2),
            "valence":   round(self.valence,  2),
            "focus":     round(self.focus,    2),
            "openness":  round(self.openness, 2),
            "social":    round(self.social,   2),
            "arc_phase": self.arc_phase.value,
            "tracks_played": self._tracks_played,
        }

    def summary(self) -> str:
        return (
            f"energy={self.energy:.2f}  valence={self.valence:.2f}  "
            f"focus={self.focus:.2f}  openness={self.openness:.2f}  "
            f"social={self.social:.2f}  arc={self.arc_phase.value}"
        )


def _clamp(value: float) -> float:
    """Keep all state values strictly within [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


# ── Update weights ────────────────────────────────────────────────────────
# Each weight is how much a feedback event shifts a dimension.
# Negative = pushes dimension down. Positive = pushes dimension up.
# Signs are relative to the track's own energy/valence profile.

WEIGHTS = {
    FeedbackEvent.EARLY_SKIP: {
        "energy":   -0.15,   # listener did not want this energy level
        "valence":  -0.08,   # rejected the mood
        "openness": -0.10,   # unfamiliar track rejected → close off novelty
        "focus":    +0.05,   # skipping = actively engaged, not passive
    },
    FeedbackEvent.SKIP: {
        "energy":   -0.10,
        "valence":  -0.05,
        "openness": -0.06,
        "focus":    +0.03,
    },
    FeedbackEvent.PARTIAL_LISTEN: {
        "energy":   +0.02,   # mild positive signal
        "valence":  +0.02,
        "focus":    -0.03,   # partial = drifting attention
    },
    FeedbackEvent.FULL_LISTEN: {
        "energy":   +0.06,   # this energy level was right
        "valence":  +0.04,
        "openness": +0.04,   # completed unfamiliar track → open to more
        "focus":    -0.04,
    },
    FeedbackEvent.REPLAY: {
        "energy":   +0.03,
        "valence":  +0.08,   # strong positive mood signal
        "openness": -0.05,   # replaying = wants familiar, not new
        "focus":    +0.06,   # active choice to replay = engaged
    },
    FeedbackEvent.THUMBS_UP: {
        "energy":   +0.08,
        "valence":  +0.10,
        "openness": +0.06,
        "focus":    +0.04,
    },
    FeedbackEvent.THUMBS_DOWN: {
        "energy":   -0.12,
        "valence":  -0.10,
        "openness": -0.08,
        "focus":    +0.02,
    },
}

# Arc phase thresholds — how many tracks trigger each phase transition
ARC_TRANSITIONS = {
    ArcPhase.WARMUP:   4,    # after 4 tracks → BUILD
    ArcPhase.BUILD:    8,    # after 8 more   → PEAK
    ArcPhase.PEAK:     4,    # after 4 more   → COOLDOWN
    ArcPhase.COOLDOWN: 9999, # stays here until session ends
}


def update_state(
    state:    ListenerState,
    event:    FeedbackEvent,
    track:    Track,
) -> ListenerState:
    """
    Apply a feedback event to the listener state and return the updated state.
    The direction of energy/valence updates depends on the track's own profile —
    a skip on a HIGH-energy track means energy should go DOWN.
    A full listen on a HIGH-valence track means valence should go UP.
    """
    weights = WEIGHTS.get(event, {})

    # Energy update — sign depends on whether the track was high or low energy
    if "energy" in weights:
        direction = 1 if track.energy_est >= 0.5 else -1
        state.energy = _clamp(state.energy + weights["energy"] * direction)

    # Valence update — sign depends on track's valence profile
    if "valence" in weights:
        direction = 1 if track.valence_est >= 0.5 else -1
        state.valence = _clamp(state.valence + weights["valence"] * direction)

    # Openness and focus updates are direction-independent
    if "openness" in weights:
        state.openness = _clamp(state.openness + weights["openness"])

    if "focus" in weights:
        state.focus = _clamp(state.focus + weights["focus"])

    # Track history
    state._history.append({
        "track":  track.name,
        "artist": track.artist,
        "event":  event.value,
    })

    return state


def advance_track(state: ListenerState, track: Track) -> ListenerState:
    """
    Called when a new track starts playing.
    Increments the play counter and advances the arc phase if needed.
    """
    state._tracks_played += 1

    # Arc phase transitions
    played = state._tracks_played
    if state.arc_phase == ArcPhase.WARMUP and played >= ARC_TRANSITIONS[ArcPhase.WARMUP]:
        state.arc_phase = ArcPhase.BUILD
    elif state.arc_phase == ArcPhase.BUILD and played >= (
        ARC_TRANSITIONS[ArcPhase.WARMUP] + ARC_TRANSITIONS[ArcPhase.BUILD]
    ):
        state.arc_phase = ArcPhase.PEAK
    elif state.arc_phase == ArcPhase.PEAK and played >= (
        ARC_TRANSITIONS[ArcPhase.WARMUP] +
        ARC_TRANSITIONS[ArcPhase.BUILD] +
        ARC_TRANSITIONS[ArcPhase.PEAK]
    ):
        state.arc_phase = ArcPhase.COOLDOWN

    return state


def init_state(context: str = "general") -> ListenerState:
    """
    Initialise listener state from a declared session context.
    Different contexts start with different baseline values.
    """
    presets = {
        "study":   ListenerState(energy=0.3, valence=0.5, focus=0.8, openness=0.5, social=0.1),
        "workout": ListenerState(energy=0.7, valence=0.7, focus=0.3, openness=0.6, social=0.2),
        "party":   ListenerState(energy=0.8, valence=0.8, focus=0.2, openness=0.8, social=0.9),
        "focus":   ListenerState(energy=0.4, valence=0.5, focus=0.9, openness=0.4, social=0.1),
        "chill":   ListenerState(energy=0.3, valence=0.6, focus=0.5, openness=0.6, social=0.3),
        "general": ListenerState(energy=0.5, valence=0.5, focus=0.5, openness=0.7, social=0.3),
    }
    return presets.get(context, presets["general"])