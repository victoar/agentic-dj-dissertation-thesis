---
name: AgDJ — Known limitations and constraints
description: Real API/platform constraints that affect design decisions, so suggestions don't recommend unavailable features
type: project
---

These are acknowledged limitations documented in the thesis. Suggestions should not recommend workarounds that violate these constraints.

1. **Spotify audio-features + recommendations = 403/404** for apps registered after Nov 2024. Replaced entirely by Last.fm tags + search-based candidate discovery.
2. **Spotify popularity field = 0** in Development Mode. Replaced by Last.fm listener count normalised to 0-100.
3. **Spotify Premium required** for playback control endpoints (play, queue, skip).
4. **Gemini free tier: 5 req/min**. This is why a true multi-step ReAct loop was dropped in favour of single-shot (2 API calls/cycle). Retry uses exponential backoff: 12s, 24s, 48s, 96s on 429.
5. **BPM data unavailable** — Spotify audio-features (which had tempo) is deprecated. BPM compatibility checks fall back to `bpm_ok=True`. Future work to source from Last.fm or third-party.
6. **Tag lexicon**: 110 entries covers mainstream well; niche genres (vaporwave, hyperpop, witch house) fall back to sentence-transformer semantic similarity (cosine threshold 0.45, contribution attenuated 0.6x).
7. **State model weights are hand-tuned**, not learned. Calibration experiment planned for Week 6.
8. **Single-user, single-threaded**: module-level state in tools.py — not suitable for multi-user deployment without class-based session refactor.
