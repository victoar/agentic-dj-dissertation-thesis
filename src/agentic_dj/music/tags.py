"""
Tag-to-feature translator.

Last.fm gives us human-readable tags like "energetic", "chillout", "melancholic".
The state model needs numerical estimates for energy and valence (0.0 to 1.0).
This module translates between them.

The mapping is built from a curated lexicon of tags, each annotated with its
energy and valence contribution. When a track has multiple tags, we aggregate
them with popularity weighting (more popular tags count more).
"""

from dataclasses import dataclass


# ── Tag lexicon ────────────────────────────────────────────────────────────
# Each tag maps to (energy_contribution, valence_contribution) on a -1..+1 scale.
# Values are aggregated and then shifted to the 0..1 scale for the state model.
#
# Principles:
#   - Tempo/intensity words push energy
#   - Mood/emotion words push valence
#   - Some tags carry both (e.g. "melancholic" = low energy + low valence)
#   - Genre tags carry weak signals only — they are ambiguous
#
# This lexicon will be expanded during development. It is OK that it's
# incomplete — unknown tags simply contribute nothing to the estimate.

TAG_LEXICON: dict[str, tuple[float, float]] = {
    # ── Strong energy signals ─────────────────────────────────────────────
    "energetic":       (+0.8, +0.3),
    "high energy":     (+0.9, +0.2),
    "intense":         (+0.7, -0.1),
    "powerful":        (+0.7, +0.2),
    "aggressive":      (+0.8, -0.3),
    "hard":            (+0.6, -0.1),
    "heavy":           (+0.6, -0.2),
    "loud":            (+0.6,  0.0),
    "fast":            (+0.7,  0.0),
    "upbeat":          (+0.6, +0.6),
    "dance":           (+0.7, +0.5),
    "danceable":       (+0.7, +0.5),
    "party":           (+0.8, +0.7),
    "banger":          (+0.9, +0.5),
    "anthem":          (+0.8, +0.6),
    "pumping":         (+0.8, +0.3),
    "driving":         (+0.7,  0.0),
    "workout":         (+0.8, +0.3),
    "running":         (+0.7, +0.3),

    # ── Strong low-energy signals ─────────────────────────────────────────
    "calm":            (-0.6, +0.2),
    "chill":           (-0.5, +0.3),
    "chillout":        (-0.6, +0.3),
    "mellow":          (-0.6, +0.2),
    "relaxing":        (-0.7, +0.3),
    "peaceful":        (-0.7, +0.4),
    "soft":            (-0.5, +0.1),
    "quiet":           (-0.7,  0.0),
    "slow":            (-0.6,  0.0),
    "downtempo":       (-0.6, -0.1),
    "ambient":         (-0.7,  0.0),
    "sleep":           (-0.9, -0.1),
    "lullaby":         (-0.9, +0.2),
    "meditation":      (-0.8, +0.2),
    "acoustic":        (-0.3, +0.1),
    "minimalist":      (-0.4,  0.0),

    # ── Positive valence signals ──────────────────────────────────────────
    "happy":           (+0.2, +0.9),
    "joyful":          (+0.3, +0.9),
    "uplifting":       (+0.4, +0.8),
    "feel good":       (+0.3, +0.8),
    "cheerful":        (+0.3, +0.8),
    "sunny":           (+0.2, +0.7),
    "bright":          (+0.2, +0.6),
    "fun":             (+0.4, +0.7),
    "playful":         (+0.2, +0.6),
    "optimistic":      (+0.1, +0.7),
    "summer":          (+0.4, +0.7),

    # ── Negative valence signals ──────────────────────────────────────────
    "sad":             (-0.3, -0.8),
    "melancholic":     (-0.4, -0.7),
    "melancholy":      (-0.4, -0.7),
    "dark":            (-0.1, -0.6),
    "depressing":      (-0.5, -0.9),
    "moody":           (-0.2, -0.5),
    "sombre":          (-0.3, -0.6),
    "somber":          (-0.3, -0.6),
    "nostalgic":       (-0.2, -0.3),
    "bittersweet":     (-0.2, -0.3),
    "lonely":          (-0.3, -0.6),
    "haunting":        (-0.1, -0.5),
    "emotional":       (-0.1, -0.3),
    "atmospheric":     (-0.2, -0.1),
    "brooding":        (-0.1, -0.5),
    "gloomy":          (-0.3, -0.7),
    "angst":           (+0.1, -0.6),
    "angry":           (+0.5, -0.7),

    # ── Focus & context tags ──────────────────────────────────────────────
    "focus":           (-0.2,  0.0),
    "concentration":   (-0.3,  0.0),
    "study":           (-0.3, +0.1),
    "background":      (-0.2,  0.0),
    "late night":      (-0.3, -0.1),
    "night":           (-0.2, -0.1),
    "morning":         (+0.2, +0.3),

    # ── Genre tags (weak signals — genres span moods) ─────────────────────
    "rock":            (+0.4,  0.0),
    "metal":           (+0.7, -0.2),
    "punk":            (+0.7, -0.1),
    "hardcore":        (+0.8, -0.2),
    "electronic":      (+0.3,  0.0),
    "techno":          (+0.6,  0.0),
    "house":           (+0.5, +0.3),
    "edm":             (+0.6, +0.4),
    "trance":          (+0.5, +0.3),
    "dubstep":         (+0.7, -0.1),
    "drum and bass":   (+0.8,  0.0),
    "pop":             (+0.3, +0.4),
    "indie":           (+0.1, +0.1),
    "indie pop":       (+0.3, +0.4),
    "indie rock":      (+0.4, +0.1),
    "folk":            (-0.3, +0.1),
    "jazz":            (-0.1, +0.1),
    "blues":           (-0.1, -0.3),
    "classical":       (-0.2,  0.0),
    "hip hop":         (+0.3,  0.0),
    "hip-hop":         (+0.3,  0.0),
    "rap":             (+0.4,  0.0),
    "r&b":             (+0.1, +0.1),
    "soul":            (+0.1, +0.3),
    "funk":            (+0.5, +0.5),
    "disco":           (+0.6, +0.7),
    "reggae":          (-0.1, +0.4),
    "country":         (-0.1, +0.2),
    "synthpop":        (+0.4, +0.3),
    "synthwave":       (+0.3, +0.1),
    "shoegaze":        (-0.2, -0.1),
    "post-rock":       (-0.1,  0.0),
    "dream pop":       (-0.2, +0.2),
    "lo-fi":           (-0.4, +0.1),
    "lofi":            (-0.4, +0.1),
    "idm":             (+0.2, -0.1),
    "trap":            (+0.5,  0.0),
    "soundtrack":      (-0.1,  0.0),
    "instrumental":    (-0.2,  0.0),
}


@dataclass
class TagEstimate:
    """Result of estimating features from a list of tags."""
    energy:       float        # final estimate, 0.0 to 1.0
    valence:      float        # final estimate, 0.0 to 1.0
    matched_tags: list[str]    # which tags contributed to the estimate
    confidence:   float        # 0.0 to 1.0 — higher when more tags matched


def _normalise(tag: str) -> str:
    """Canonical form for tag lookup — lowercase, no separators, trimmed."""
    # Lowercase, collapse any separator (space, dash, underscore) to nothing,
    # then try both the joined and spaced forms when looking up.
    return tag.strip().lower().replace("_", " ").replace("-", " ")


def estimate_features(
    tags: list[str],
    tag_weights: list[float] | None = None,
) -> TagEstimate:
    """
    Estimate energy and valence for a track given its Last.fm tags.

    Args:
        tags: list of tag strings from Last.fm
        tag_weights: optional list of relative weights (e.g. Last.fm popularity
                     counts). If provided, must match the length of tags.

    Returns:
        TagEstimate with energy and valence in [0.0, 1.0], plus diagnostics.
    """
    if not tags:
        return TagEstimate(energy=0.5, valence=0.5, matched_tags=[], confidence=0.0)

    if tag_weights is None:
        tag_weights = [1.0] * len(tags)
    elif len(tag_weights) != len(tags):
        raise ValueError("tag_weights length must match tags length")

    energy_sum:  float = 0.0
    valence_sum: float = 0.0
    weight_sum:  float = 0.0
    matched:     list[str] = []

    for tag, weight in zip(tags, tag_weights):
        normalised = _normalise(tag)
        # Try exact match first, then try collapsing internal whitespace
        # (because the lexicon has both "hip hop" and "chillout" styles)
        lookup = normalised if normalised in TAG_LEXICON else normalised.replace(" ", "")
        if lookup in TAG_LEXICON:
            e_delta, v_delta = TAG_LEXICON[lookup]
            energy_sum  += e_delta * weight
            valence_sum += v_delta * weight
            weight_sum  += weight
            matched.append(lookup)

    # If nothing matched, return neutral with zero confidence
    if weight_sum == 0:
        return TagEstimate(energy=0.5, valence=0.5, matched_tags=[], confidence=0.0)

    # Weighted average of contributions, still in -1..+1 range
    avg_energy  = energy_sum  / weight_sum
    avg_valence = valence_sum / weight_sum

    # Shift from -1..+1 to 0..1 (neutral = 0.5)
    energy  = 0.5 + 0.5 * max(-1.0, min(1.0, avg_energy))
    valence = 0.5 + 0.5 * max(-1.0, min(1.0, avg_valence))

    # Confidence scales with how many tags matched, capped at 5 matches
    confidence = min(1.0, len(matched) / 5.0)

    return TagEstimate(
        energy       = round(energy,  3),
        valence      = round(valence, 3),
        matched_tags = matched,
        confidence   = round(confidence, 2),
    )


# ── Semantic fallback for unknown tags ─────────────────────────────────────
# Lazy-loaded to avoid importing torch unless needed.

_embedding_model = None
_lexicon_embeddings = None
_lexicon_keys = None


def _get_embedding_model():
    """Load the sentence transformer lazily. Small model, fast on CPU."""
    global _embedding_model, _lexicon_embeddings, _lexicon_keys

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print("  [tags] Loading sentence-transformer model (one-time)...")
        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        _lexicon_keys = list(TAG_LEXICON.keys())
        _lexicon_embeddings = _embedding_model.encode(
            _lexicon_keys,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

    return _embedding_model, _lexicon_embeddings, _lexicon_keys


def _find_nearest_lexicon_tags(
    unknown_tag: str,
    top_k: int = 3,
    min_similarity: float = 0.45,
) -> list[tuple[str, float]]:
    """
    Given an unknown tag, find the top_k most semantically similar lexicon
    entries. Returns a list of (lexicon_tag, similarity) pairs, filtered by
    min_similarity. If no matches pass the threshold, returns an empty list.
    """
    import numpy as np

    model, lex_emb, lex_keys = _get_embedding_model()
    query_emb = model.encode(
        [unknown_tag],
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )[0]

    similarities = lex_emb @ query_emb  # dot product, both already normalised
    top_indices  = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        sim = float(similarities[idx])
        if sim >= min_similarity:
            results.append((lex_keys[idx], sim))

    return results


def estimate_features_with_fallback(
    tags:         list[str],
    tag_weights:  list[float] | None = None,
    use_semantic: bool = True,
    popularity:   int | None = None,
) -> TagEstimate:
    """
    Robust version of estimate_features that handles unknown tags.

    Strategy:
      1. Match tags against the lexicon directly (primary path).
      2. For unmatched tags, find semantically similar lexicon entries
         and use their feature values, weighted by similarity.
      3. If everything fails, optionally use popularity as a weak valence signal.
    """
    if tag_weights is None:
        tag_weights = [1.0] * len(tags)

    energy_sum:  float = 0.0
    valence_sum: float = 0.0
    weight_sum:  float = 0.0
    matched:     list[str] = []
    inferred:    list[str] = []  # track which tags used the semantic fallback

    for tag, weight in zip(tags, tag_weights):
        normalised = _normalise(tag)

        # Primary: direct lexicon match
        if normalised in TAG_LEXICON:
            e, v = TAG_LEXICON[normalised]
            energy_sum  += e * weight
            valence_sum += v * weight
            weight_sum  += weight
            matched.append(normalised)
            continue

        # Fallback: semantic similarity
        if use_semantic:
            neighbours = _find_nearest_lexicon_tags(normalised, top_k=2, min_similarity=0.45)
            if neighbours:
                # Weighted average of neighbour values, attenuated by similarity
                e_est = sum(TAG_LEXICON[n][0] * s for n, s in neighbours) / sum(s for _, s in neighbours)
                v_est = sum(TAG_LEXICON[n][1] * s for n, s in neighbours) / sum(s for _, s in neighbours)
                # Dampen the contribution — inferred values are less reliable
                attenuation = 0.6
                energy_sum  += e_est * weight * attenuation
                valence_sum += v_est * weight * attenuation
                weight_sum  += weight * attenuation
                inferred.append(f"{normalised}~{neighbours[0][0]}")

    # Compute the estimate
    if weight_sum > 0:
        avg_e = energy_sum  / weight_sum
        avg_v = valence_sum / weight_sum
        energy  = 0.5 + 0.5 * max(-1.0, min(1.0, avg_e))
        valence = 0.5 + 0.5 * max(-1.0, min(1.0, avg_v))
    else:
        # Everything failed — fall back to popularity if available
        energy  = 0.5
        if popularity is not None:
            # High popularity → slight positive valence bias
            valence = 0.5 + (popularity - 50) / 200  # maps 0..100 → 0.25..0.75
            valence = max(0.3, min(0.7, valence))
        else:
            valence = 0.5

    # Confidence reflects both matched and inferred contributions
    effective_matches = len(matched) + len(inferred) * 0.6
    confidence = min(1.0, effective_matches / 5.0)

    return TagEstimate(
        energy       = round(energy,  3),
        valence      = round(valence, 3),
        matched_tags = matched + inferred,
        confidence   = round(confidence, 2),
    )


def estimate_from_lastfm_top_tags(pylast_top_tags) -> TagEstimate:
    """
    Convenience wrapper for the shape pylast returns.
    pylast's track.get_top_tags() returns a list of TopItem objects
    where .item.name is the tag name and .weight is the popularity count.
    """
    tags    = [t.item.name for t in pylast_top_tags]
    weights = [float(t.weight) if hasattr(t, "weight") and t.weight else 1.0
               for t in pylast_top_tags]
    return estimate_features(tags, weights)
