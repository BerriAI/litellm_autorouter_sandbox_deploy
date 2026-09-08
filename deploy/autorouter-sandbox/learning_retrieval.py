"""
Shared retrieval ranking for the two learning experiments (memory_guardrail.py, experiment 1,
and subtask_memory_guardrail.py, experiment 5).

Lives in one place on purpose: those two experiments differ only in WHAT they key on (the
human ask vs the current subtask), so any difference in how candidates are ranked would
confound a 1-vs-5 comparison.

Ranking is top-N with a relative gap rather than a single absolute cutoff. A fixed cutoff was
the first version and it behaved badly at both ends: too high and it excluded a genuinely
related query (measured 0.29 against text-embedding-3-small, where unrelated text scored
0.06-0.15); too low and, with only a handful of learnings stored, it returned the entire store
regardless of relevance and labelled it a match.

So:
  - `noise_floor` rejects results that are not plausibly about anything (a low, cheap bar, not
    a relevance judgement)
  - after the best result, each next one is kept only while it stays within `max_gap` of the
    best. A candidate much worse than the top hit is not a second opinion, it is filler.
"""

from __future__ import annotations

import math
from typing import Final

# Below this, a result is embedding noise rather than a weak match: measured unrelated text
# against text-embedding-3-small scores 0.06-0.15, related text 0.29-0.58. This is deliberately
# set to reject only the former, so it is not doing the relevance work the gap rule does.
DEFAULT_NOISE_FLOOR: Final = 0.18

# How far below the best hit a further result may sit and still be worth injecting. Relative,
# so it does not assume a particular score range the way a fixed cutoff does.
DEFAULT_MAX_GAP: Final = 0.12


def cosine(a: list[float], b: list[float]) -> float:
    dot: Final = sum(x * y for x, y in zip(a, b))
    na: Final = math.sqrt(sum(x * x for x in a))
    nb: Final = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def rank_candidates(
    scored: list[tuple[float, str]],
    limit: int,
    noise_floor: float = DEFAULT_NOISE_FLOOR,
    max_gap: float = DEFAULT_MAX_GAP,
) -> list[tuple[float, str]]:
    """Best first, at most `limit`, dropping noise and anything far behind the best hit.

    The best surviving result is always kept even if it is weak, since "the most related thing
    I know" is the question being asked; the gap rule only decides how many of its neighbours
    come with it.
    """
    above_noise: Final = sorted(
        (pair for pair in scored if pair[0] >= noise_floor),
        key=lambda pair: pair[0],
        reverse=True,
    )
    if not above_noise:
        return []
    best: Final = above_noise[0][0]
    return [pair for pair in above_noise[:limit] if best - pair[0] <= max_gap]
