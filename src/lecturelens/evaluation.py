"""Retrieval metrics.

For each test question we know the time range(s) where the answer is spoken.
A retrieved chunk is relevant if its time window overlaps one of those ranges.

  * hit rate@k: share of questions with at least one relevant chunk in the top k
  * MRR@k (mean reciprocal rank): average of 1/rank of the first relevant chunk
    (1.0 if it is first, 0.5 if second, ... 0 if not in the top k), so it
    rewards putting the right moment at the top.
"""

from collections.abc import Sequence

from lecturelens.models import Chunk

TimeRange = tuple[float, float]


def is_relevant(chunk: Chunk, answer_ranges: Sequence[TimeRange]) -> bool:
    """True if the chunk's time window overlaps any answer range."""
    return any(chunk.start < end and start < chunk.end for start, end in answer_ranges)


def first_relevant_rank(results: Sequence[Chunk], answer_ranges: Sequence[TimeRange]) -> int | None:
    """1-based rank of the first relevant chunk, or None if no result is relevant."""
    return next((rank for rank, chunk in enumerate(results, start=1) if is_relevant(chunk, answer_ranges)), None)


def hit_rate(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0


def mean_reciprocal_rank(ranks: Sequence[int | None], k: int) -> float:
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks) if ranks else 0.0
