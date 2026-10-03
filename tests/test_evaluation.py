import pytest

from lecturelens.evaluation import first_relevant_rank, hit_rate, is_relevant, mean_reciprocal_rank
from lecturelens.models import Chunk


def chunk(start: float, end: float) -> Chunk:
    return Chunk("v:0", "v", "T", "text", start, end)


@pytest.mark.parametrize(
    "start, end, expected",
    [(0, 60, True), (55, 115, True), (100, 160, True), (110, 170, False), (0, 50, False)],
)
def test_is_relevant_when_chunk_overlaps_answer(start, end, expected):
    assert is_relevant(chunk(start, end), [(50, 110)]) is expected


def test_is_relevant_with_multiple_answer_ranges():
    assert is_relevant(chunk(900, 960), [(10, 20), (950, 955)])


def test_first_relevant_rank():
    results = [chunk(0, 60), chunk(300, 360), chunk(600, 660)]
    assert first_relevant_rank(results, [(310, 320)]) == 2
    assert first_relevant_rank(results, [(1000, 1010)]) is None


def test_hit_rate_and_mrr():
    ranks = [1, 2, None, 5]
    assert hit_rate(ranks, k=1) == 0.25
    assert hit_rate(ranks, k=3) == 0.5
    assert hit_rate(ranks, k=5) == 0.75
    assert mean_reciprocal_rank(ranks, k=5) == pytest.approx((1 + 0.5 + 0.2) / 4)
    assert mean_reciprocal_rank(ranks, k=1) == 0.25


def test_metrics_on_empty_input():
    assert hit_rate([], 5) == 0.0 and mean_reciprocal_rank([], 5) == 0.0
