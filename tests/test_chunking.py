import pytest

from lecturelens.chunking import chunk_transcript
from lecturelens.models import Segment, Transcript, format_timestamp


def make_transcript(n_segments: int, seconds_each: float = 5.0, youtube: bool = True) -> Transcript:
    segments = [Segment(f"s{i}", i * seconds_each, seconds_each) for i in range(n_segments)]
    return Transcript("abcdefghijk", "Lecture", "Channel", "en", segments, youtube=youtube)


def test_every_segment_appears_in_some_chunk():
    chunks = chunk_transcript(make_transcript(100), window_seconds=60, overlap_seconds=15)
    words = {w for c in chunks for w in c.text.split()}
    assert words == {f"s{i}" for i in range(100)}


def test_chunks_respect_window_size_and_overlap():
    chunks = chunk_transcript(make_transcript(100), window_seconds=60, overlap_seconds=15)
    for chunk in chunks:
        assert chunk.end - chunk.start <= 60 + 5  # window plus at most one segment's duration
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start < prev.end  # consecutive chunks overlap...
        assert nxt.start > prev.start  # ...but always move forward
        assert prev.end - nxt.start == pytest.approx(15, abs=5)


def test_no_overlap_gives_back_to_back_chunks():
    chunks = chunk_transcript(make_transcript(24), window_seconds=60, overlap_seconds=0)
    assert [c.start for c in chunks] == [0, 60]


def test_chunk_ids_and_links():
    chunks = chunk_transcript(make_transcript(30))
    assert [c.chunk_id for c in chunks][:2] == ["abcdefghijk:0", "abcdefghijk:1"]
    assert chunks[1].url == f"https://www.youtube.com/watch?v=abcdefghijk&t={int(chunks[1].start)}s"
    assert chunk_transcript(make_transcript(30, youtube=False))[0].url is None


def test_single_segment_longer_than_window_still_makes_one_chunk():
    transcript = Transcript("abcdefghijk", "T", "C", "en", [Segment("long", 0, 500)])
    chunks = chunk_transcript(transcript, window_seconds=60, overlap_seconds=15)
    assert len(chunks) == 1 and chunks[0].end == 500


def test_empty_transcript_gives_no_chunks():
    assert chunk_transcript(make_transcript(0)) == []


@pytest.mark.parametrize("window, overlap", [(0, 0), (60, 60), (60, 90), (60, -1)])
def test_invalid_settings_are_rejected(window, overlap):
    with pytest.raises(ValueError):
        chunk_transcript(make_transcript(10), window, overlap)


@pytest.mark.parametrize("seconds, expected", [(0, "0:00"), (65.9, "1:05"), (3723, "1:02:03")])
def test_format_timestamp(seconds, expected):
    assert format_timestamp(seconds) == expected
