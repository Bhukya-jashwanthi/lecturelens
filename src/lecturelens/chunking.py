"""Split a transcript into overlapping time windows ("chunks").

Transcript fragments are only ~3 seconds long, too short to carry meaning on
their own. We merge consecutive fragments into windows of about
`window_seconds`, and start each new window `overlap_seconds` before the
previous one ended, so an idea spoken across a boundary appears whole in at
least one chunk.

Chunking by time (instead of by characters, as most text RAG does) gives
every chunk an exact start time, which becomes a clickable citation.

Run directly to see the chunks of a cached lecture:
    python -m lecturelens.chunking aircAruvnKk
"""

import sys

from lecturelens.models import Chunk, Transcript

# Chosen by evaluation (eval/results.md): with hybrid search, 30s windows put the
# right moment first far more often than 60s or 120s windows (MRR 0.90 vs 0.77 / 0.80).
DEFAULT_WINDOW_SECONDS = 30.0
DEFAULT_OVERLAP_SECONDS = 5.0


def chunk_transcript(
    transcript: Transcript,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> list[Chunk]:
    """Group transcript segments into overlapping ~`window_seconds` chunks."""
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if not 0 <= overlap_seconds < window_seconds:
        raise ValueError("overlap_seconds must be >= 0 and smaller than window_seconds")

    segments = transcript.segments
    chunks: list[Chunk] = []
    i = 0
    while i < len(segments):
        window_start = segments[i].start

        # Take every segment that *starts* inside this window (always at least one,
        # so a single very long segment still becomes a chunk).
        j = i + 1
        while j < len(segments) and segments[j].start < window_start + window_seconds:
            j += 1
        window = segments[i:j]
        window_end = window[-1].start + window[-1].duration

        chunks.append(
            Chunk(
                chunk_id=f"{transcript.video_id}:{len(chunks)}",
                video_id=transcript.video_id,
                title=transcript.title,
                text=" ".join(s.text for s in window),
                start=window_start,
                end=window_end,
                youtube=transcript.youtube,
            )
        )
        if j >= len(segments):
            break

        # Next window begins at the first segment inside the last `overlap_seconds`
        # of this one. It must be after `i`, or we would loop forever.
        next_i = j
        while next_i - 1 > i and segments[next_i - 1].start >= window_end - overlap_seconds:
            next_i -= 1
        i = next_i

    return chunks


if __name__ == "__main__":
    from lecturelens.transcript import TranscriptError, load_transcript

    if len(sys.argv) != 2:
        sys.exit("Usage: python -m lecturelens.chunking <youtube-url-or-id>")
    try:
        lecture = load_transcript(sys.argv[1])
    except TranscriptError as exc:
        sys.exit(f"Error: {exc}")

    print(f"{lecture.title}: {len(lecture.segments)} segments\n")
    print(f"{'window':>8} {'overlap':>8} {'chunks':>7} {'avg words':>10}")
    for window, overlap in [(30, 5), (60, 15), (120, 30), (300, 60)]:
        result = chunk_transcript(lecture, window, overlap)
        avg_words = sum(len(c.text.split()) for c in result) / len(result)
        print(f"{window:>7}s {overlap:>7}s {len(result):>7} {avg_words:>10.0f}")

    default = chunk_transcript(lecture)
    print(f"\nFirst 3 chunks with the defaults ({DEFAULT_WINDOW_SECONDS:.0f}s window, {DEFAULT_OVERLAP_SECONDS:.0f}s overlap):")
    for chunk in default[:3]:
        print(f"\n[{chunk.chunk_id}] {chunk.timestamp} -> {chunk.end:.0f}s  {chunk.url or ''}")
        print(f"  {chunk.text[:220]}...")
