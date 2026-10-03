"""Fetch YouTube transcripts and cache them on disk.

A transcript arrives as hundreds of short fragments (~3 seconds each), each with
its start time. We keep that timing information because it is what makes
clickable timestamp citations possible later on.

Run directly to try it:
    python -m lecturelens.transcript https://www.youtube.com/watch?v=aircAruvnKk
"""

import json
import logging
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import CouldNotRetrieveTranscript

from lecturelens.config import get_settings

logger = logging.getLogger(__name__)

# YouTube video IDs are exactly 11 characters from this alphabet.
_VIDEO_ID = r"[A-Za-z0-9_-]{11}"
_URL_PATTERNS = [
    re.compile(rf"(?:youtube\.com/(?:watch\?(?:.*&)?v=|embed/|shorts/|live/)|youtu\.be/)({_VIDEO_ID})"),
    re.compile(rf"^({_VIDEO_ID})$"),  # a bare ID
]


class TranscriptError(Exception):
    """Raised when a transcript cannot be obtained (disabled, private, wrong URL...)."""


@dataclass
class Segment:
    """One transcript fragment: what was said, starting when, for how long (seconds)."""

    text: str
    start: float
    duration: float


@dataclass
class Transcript:
    video_id: str
    title: str
    channel: str
    language: str
    segments: list[Segment] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"https://www.youtube.com/watch?v={self.video_id}"


def extract_video_id(url_or_id: str) -> str:
    """Return the 11-character video ID from any common YouTube URL format, or a bare ID."""
    candidate = url_or_id.strip()
    for pattern in _URL_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match.group(1)
    raise TranscriptError(f"Not a valid YouTube URL or video ID: {url_or_id!r}")


def fetch_metadata(video_id: str) -> tuple[str, str]:
    """Return (title, channel) using YouTube's public oEmbed endpoint (no API key needed).

    Metadata is nice-to-have, so on any failure we fall back to the video ID
    rather than failing the whole import.
    """
    video_url = f"https://www.youtube.com/watch?v={video_id}"
    oembed = "https://www.youtube.com/oembed?" + urllib.parse.urlencode({"url": video_url, "format": "json"})
    try:
        with urllib.request.urlopen(oembed, timeout=10) as response:
            data = json.load(response)
        return data.get("title", video_id), data.get("author_name", "Unknown")
    except Exception as exc:  # network error, private video, etc.
        logger.warning("Could not fetch metadata for %s: %s", video_id, exc)
        return video_id, "Unknown"


def _clean(text: str) -> str:
    """Collapse line breaks and repeated spaces inside a fragment."""
    return " ".join(text.split())


def fetch_transcript(video_id: str, languages: tuple[str, ...] = ("en",)) -> Transcript:
    """Download a transcript from YouTube (no caching; see load_transcript)."""
    try:
        fetched = YouTubeTranscriptApi().fetch(video_id, languages=languages)
    except CouldNotRetrieveTranscript as exc:
        # The library's errors are long and technical; keep the first line for users.
        reason = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
        raise TranscriptError(f"No transcript available for {video_id}: {reason}") from exc

    segments = [
        Segment(text=_clean(s["text"]), start=s["start"], duration=s["duration"])
        for s in fetched.to_raw_data()
        if _clean(s["text"])  # drop empty fragments
    ]
    title, channel = fetch_metadata(video_id)
    return Transcript(video_id, title, channel, fetched.language_code, segments)


def load_transcript(url_or_id: str, cache_dir: Path | None = None) -> Transcript:
    """Return the transcript for a video, using the on-disk JSON cache when available."""
    video_id = extract_video_id(url_or_id)
    cache_dir = cache_dir or get_settings().transcripts_dir
    cache_file = cache_dir / f"{video_id}.json"

    if cache_file.exists():
        logger.info("Loaded %s from cache", video_id)
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        data["segments"] = [Segment(**s) for s in data["segments"]]
        return Transcript(**data)

    transcript = fetch_transcript(video_id)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(asdict(transcript), ensure_ascii=False, indent=1), encoding="utf-8")
    logger.info("Fetched and cached %s (%d segments)", video_id, len(transcript.segments))
    return transcript


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) != 2:
        sys.exit("Usage: python -m lecturelens.transcript <youtube-url-or-id>")
    t = load_transcript(sys.argv[1])
    minutes = (t.segments[-1].start + t.segments[-1].duration) / 60
    print(f"\n{t.title}  ({t.channel})")
    print(f"{len(t.segments)} segments, {minutes:.1f} min, language={t.language}")
    for s in t.segments[:3]:
        print(f"  [{s.start:7.2f}s] {s.text}")
