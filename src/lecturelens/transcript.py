"""Load lecture transcripts from YouTube or caption files, with an on-disk cache.

A transcript arrives as hundreds of short fragments (~3 seconds each), each with
its start time. We keep that timing information because it is what makes
clickable timestamp citations possible later on.

Run directly to try it:
    python -m lecturelens.transcript https://www.youtube.com/watch?v=aircAruvnKk
    python -m lecturelens.transcript lecture.vtt
    python -m lecturelens.transcript lecture.txt https://www.youtube.com/watch?v=aircAruvnKk
"""

import json
import logging
import re
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path

from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
    YouTubeTranscriptApi,
)

from lecturelens.config import get_settings
from lecturelens.models import Segment, Transcript
from lecturelens.subtitles import parse_subtitles

logger = logging.getLogger(__name__)

# YouTube video IDs are exactly 11 characters from this alphabet.
_VIDEO_ID = r"[A-Za-z0-9_-]{11}"
_URL_PATTERNS = [
    re.compile(rf"(?:youtube\.com/(?:watch\?(?:.*&)?v=|embed/|shorts/|live/)|youtu\.be/)({_VIDEO_ID})"),
    re.compile(rf"^({_VIDEO_ID})$"),  # a bare ID
]

# Short, actionable messages for the library's (long, technical) exceptions.
_FRIENDLY_ERRORS = {
    RequestBlocked: (
        "YouTube is temporarily blocking transcript requests from your network. "
        "Try again later, or import the captions as a file instead "
        "(YouTube: '...more' -> 'Show transcript', copy it into a .txt file)."
    ),
    TranscriptsDisabled: "captions are turned off for this video.",
    NoTranscriptFound: "this video has no English captions.",
    VideoUnavailable: "the video is private, deleted, or the ID is wrong.",
}


class TranscriptError(Exception):
    """Raised when a transcript cannot be obtained (blocked, disabled, wrong URL, empty file...)."""


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
        reason = next(
            (msg for err_type, msg in _FRIENDLY_ERRORS.items() if isinstance(exc, err_type)),
            f"YouTube returned an error ({type(exc).__name__}).",
        )
        raise TranscriptError(f"Could not get a transcript for {video_id}: {reason}") from exc

    segments = [
        Segment(text=_clean(s["text"]), start=s["start"], duration=s["duration"])
        for s in fetched.to_raw_data()
        if _clean(s["text"])  # drop empty fragments
    ]
    title, channel = fetch_metadata(video_id)
    return Transcript(video_id, title, channel, fetched.language_code, segments)


def _cache_path(video_id: str, cache_dir: Path | None) -> Path:
    return (cache_dir or get_settings().transcripts_dir) / f"{video_id}.json"


def _save(transcript: Transcript, cache_file: Path) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(asdict(transcript), ensure_ascii=False, indent=1), encoding="utf-8")


def load_transcript(url_or_id: str, cache_dir: Path | None = None) -> Transcript:
    """Return the transcript for a video, using the on-disk JSON cache when available."""
    video_id = extract_video_id(url_or_id)
    cache_file = _cache_path(video_id, cache_dir)

    if cache_file.exists():
        logger.info("Loaded %s from cache", video_id)
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        data["segments"] = [Segment(**s) for s in data["segments"]]
        return Transcript(**data)

    transcript = fetch_transcript(video_id)
    _save(transcript, cache_file)
    logger.info("Fetched and cached %s (%d segments)", video_id, len(transcript.segments))
    return transcript


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "lecture"


def import_caption_file(
    path: Path, youtube_url: str | None = None, title: str | None = None, cache_dir: Path | None = None
) -> Transcript:
    """Import an .srt/.vtt file or copied YouTube transcript text, and cache it like a download.

    If `youtube_url` is given, the captions are linked to that video so citations
    become clickable YouTube timestamps.
    """
    segments = parse_subtitles(path.read_text(encoding="utf-8", errors="replace"))
    if not segments:
        raise TranscriptError(f"No timed captions found in {path.name}. Expected SRT, VTT or YouTube transcript text.")

    if youtube_url:
        video_id = extract_video_id(youtube_url)
        meta_title, channel = fetch_metadata(video_id)
        transcript = Transcript(video_id, title or meta_title, channel, "en", segments, youtube=True)
    else:
        transcript = Transcript(f"file-{_slug(path.stem)}", title or path.stem, "Imported file", "en", segments, youtube=False)

    _save(transcript, _cache_path(transcript.video_id, cache_dir))
    logger.info("Imported %s as %s (%d segments)", path.name, transcript.video_id, len(segments))
    return transcript


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if len(sys.argv) not in (2, 3):
        sys.exit(
            "Usage: python -m lecturelens.transcript <youtube-url-or-id>\n"
            "       python -m lecturelens.transcript <caption-file> [youtube-url]"
        )
    try:
        source = Path(sys.argv[1])
        if source.is_file():
            t = import_caption_file(source, youtube_url=sys.argv[2] if len(sys.argv) == 3 else None)
        else:
            t = load_transcript(sys.argv[1])
    except TranscriptError as exc:
        sys.exit(f"Error: {exc}")
    minutes = (t.segments[-1].start + t.segments[-1].duration) / 60
    print(f"\n{t.title}  ({t.channel})")
    print(f"{len(t.segments)} segments, {minutes:.1f} min, id={t.video_id}, link={t.url or 'none'}")
    for s in t.segments[:3]:
        print(f"  [{s.start:7.2f}s] {s.text}")
