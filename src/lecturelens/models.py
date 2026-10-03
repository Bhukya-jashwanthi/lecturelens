"""Data shapes shared across the pipeline."""

from dataclasses import dataclass, field


@dataclass
class Segment:
    """One transcript fragment: what was said, starting when, for how long (seconds)."""

    text: str
    start: float
    duration: float


@dataclass
class Transcript:
    """A full lecture transcript.

    `video_id` is the YouTube ID for YouTube sources, or "file-<name>" for
    imported caption files that are not linked to a YouTube video.
    """

    video_id: str
    title: str
    channel: str
    language: str
    segments: list[Segment] = field(default_factory=list)
    youtube: bool = True

    @property
    def url(self) -> str | None:
        """Link to the video, or None for caption files without a YouTube video."""
        return f"https://www.youtube.com/watch?v={self.video_id}" if self.youtube else None


@dataclass
class Chunk:
    """A searchable window of a lecture (~30 seconds by default), the unit we embed and retrieve."""

    chunk_id: str  # "<video_id>:<index>", unique across all lectures
    video_id: str
    title: str
    text: str
    start: float
    end: float
    youtube: bool = True

    @property
    def timestamp(self) -> str:
        return format_timestamp(self.start)

    @property
    def url(self) -> str | None:
        """Link that opens the video at this chunk's start time."""
        if not self.youtube:
            return None
        return f"https://www.youtube.com/watch?v={self.video_id}&t={int(self.start)}s"


def format_timestamp(seconds: float) -> str:
    """Format seconds as m:ss, or h:mm:ss for times past one hour (like YouTube does)."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
