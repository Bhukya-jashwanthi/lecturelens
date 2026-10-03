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
