"""Parse caption files into timed segments.

Supported formats (detected automatically):
  * SRT  - "00:01:02,345 --> 00:01:05,000" cue timings (common subtitle downloads)
  * VTT  - "00:01:02.345 --> 00:01:05.000" (Zoom, Teams, Google Meet, Coursera, YouTube)
  * YouTube transcript panel text, copied from "...more -> Show transcript":
        0:04
        This is a 3.
        0:06
        It's sloppily written...

This gives LectureLens an input path that does not depend on YouTube allowing
automated transcript downloads.
"""

import re

from lecturelens.models import Segment

# Timestamps like "1:02:03.456", "02:03,456" or "02:03.456" (hours optional, , or . before ms).
_CUE_TIME = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{3})"
_CUE_LINE = re.compile(rf"^\s*{_CUE_TIME}\s*-->\s*{_CUE_TIME}")
# Panel timestamps like "0:04", "12:34" or "1:02:03" alone on a line.
_PANEL_TIME = re.compile(r"^\s*(?:(\d+):)?(\d{1,2}):(\d{2})\s*$")
# Screen-reader text the panel sometimes includes, e.g. "4 seconds" or "1 minute, 2 seconds".
_PANEL_NOISE = re.compile(r"^\s*\d+\s+(?:hours?|minutes?|seconds?)(?:,\s*\d+\s+(?:minutes?|seconds?))*\s*$")
_TAGS = re.compile(r"<[^>]+>")  # VTT styling/timing tags such as <c> or <00:00:01.000>

_LAST_SEGMENT_SECONDS = 5.0  # duration assumed for the final panel line (no next timestamp)


def _seconds(hours: str | None, minutes: str, seconds: str, millis: str = "0") -> float:
    return int(hours or 0) * 3600 + int(minutes) * 60 + int(seconds) + int(millis) / 1000


def _clean(text: str) -> str:
    return " ".join(_TAGS.sub("", text).split())


def parse_subtitles(text: str) -> list[Segment]:
    """Parse SRT, VTT or copied YouTube-panel text into segments. Returns [] if nothing parses."""
    lines = text.lstrip("﻿").splitlines()
    if any(_CUE_LINE.match(line) for line in lines):
        return _parse_cues(lines)
    return _parse_panel(lines)


def _parse_cues(lines: list[str]) -> list[Segment]:
    """SRT/VTT: a timing line, then one or more text lines, then a blank line."""
    segments: list[Segment] = []
    last_line = ""
    i = 0
    while i < len(lines):
        match = _CUE_LINE.match(lines[i])
        i += 1
        if not match:
            continue  # headers ("WEBVTT"), NOTE blocks, numeric cue ids, blank lines
        start = _seconds(*match.groups()[:4])
        end = _seconds(*match.groups()[4:])

        new_lines = []
        while i < len(lines) and lines[i].strip():
            line = _clean(lines[i])
            # Auto-generated captions "roll": each cue repeats the previous line.
            if line and line != last_line:
                new_lines.append(line)
                last_line = line
            i += 1
        if new_lines:
            segments.append(Segment(" ".join(new_lines), start, round(max(end - start, 0.0), 3)))
    return segments


def _parse_panel(lines: list[str]) -> list[Segment]:
    """YouTube panel copy: a timestamp line followed by the text spoken from that time."""
    starts: list[float] = []
    texts: list[list[str]] = []
    for line in lines:
        match = _PANEL_TIME.match(line)
        if match:
            starts.append(_seconds(*match.groups()))
            texts.append([])
        elif starts and line.strip() and not _PANEL_NOISE.match(line):
            texts[-1].append(_clean(line))

    segments = []
    for idx, (start, parts) in enumerate(zip(starts, texts)):
        if not parts:
            continue
        next_start = starts[idx + 1] if idx + 1 < len(starts) else start + _LAST_SEGMENT_SECONDS
        segments.append(Segment(" ".join(parts), start, round(max(next_start - start, 0.0), 3)))
    return segments
