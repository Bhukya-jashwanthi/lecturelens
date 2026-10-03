from lecturelens.subtitles import parse_subtitles

SRT = """1
00:00:04,220 --> 00:00:05,400
This is a 3.

2
00:00:06,060 --> 00:00:10,713
It's sloppily written
at a low resolution.
"""

VTT = """WEBVTT
Kind: captions
Language: en

NOTE this block is ignored

00:00:01.000 --> 00:00:03.500 align:start position:0%
<c>hello</c> <00:00:02.000><c>world</c>

00:00:03.500 --> 00:00:06.000
hello world
neural networks

01:02:03.000 --> 01:02:04.000
past the one hour mark
"""

PANEL = """0:04
This is a 3.
0:06
6 seconds
It's sloppily written
1:02:03
past the one hour mark
"""


def test_parses_srt_with_multiline_cues():
    segments = parse_subtitles(SRT)
    assert [s.text for s in segments] == ["This is a 3.", "It's sloppily written at a low resolution."]
    assert segments[0].start == 4.22
    assert segments[1].duration == 4.653


def test_parses_vtt_strips_tags_and_rolling_duplicates():
    segments = parse_subtitles(VTT)
    assert [s.text for s in segments] == ["hello world", "neural networks", "past the one hour mark"]
    assert segments[1].start == 3.5
    assert segments[2].start == 3723.0  # 1h 2m 3s


def test_parses_youtube_panel_text_and_skips_screen_reader_noise():
    segments = parse_subtitles(PANEL)
    assert [s.text for s in segments] == ["This is a 3.", "It's sloppily written", "past the one hour mark"]
    assert [s.start for s in segments] == [4.0, 6.0, 3723.0]
    assert segments[0].duration == 2.0  # until the next timestamp


def test_handles_byte_order_mark():
    assert parse_subtitles("﻿" + SRT)[0].text == "This is a 3."


def test_returns_empty_list_for_text_without_timestamps():
    assert parse_subtitles("just some notes\nwith no timing") == []
    assert parse_subtitles("") == []
