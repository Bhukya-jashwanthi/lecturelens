import pytest

from lecturelens.transcript import TranscriptError, extract_video_id, import_caption_file, load_transcript


@pytest.mark.parametrize(
    "url",
    [
        "https://www.youtube.com/watch?v=aircAruvnKk",
        "https://youtube.com/watch?v=aircAruvnKk&t=120s",
        "https://www.youtube.com/watch?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&v=aircAruvnKk",
        "https://youtu.be/aircAruvnKk",
        "https://youtu.be/aircAruvnKk?si=abc123",
        "https://www.youtube.com/embed/aircAruvnKk",
        "https://www.youtube.com/shorts/aircAruvnKk",
        "https://m.youtube.com/watch?v=aircAruvnKk",
        "aircAruvnKk",
        "  aircAruvnKk  ",
    ],
)
def test_extract_video_id_accepts_common_formats(url):
    assert extract_video_id(url) == "aircAruvnKk"


@pytest.mark.parametrize("bad", ["", "hello", "https://example.com/watch?v=aircAruvnKk", "https://youtu.be/short"])
def test_extract_video_id_rejects_invalid_input(bad):
    with pytest.raises(TranscriptError):
        extract_video_id(bad)


def test_import_caption_file_without_youtube_link_is_cached_and_reloadable(tmp_path):
    captions = tmp_path / "Week 3 Lecture.vtt"
    captions.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:04.000\nGradient descent\n", encoding="utf-8")

    transcript = import_caption_file(captions, cache_dir=tmp_path / "cache")

    assert transcript.video_id == "file-week-3-lecture"
    assert transcript.title == "Week 3 Lecture"
    assert transcript.youtube is False and transcript.url is None
    assert (tmp_path / "cache" / "file-week-3-lecture.json").exists()


def test_import_caption_file_rejects_files_without_timestamps(tmp_path):
    notes = tmp_path / "notes.txt"
    notes.write_text("no timestamps here", encoding="utf-8")
    with pytest.raises(TranscriptError, match="No timed captions"):
        import_caption_file(notes, cache_dir=tmp_path)


def test_load_transcript_uses_cache_without_network(tmp_path):
    cached = tmp_path / "aircAruvnKk.json"
    cached.write_text(
        '{"video_id": "aircAruvnKk", "title": "T", "channel": "C", "language": "en",'
        ' "segments": [{"text": "hi", "start": 1.0, "duration": 2.0}]}',
        encoding="utf-8",
    )
    transcript = load_transcript("https://youtu.be/aircAruvnKk", cache_dir=tmp_path)
    assert transcript.segments[0].text == "hi"
    assert transcript.youtube is True  # older cache files without the field default to YouTube
