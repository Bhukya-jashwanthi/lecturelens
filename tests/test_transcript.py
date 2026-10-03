import pytest

from lecturelens.transcript import TranscriptError, extract_video_id


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
