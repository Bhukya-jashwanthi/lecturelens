"""RAG chain tests with a fake LLM and a fake search function: no API calls."""

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from lecturelens.models import Chunk
from lecturelens.prompts import NOT_COVERED
from lecturelens.rag_chain import LectureQA, format_sources, render_citations
from lecturelens.retriever import RetrievedChunk


def source(n: int, start: float, youtube: bool = True) -> RetrievedChunk:
    chunk = Chunk(f"vid:{n}", "abcdefghijk", "Neural nets", f"text {n}", start, start + 60, youtube)
    return RetrievedChunk(chunk, score=0.03, semantic_rank=n, keyword_rank=None)


SOURCES = [source(1, 4), source(2, 680)]


class FakeSearch:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def __call__(self, query, k, video_ids=None):
        self.queries.append(query)
        return self.results


def test_render_citations_creates_timestamp_links():
    text, cited = render_citations("Bias shifts the threshold [2].", SOURCES)
    assert text == "Bias shifts the threshold [▶ 11:20](https://www.youtube.com/watch?v=abcdefghijk&t=680s)."
    assert cited == [2]


def test_render_citations_handles_lists_and_drops_invented_numbers():
    text, cited = render_citations("A [1, 2]. B [7].", SOURCES)
    assert "[▶ 0:04]" in text and "[▶ 11:20]" in text
    assert "[7]" not in text and text.endswith("B.")
    assert cited == [1, 2]


def test_render_citations_separates_adjacent_links():
    text, _ = render_citations("Both [1][2].", SOURCES)
    assert ")[" not in text and ") [▶ 11:20]" in text


def test_render_citations_without_youtube_link():
    text, _ = render_citations("Point [1].", [source(1, 65, youtube=False)])
    assert text == "Point [1:05]."


def test_format_sources_numbers_each_chunk():
    formatted = format_sources(SOURCES)
    assert formatted.startswith("[1] Neural nets @ 0:04\ntext 1")
    assert "[2] Neural nets @ 11:20\ntext 2" in formatted


def test_ask_returns_grounded_answer_with_citations():
    qa = LectureQA(llm=FakeListChatModel(responses=["The bias shifts the threshold [2]."]), search=FakeSearch(SOURCES))
    answer = qa.ask("what is the bias?")
    assert answer.grounded and answer.cited == [2]
    assert "[▶ 11:20]" in answer.text
    assert answer.search_query == "what is the bias?"  # no history -> no condensing


def test_ask_detects_not_covered_answers():
    qa = LectureQA(llm=FakeListChatModel(responses=[NOT_COVERED]), search=FakeSearch(SOURCES))
    answer = qa.ask("who won the 2022 world cup?")
    assert not answer.grounded and answer.cited == []


def test_follow_up_question_is_condensed_before_search():
    llm = FakeListChatModel(responses=["What is the bias, explained simply?", "It moves the threshold [1]."])
    search = FakeSearch(SOURCES)
    answer = LectureQA(llm=llm, search=search).ask(
        "explain that more simply", history=[("what is the bias?", "The bias shifts the threshold [2].")]
    )
    assert search.queries == ["What is the bias, explained simply?"]
    assert answer.question == "explain that more simply"
    assert answer.search_query == "What is the bias, explained simply?"


def test_ask_without_indexed_lectures_does_not_call_the_llm():
    llm = FakeListChatModel(responses=["should not be used"])
    answer = LectureQA(llm=llm, search=FakeSearch([])).ask("anything")
    assert not answer.grounded and "No lectures" in answer.text
