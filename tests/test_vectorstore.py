"""Vector store tests using a fake, deterministic embedding model: offline, fast and free.

The fake model maps identical text to identical vectors, which is enough to test
our indexing, filtering and bookkeeping logic without calling the Gemini API.
"""

import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from lecturelens.models import Segment, Transcript
from lecturelens.vectorstore import (
    build_vectorstore,
    collection_name,
    delete_lecture,
    index_transcript,
    list_lectures,
    semantic_search,
)


@pytest.fixture
def store(tmp_path):
    return build_vectorstore(DeterministicFakeEmbedding(size=32), tmp_path / "chroma", "test")


def lecture(video_id: str, title: str, n_segments: int = 40) -> Transcript:
    segments = [Segment(f"{title} sentence {i}.", i * 5.0, 5.0) for i in range(n_segments)]
    return Transcript(video_id, title, "Channel", "en", segments)


def test_index_and_list_lectures(store):
    index_transcript(lecture("aaaaaaaaaaa", "Neural networks"), store)
    index_transcript(lecture("bbbbbbbbbbb", "Backpropagation"), store)

    lectures = list_lectures(store)
    assert [lec["title"] for lec in lectures] == ["Backpropagation", "Neural networks"]
    assert all(lec["chunks"] > 1 for lec in lectures)


def test_reindexing_replaces_instead_of_duplicating(store):
    first = index_transcript(lecture("aaaaaaaaaaa", "Neural networks"), store)
    second = index_transcript(lecture("aaaaaaaaaaa", "Neural networks"), store)
    assert first == second
    assert list_lectures(store)[0]["chunks"] == first


def test_reindexing_with_fewer_chunks_removes_stale_ones(store):
    long_version = index_transcript(lecture("aaaaaaaaaaa", "Neural networks", n_segments=40), store)
    short_version = index_transcript(lecture("aaaaaaaaaaa", "Neural networks", n_segments=10), store)
    assert short_version < long_version
    assert list_lectures(store)[0]["chunks"] == short_version


class FailingEmbedding(DeterministicFakeEmbedding):
    def embed_documents(self, texts):
        raise RuntimeError("quota exhausted")


def test_failed_reindex_keeps_the_existing_index(tmp_path):
    good = build_vectorstore(DeterministicFakeEmbedding(size=32), tmp_path / "chroma", "test")
    count = index_transcript(lecture("aaaaaaaaaaa", "Neural networks"), good)

    failing = build_vectorstore(FailingEmbedding(size=32), tmp_path / "chroma", "test")  # same collection
    with pytest.raises(RuntimeError):
        index_transcript(lecture("aaaaaaaaaaa", "Neural networks"), failing)

    assert list_lectures(good)[0]["chunks"] == count  # nothing was deleted


def test_search_returns_exact_match_first_with_metadata(store):
    transcript = lecture("aaaaaaaaaaa", "Neural networks")
    index_transcript(transcript, store)
    target = store.get(ids=["aaaaaaaaaaa:1"])["documents"][0]

    chunk, score = semantic_search(target, k=3, store=store)[0]

    assert chunk.chunk_id == "aaaaaaaaaaa:1"
    assert score == pytest.approx(1.0, abs=1e-6)  # identical text -> identical vector
    assert chunk.start > 0 and chunk.url.endswith(f"&t={int(chunk.start)}s")


def test_search_can_be_limited_to_selected_lectures(store):
    index_transcript(lecture("aaaaaaaaaaa", "Neural networks"), store)
    index_transcript(lecture("bbbbbbbbbbb", "Backpropagation"), store)

    results = semantic_search("anything", k=10, video_ids=["bbbbbbbbbbb"], store=store)

    assert results and {chunk.video_id for chunk, _ in results} == {"bbbbbbbbbbb"}


def test_delete_lecture(store):
    index_transcript(lecture("aaaaaaaaaaa", "Neural networks"), store)
    assert delete_lecture("aaaaaaaaaaa", store) > 0
    assert list_lectures(store) == []


def test_collection_name_is_safe_and_model_specific():
    assert collection_name("gemini-embedding-2", 768) == "lectures_gemini-embedding-2_768"
    assert collection_name("models/x", 768) != collection_name("models/x", 3072)
