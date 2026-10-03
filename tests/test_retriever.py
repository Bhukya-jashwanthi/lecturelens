import pytest
from langchain_core.embeddings import DeterministicFakeEmbedding

from lecturelens.models import Segment, Transcript
from lecturelens.retriever import hybrid_search, keyword_search, reciprocal_rank_fusion, tokenize
from lecturelens.vectorstore import build_vectorstore, index_transcript

# --- pure functions -------------------------------------------------------------


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What is the Bias used for, in 2 layers?") == ["bias", "used", "2", "layers"]


def test_tokenize_normalizes_thousands_separators():
    assert tokenize("about 13,000 weights") == tokenize("about 13000 weights") == ["about", "13000", "weights"]
    assert tokenize("1,234,567 and 3,5") == ["1234567", "3", "5"]  # "3,5" is a list, not a number


def test_rrf_rewards_items_ranked_well_in_both_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "a"]])
    assert [item for item, _ in fused] == ["b", "a", "c"]


def test_rrf_score_formula():
    fused = dict(reciprocal_rank_fusion([["a"], ["x", "a"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["x"] == pytest.approx(1 / 61)


def test_rrf_handles_empty_lists():
    assert reciprocal_rank_fusion([[], []]) == []
    assert [i for i, _ in reciprocal_rank_fusion([["a", "b"], []])] == ["a", "b"]


# --- with a real (local) Chroma store and fake embeddings -----------------------


@pytest.fixture
def store(tmp_path):
    store = build_vectorstore(DeterministicFakeEmbedding(size=32), tmp_path / "chroma", "test")
    topics = ["neurons hold activations", "the bias shifts the threshold", "sigmoid squishes values"]
    # 3 topics, each ~60s of speech, so each becomes its own chunk.
    segments = [Segment(f"{topic} sentence {i}.", t * 60 + i * 5.0, 5.0) for t, topic in enumerate(topics) for i in range(9)]
    index_transcript(Transcript("aaaaaaaaaaa", "Neural nets", "C", "en", segments), store)
    index_transcript(
        Transcript("bbbbbbbbbbb", "Cooking", "C", "en", [Segment("bias in cooking reviews", 0, 5)]), store
    )
    return store


def test_keyword_search_finds_exact_term(store):
    results = keyword_search("threshold", k=5, store=store)
    assert results and "threshold" in results[0][0].text


def test_keyword_search_skips_chunks_without_any_query_word(store):
    assert keyword_search("photosynthesis", store=store) == []


def test_hybrid_search_includes_keyword_match_and_explains_ranks(store):
    results = hybrid_search("threshold", k=4, store=store)
    # Chunks overlap, so more than one chunk can contain the term; the top result must be one of them.
    assert "threshold" in results[0].chunk.text
    assert results[0].keyword_rank is not None
    # The best BM25 match is never dropped by fusion.
    best_keyword_id = keyword_search("threshold", k=1, store=store)[0][0].chunk_id
    assert best_keyword_id in {r.chunk.chunk_id for r in results}
    assert all(r.score > 0 for r in results)
    assert results == sorted(results, key=lambda r: r.score, reverse=True)


def test_hybrid_search_respects_lecture_filter(store):
    results = hybrid_search("bias", k=5, video_ids=["bbbbbbbbbbb"], store=store)
    assert results and {r.chunk.video_id for r in results} == {"bbbbbbbbbbb"}
