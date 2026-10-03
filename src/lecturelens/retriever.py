"""Hybrid retrieval: semantic search + BM25 keyword search, merged with Reciprocal Rank Fusion.

Semantic search understands meaning ("squishing function" -> sigmoid) but can
under-rank chunks that contain the exact term asked about ("bias"). BM25
keyword search has the opposite strengths. RRF merges the two rankings using
only positions, because their raw scores are on incomparable scales.

Run directly to compare the three methods side by side:
    python -m lecturelens.retriever "what is the bias used for"
"""

import re
import sys
from dataclasses import dataclass

from langchain_chroma import Chroma
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from rank_bm25 import BM25Okapi

from lecturelens.models import Chunk
from lecturelens.vectorstore import chunk_to_document, document_to_chunk, get_vectorstore, semantic_search

RRF_K = 60  # standard constant from the original RRF paper (Cormack et al., 2009)

# Very common words carry no information for keyword matching.
_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in is it its of on or so that the "
    "then there these this to was we what when where which who why will with you your".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),(?=\d{3}\b)")  # the comma in "13,000"


@dataclass
class RetrievedChunk:
    """A search result plus *why* it was retrieved (useful for debugging and the UI)."""

    chunk: Chunk
    score: float  # fused RRF score (higher = more relevant)
    semantic_rank: int | None  # 1-based position in semantic results, None if absent
    keyword_rank: int | None  # 1-based position in BM25 results, None if absent


def tokenize(text: str) -> list[str]:
    """Lowercase words and numbers, without stopwords. "13,000" and "13000" become the same token."""
    text = _THOUSANDS_SEPARATOR.sub("", text.lower())
    return [t for t in _TOKEN.findall(text) if t not in _STOPWORDS]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = RRF_K) -> list[tuple[str, float]]:
    """Merge ranked lists of IDs. Each list adds 1/(k + rank) for every ID it contains."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item_id in enumerate(ranking, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def keyword_search(
    query: str, k: int = 5, video_ids: list[str] | None = None, store: Chroma | None = None
) -> list[tuple[Chunk, float]]:
    """Rank chunks with BM25. Chunks sharing no words with the query are left out.

    The BM25 index is rebuilt from the stored chunks on each call. That takes
    milliseconds for a few thousand chunks and can never go stale; a large
    deployment would cache it and rebuild only when lectures change.
    """
    store = store or get_vectorstore()
    where = {"video_id": {"$in": video_ids}} if video_ids else None
    stored = store.get(where=where, include=["documents", "metadatas"])
    if not stored["ids"]:
        return []

    docs = [
        Document(id=doc_id, page_content=text, metadata=meta)
        for doc_id, text, meta in zip(stored["ids"], stored["documents"], stored["metadatas"])
    ]
    bm25 = BM25Okapi([tokenize(d.page_content) for d in docs])
    scores = bm25.get_scores(tokenize(query))

    ranked = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
    return [(document_to_chunk(doc), float(score)) for doc, score in ranked[:k] if score > 0]


def hybrid_search(
    query: str,
    k: int = 5,
    video_ids: list[str] | None = None,
    candidates: int = 20,
    store: Chroma | None = None,
) -> list[RetrievedChunk]:
    """Top-k chunks by RRF over the top `candidates` of semantic and keyword search."""
    semantic = semantic_search(query, k=candidates, video_ids=video_ids, store=store)
    keyword = keyword_search(query, k=candidates, video_ids=video_ids, store=store)

    chunks = {c.chunk_id: c for c, _ in semantic} | {c.chunk_id: c for c, _ in keyword}
    semantic_ids = [c.chunk_id for c, _ in semantic]
    keyword_ids = [c.chunk_id for c, _ in keyword]
    fused = reciprocal_rank_fusion([semantic_ids, keyword_ids])

    return [
        RetrievedChunk(
            chunk=chunks[chunk_id],
            score=score,
            semantic_rank=semantic_ids.index(chunk_id) + 1 if chunk_id in semantic_ids else None,
            keyword_rank=keyword_ids.index(chunk_id) + 1 if chunk_id in keyword_ids else None,
        )
        for chunk_id, score in fused[:k]
    ]


class HybridRetriever(BaseRetriever):
    """LangChain-compatible wrapper, so hybrid search plugs into any LangChain chain."""

    k: int = 5
    video_ids: list[str] | None = None

    def _get_relevant_documents(self, query: str, *, run_manager: CallbackManagerForRetrieverRun) -> list[Document]:
        results = hybrid_search(query, k=self.k, video_ids=self.video_ids)
        docs = []
        for r in results:
            doc = chunk_to_document(r.chunk)
            doc.metadata |= {"score": r.score, "semantic_rank": r.semantic_rank, "keyword_rank": r.keyword_rank}
            docs.append(doc)
        return docs


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Usage: python -m lecturelens.retriever "your question"')
    question = " ".join(sys.argv[1:])

    print(f"Q: {question}\n\nSEMANTIC ONLY")
    for chunk, score in semantic_search(question, k=3):
        print(f"  {score:.3f}  [{chunk.timestamp}] {chunk.text[:90]}...")
    print("\nKEYWORD ONLY (BM25)")
    for chunk, score in keyword_search(question, k=3):
        print(f"  {score:5.2f}  [{chunk.timestamp}] {chunk.text[:90]}...")
    print("\nHYBRID (RRF)")
    for r in hybrid_search(question, k=3):
        ranks = f"sem #{r.semantic_rank or '-'}, kw #{r.keyword_rank or '-'}"
        print(f"  {r.score:.4f}  [{r.chunk.timestamp}] ({ranks}) {r.chunk.text[:70]}...")
