"""Embed lecture chunks and store them in ChromaDB for semantic search.

Indexing:  chunk text -> Gemini embedding (768 numbers) -> ChromaDB, with metadata
           (video ID, title, start/end time) so results can be cited.
Search:    question  -> embedding -> the k chunks whose vectors are closest
           (cosine similarity) -> back to Chunk objects.

Run directly:
    python -m lecturelens.vectorstore index aircAruvnKk
    python -m lecturelens.vectorstore list
    python -m lecturelens.vectorstore search "how does the network learn?"
"""

import logging
import re
import sys
from functools import lru_cache
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from lecturelens.chunking import DEFAULT_OVERLAP_SECONDS, DEFAULT_WINDOW_SECONDS, chunk_transcript
from lecturelens.config import get_settings
from lecturelens.models import Chunk, Transcript
from lecturelens.retry import with_retries

logger = logging.getLogger(__name__)


def collection_name(model: str, dimensions: int) -> str:
    """Name the collection after the embedding model and size.

    Vectors from different models (or sizes) are not comparable. Giving each
    combination its own collection means switching models can never silently
    mix incompatible vectors; it just starts a fresh, empty index.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "-", f"lectures_{model}_{dimensions}")


@lru_cache
def get_embeddings() -> Embeddings:
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,
        google_api_key=settings.google_api_key,
        output_dimensionality=settings.embedding_dimensions,
    )


def build_vectorstore(embeddings: Embeddings, persist_dir: Path, name: str) -> Chroma:
    return Chroma(
        collection_name=name,
        embedding_function=embeddings,
        persist_directory=str(persist_dir),
        # Rank by cosine similarity (angle between vectors), the standard for text embeddings.
        collection_configuration={"hnsw": {"space": "cosine"}},
    )


@lru_cache
def get_vectorstore() -> Chroma:
    """The app's persistent vector store, configured from settings."""
    settings = get_settings()
    return build_vectorstore(
        get_embeddings(),
        settings.chroma_dir,
        collection_name(settings.embedding_model, settings.embedding_dimensions),
    )


def chunk_to_document(chunk: Chunk) -> Document:
    return Document(
        id=chunk.chunk_id,
        page_content=chunk.text,
        metadata={
            "video_id": chunk.video_id,
            "title": chunk.title,
            "start": chunk.start,
            "end": chunk.end,
            "youtube": chunk.youtube,
        },
    )


def document_to_chunk(doc: Document) -> Chunk:
    meta = doc.metadata
    return Chunk(
        chunk_id=doc.id or f"{meta['video_id']}:?",
        video_id=meta["video_id"],
        title=meta["title"],
        text=doc.page_content,
        start=meta["start"],
        end=meta["end"],
        youtube=meta["youtube"],
    )


def index_transcript(
    transcript: Transcript,
    store: Chroma | None = None,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap_seconds: float = DEFAULT_OVERLAP_SECONDS,
) -> int:
    """Chunk, embed and store a lecture. Re-indexing the same lecture replaces it. Returns the chunk count."""
    store = store or get_vectorstore()
    chunks = chunk_transcript(transcript, window_seconds, overlap_seconds)

    # Idempotent: remove the lecture's old chunks first, so indexing twice never
    # creates duplicates (and changed chunking settings fully take effect).
    old_ids = store.get(where={"video_id": transcript.video_id}, include=[])["ids"]
    if old_ids:
        store.delete(ids=old_ids)

    docs = [chunk_to_document(c) for c in chunks]
    with_retries(lambda: store.add_documents(docs, ids=[d.id for d in docs]))
    logger.info("Indexed %s: %d chunks", transcript.video_id, len(docs))
    return len(docs)


def list_lectures(store: Chroma | None = None) -> list[dict]:
    """One entry per indexed lecture: video_id, title, youtube, chunks (count)."""
    store = store or get_vectorstore()
    lectures: dict[str, dict] = {}
    for meta in store.get(include=["metadatas"])["metadatas"]:
        entry = lectures.setdefault(
            meta["video_id"], {"video_id": meta["video_id"], "title": meta["title"], "youtube": meta["youtube"], "chunks": 0}
        )
        entry["chunks"] += 1
    return sorted(lectures.values(), key=lambda e: e["title"].lower())


def delete_lecture(video_id: str, store: Chroma | None = None) -> int:
    store = store or get_vectorstore()
    ids = store.get(where={"video_id": video_id}, include=[])["ids"]
    if ids:
        store.delete(ids=ids)
    return len(ids)


def semantic_search(
    query: str, k: int = 5, video_ids: list[str] | None = None, store: Chroma | None = None
) -> list[tuple[Chunk, float]]:
    """Return the k chunks most similar in meaning to `query`, with cosine similarity (higher = closer)."""
    store = store or get_vectorstore()
    where = {"video_id": {"$in": video_ids}} if video_ids else None
    results = with_retries(lambda: store.similarity_search_with_score(query, k=k, filter=where))
    # Chroma returns cosine *distance* (0 = identical); convert to similarity.
    return [(document_to_chunk(doc), 1 - distance) for doc, distance in results]


if __name__ == "__main__":
    from lecturelens.transcript import TranscriptError, load_transcript

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    usage = "Usage: python -m lecturelens.vectorstore index <url-or-id> | list | search <question>"
    if len(sys.argv) < 2:
        sys.exit(usage)
    command, args = sys.argv[1], sys.argv[2:]

    if command == "index" and len(args) == 1:
        try:
            lecture = load_transcript(args[0])
        except TranscriptError as exc:
            sys.exit(f"Error: {exc}")
        print(f"Indexed '{lecture.title}' as {index_transcript(lecture)} chunks.")
    elif command == "list" and not args:
        for lec in list_lectures():
            print(f"{lec['chunks']:>4} chunks  {lec['video_id']}  {lec['title']}")
    elif command == "search" and args:
        for chunk, score in semantic_search(" ".join(args)):
            print(f"\n{score:.3f}  [{chunk.timestamp}] {chunk.title}\n       {chunk.url or ''}\n       {chunk.text[:200]}...")
    else:
        sys.exit(usage)
