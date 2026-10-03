"""The RAG pipeline: retrieve lecture chunks, then answer with Gemini using only those chunks.

    question -> (condense, if there is chat history) -> hybrid search -> numbered sources
             -> LLM answer citing [n] -> our code turns [n] into timestamp links

Run directly:
    python -m lecturelens.rag_chain "what is the bias used for?"
    python -m lecturelens.rag_chain            (interactive chat with follow-up questions)
"""

import logging
import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

from lecturelens.config import get_settings
from lecturelens.prompts import ANSWER_PROMPT, CONDENSE_PROMPT, NOT_COVERED
from lecturelens.retriever import RetrievedChunk, hybrid_search
from lecturelens.retry import with_retries

logger = logging.getLogger(__name__)

# Matches citations like [2], [1, 3] or [1;3].
_CITATION = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_HISTORY_TURNS = 3  # previous Q&A pairs used to understand follow-up questions

SearchFn = Callable[..., list[RetrievedChunk]]


@dataclass
class Answer:
    question: str
    search_query: str  # what was actually searched (the condensed question)
    text: str  # Markdown answer with clickable timestamp citations
    raw_text: str  # the LLM's answer before citation rendering
    sources: list[RetrievedChunk] = field(default_factory=list)
    cited: list[int] = field(default_factory=list)  # 1-based numbers of the sources actually cited
    grounded: bool = True  # False when the lectures do not cover the question


def format_sources(results: Sequence[RetrievedChunk]) -> str:
    """Number the retrieved chunks so the LLM can cite them as [1], [2], ..."""
    return "\n\n".join(
        f"[{n}] {r.chunk.title} @ {r.chunk.timestamp}\n{r.chunk.text}" for n, r in enumerate(results, start=1)
    )


def render_citations(text: str, sources: Sequence[RetrievedChunk]) -> tuple[str, list[int]]:
    """Replace [n] markers with timestamp links. Numbers that match no source are dropped.

    Returns the rendered text and the sorted list of valid source numbers that were cited.
    """
    cited: set[int] = set()

    def replace(match: re.Match) -> str:
        links = []
        for number in (int(n) for n in re.split(r"[,;]", match.group(1))):
            if not 1 <= number <= len(sources):
                logger.warning("Dropped citation [%d]: only %d sources", number, len(sources))
                continue
            cited.add(number)
            chunk = sources[number - 1].chunk
            links.append(f"[▶ {chunk.timestamp}]({chunk.url})" if chunk.url else f"[{chunk.timestamp}]")
        return " ".join(links)

    rendered = _CITATION.sub(replace, text)
    rendered = re.sub(r"\)\[", ") [", rendered)  # space between adjacent links: [1][2]
    rendered = re.sub(r"[ \t]+([.,;:!?])", r"\1", rendered)  # tidy spaces left by dropped citations
    return rendered, sorted(cited)


def format_history(history: Sequence[tuple[str, str]]) -> str:
    return "\n".join(f"User: {q}\nAssistant: {a[:500]}" for q, a in history[-_HISTORY_TURNS:])


def get_llm() -> BaseChatModel:
    settings = get_settings()
    return ChatGoogleGenerativeAI(
        model=settings.chat_model,
        google_api_key=settings.google_api_key,
        reasoning_effort=settings.reasoning_effort,
    )


class LectureQA:
    """Answers questions about indexed lectures. The LLM and search function are injectable for testing."""

    def __init__(self, llm: BaseChatModel | None = None, search: SearchFn = hybrid_search, top_k: int | None = None):
        llm = llm or get_llm()
        self.search = search
        self.top_k = top_k or get_settings().top_k
        # LangChain Expression Language (LCEL): prompt | model | parser forms a runnable pipeline.
        self.answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
        self.condense_chain = CONDENSE_PROMPT | llm | StrOutputParser()

    def condense(self, question: str, history: Sequence[tuple[str, str]]) -> str:
        """Turn a follow-up ("explain that more simply") into a standalone search query."""
        if not history:
            return question
        query = with_retries(
            lambda: self.condense_chain.invoke({"history": format_history(history), "question": question})
        ).strip()
        return query or question

    def ask(
        self, question: str, history: Sequence[tuple[str, str]] = (), video_ids: list[str] | None = None
    ) -> Answer:
        search_query = self.condense(question, history)
        sources = self.search(search_query, k=self.top_k, video_ids=video_ids)
        if not sources:
            return Answer(question, search_query, "No lectures are indexed yet. Add a lecture first.", "", grounded=False)

        raw = with_retries(
            lambda: self.answer_chain.invoke({"sources": format_sources(sources), "question": search_query})
        ).strip()
        text, cited = render_citations(raw, sources)
        grounded = NOT_COVERED not in raw
        return Answer(question, search_query, text, raw, sources, cited, grounded)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    qa = LectureQA()

    def show(answer: Answer) -> None:
        if answer.search_query != answer.question:
            print(f"  (searched for: {answer.search_query})")
        print(f"\n{answer.text}\n")
        for n, r in enumerate(answer.sources, start=1):
            mark = "*" if n in answer.cited else " "
            print(f"  {mark}[{n}] {r.chunk.timestamp:>6}  sem #{r.semantic_rank or '-'}, kw #{r.keyword_rank or '-'}")

    if len(sys.argv) > 1:
        show(qa.ask(" ".join(sys.argv[1:])))
    else:
        print("Ask about your lectures (empty line to quit).")
        history: list[tuple[str, str]] = []
        while question := input("\nYou: ").strip():
            answer = qa.ask(question, history)
            show(answer)
            history.append((question, answer.raw_text))
