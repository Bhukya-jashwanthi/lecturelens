"""Evaluate LectureLens retrieval (and optionally answer grounding) on a labelled test set.

    python eval/evaluate.py             # retrieval: 3 methods x 3 chunk sizes
    python eval/evaluate.py --answers   # also check the LLM refuses out-of-scope questions

Each chunk size is indexed into its own temporary Chroma collection, so the app's
real index is never touched. Results are printed and saved to eval/results.md.
"""

import argparse
import json
import logging
import tempfile
from pathlib import Path

from langchain_core.embeddings import Embeddings

from lecturelens.evaluation import first_relevant_rank, hit_rate, mean_reciprocal_rank
from lecturelens.rag_chain import LectureQA
from lecturelens.retriever import hybrid_search, keyword_search
from lecturelens.transcript import load_transcript
from lecturelens.vectorstore import build_vectorstore, get_embeddings, index_transcript, semantic_search

EVAL_DIR = Path(__file__).parent
K = 5
CHUNK_CONFIGS = [(30, 5), (60, 15), (120, 30)]  # (window, overlap) in seconds
METHODS = {
    "semantic": lambda q, store: [c for c, _ in semantic_search(q, k=K, store=store)],
    "keyword (BM25)": lambda q, store: [c for c, _ in keyword_search(q, k=K, store=store)],
    "hybrid (RRF)": lambda q, store: [r.chunk for r in hybrid_search(q, k=K, store=store)],
}


class QueryCachingEmbeddings(Embeddings):
    """Embed each distinct question once, however many configurations reuse it."""

    def __init__(self, inner: Embeddings):
        self.inner = inner
        self.cache: dict[str, list[float]] = {}

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.inner.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if text not in self.cache:
            self.cache[text] = self.inner.embed_query(text)
        return self.cache[text]


def evaluate_retrieval(test_set: dict) -> list[dict]:
    transcript = load_transcript(test_set["video_id"])
    questions = test_set["questions"]
    embeddings = QueryCachingEmbeddings(get_embeddings())
    rows = []

    # On Windows, Chroma keeps its files open until the process exits, so deleting the
    # temp folder can fail; ignoring that is safe (the OS cleans its temp directory).
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        for window, overlap in CHUNK_CONFIGS:
            store = build_vectorstore(embeddings, Path(tmp) / f"w{window}", f"eval_w{window}")
            n_chunks = index_transcript(transcript, store, window, overlap)
            for method, retrieve in METHODS.items():
                ranks = [first_relevant_rank(retrieve(q["q"], store), q["answer"]) for q in questions]
                by_type = {
                    t: hit_rate([r for r, q in zip(ranks, questions) if q["type"] == t], K)
                    for t in ("paraphrase", "keyword")
                }
                rows.append(
                    {
                        "chunks": f"{window}s / {overlap}s ({n_chunks})",
                        "method": method,
                        "hit@1": hit_rate(ranks, 1),
                        "hit@3": hit_rate(ranks, 3),
                        "hit@5": hit_rate(ranks, K),
                        "mrr": mean_reciprocal_rank(ranks, K),
                        "paraphrase": by_type["paraphrase"],
                        "keyword": by_type["keyword"],
                        "misses": [q["q"] for r, q in zip(ranks, questions) if r is None],
                    }
                )
                print(f"  {rows[-1]['chunks']:<16} {method:<15} hit@5={rows[-1]['hit@5']:.0%}  MRR={rows[-1]['mrr']:.2f}")
    return rows


def evaluate_answers(test_set: dict) -> dict:
    """Out-of-scope questions should be refused; in-scope ones should be answered."""
    qa = LectureQA()
    ids = [test_set["video_id"]]
    out_of_scope = test_set["out_of_scope"]
    in_scope = [q["q"] for q in test_set["questions"]][:: len(test_set["questions"]) // len(out_of_scope)]

    refused = [q for q in out_of_scope if not qa.ask(q, video_ids=ids).grounded]
    answered = [q for q in in_scope if qa.ask(q, video_ids=ids).grounded]
    return {
        "correct_refusals": f"{len(refused)}/{len(out_of_scope)}",
        "hallucinated": [q for q in out_of_scope if q not in refused],
        "answered_in_scope": f"{len(answered)}/{len(in_scope)}",
        "wrongly_refused": [q for q in in_scope if q not in answered],
    }


def to_markdown(rows: list[dict], n_questions: int, answers: dict | None) -> str:
    lines = [
        "# Evaluation results",
        "",
        f"Retrieval over {n_questions} hand-labelled questions (top {K} chunks). "
        "A hit means a retrieved chunk overlaps the moment where the answer is spoken.",
        "",
        "| Chunk window / overlap (chunks) | Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Hit@5 paraphrase | Hit@5 keyword |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['chunks']} | {r['method']} | {r['hit@1']:.0%} | {r['hit@3']:.0%} | {r['hit@5']:.0%} "
            f"| {r['mrr']:.2f} | {r['paraphrase']:.0%} | {r['keyword']:.0%} |"
        )
    default = next(r for r in rows if r["chunks"].startswith("30s") and r["method"].startswith("hybrid"))
    lines += ["", "Missed by the default configuration (30s chunks, hybrid):", ""]
    lines += [f"- {q}" for q in default["misses"]] or ["- none"]
    if answers:
        lines += [
            "",
            "## Answer grounding",
            "",
            f"- Out-of-scope questions correctly refused: **{answers['correct_refusals']}**",
            f"- In-scope questions answered (not wrongly refused): **{answers['answered_in_scope']}**",
        ]
        lines += [f"  - hallucinated on: {q}" for q in answers["hallucinated"]]
        lines += [f"  - wrongly refused: {q}" for q in answers["wrongly_refused"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--answers", action="store_true", help="also evaluate LLM refusals (uses ~10 LLM calls)")
    args = parser.parse_args()

    test_set = json.loads((EVAL_DIR / "questions.json").read_text(encoding="utf-8"))
    print(f"Evaluating retrieval on {len(test_set['questions'])} questions...")
    rows = evaluate_retrieval(test_set)
    answers = None
    if args.answers:
        print("Evaluating answer grounding...")
        try:
            answers = evaluate_answers(test_set)
            print(f"  refused out-of-scope: {answers['correct_refusals']}, answered in-scope: {answers['answered_in_scope']}")
        except Exception as exc:  # e.g. daily API quota reached: keep the retrieval results
            print(f"  skipped: {type(exc).__name__}: {str(exc)[:150]}")

    report = to_markdown(rows, len(test_set["questions"]), answers)
    (EVAL_DIR / "results.md").write_text(report, encoding="utf-8")
    print(f"\n{report}")


if __name__ == "__main__":
    main()
