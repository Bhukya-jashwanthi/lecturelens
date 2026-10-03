# Evaluation results

Retrieval over 24 hand-labelled questions (top 5 chunks). A hit means a retrieved chunk overlaps the moment where the answer is spoken.

| Chunk window / overlap (chunks) | Method | Hit@1 | Hit@3 | Hit@5 | MRR@5 | Hit@5 paraphrase | Hit@5 keyword |
|---|---|---|---|---|---|---|---|
| 30s / 5s (39) | semantic | 71% | 92% | 100% | 0.82 | 100% | 100% |
| 30s / 5s (39) | keyword (BM25) | 75% | 92% | 96% | 0.83 | 93% | 100% |
| 30s / 5s (39) | hybrid (RRF) | 83% | 96% | 100% | 0.90 | 100% | 100% |
| 60s / 15s (23) | semantic | 58% | 88% | 100% | 0.75 | 100% | 100% |
| 60s / 15s (23) | keyword (BM25) | 71% | 79% | 88% | 0.76 | 86% | 90% |
| 60s / 15s (23) | hybrid (RRF) | 62% | 92% | 96% | 0.77 | 93% | 100% |
| 120s / 30s (12) | semantic | 67% | 88% | 92% | 0.77 | 86% | 100% |
| 120s / 30s (12) | keyword (BM25) | 79% | 92% | 96% | 0.85 | 93% | 100% |
| 120s / 30s (12) | hybrid (RRF) | 67% | 88% | 100% | 0.80 | 100% | 100% |

Missed by the default configuration (30s chunks, hybrid):

- none

## Answer grounding

- Out-of-scope questions correctly refused: **6/6**
- In-scope questions answered (not wrongly refused): **6/6**
