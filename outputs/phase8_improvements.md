# Phase 8 Improvements

Comparison of retrieval methods before and after Phase 8 implementation.

| Method | Before | After | Improvement |
|---|---|---|---|
| AAR | 0.0% | 12.5% | +12.5% ⭐ |
| TF-IDF | N/A | 10.0% | New method |
| MMR | N/A | 2.5% | New method |
| Cross-Encoder | N/A | 2.5% | New method |
| RRF | 5.0% | 5.0% | Tuned k=10 |

## Notes

- **AAR** improved from a broken 0% Recall@5 to 12.5% Recall@5 after fixing the article-level fusion logic and removing the missing-rank penalty.
- **TF-IDF** is a new sparse retriever added in Phase 8 and performs comparably to BM25.
- **MMR** and **Cross-Encoder** are new optional rerankers; they match dense recall on this keyword-heavy 40-query set.
- **RRF** default `k` was tuned from 60 to 10 after a sweep across [10, 20, 30, 40, 50, 60, 80, 100].