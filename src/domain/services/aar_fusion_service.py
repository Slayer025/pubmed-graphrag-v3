"""Pure-domain Average Average Rank (AAR) fusion service.

AAR is a simple, parameter-free rank aggregation method.  For each chunk it
computes the average of the chunk's ranks across all input retrieval systems,
treating missing chunks as having the worst rank in that list plus one.  The
final fused list is sorted by ascending average rank (lower is better).

Unlike RRF, AAR does not require a damping constant ``k`` and is insensitive
to the absolute score ranges produced by each retriever.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AARResult:
    """One fused result produced by the AAR service."""

    chunk_id: str
    aar_score: float
    metadata: dict[str, Any]


class AARFusionService:
    """Fuse multiple ranked result lists using Average Average Rank."""

    def fuse(
        self,
        *result_lists: list[dict[str, Any]],
    ) -> list[AARResult]:
        """Merge ranked lists and score by average rank.

        Args:
            result_lists: One or more ranked lists. Each inner list is assumed
                to be ordered from best (rank 1) to worst. Each item must be a
                mapping containing at least ``chunk_id`` and ``score`` keys.
                The ``score`` value is preserved in ``metadata`` but is not used
                for ranking.

        Returns:
            List of ``AARResult`` objects sorted by ascending ``aar_score``.
            The ``aar_score`` field stores the average rank; lower is better.
        """
        if not result_lists:
            return []

        # First pass: discover every chunk id and capture metadata.
        all_chunk_ids: set[str] = set()
        metadata: dict[str, dict[str, Any]] = {}

        for ranked_list in result_lists:
            for item in ranked_list:
                chunk_id = str(item.get("chunk_id", ""))
                if not chunk_id:
                    continue
                all_chunk_ids.add(chunk_id)
                if chunk_id not in metadata:
                    metadata[chunk_id] = dict(item)

        if not all_chunk_ids:
            return []

        # Second pass: compute per-list ranks and apply missing penalties.
        ranks_by_chunk: dict[str, list[int]] = {cid: [] for cid in all_chunk_ids}

        for ranked_list in result_lists:
            if not ranked_list:
                # Empty lists contribute nothing; all chunks remain unpenalised.
                continue

            list_len = len(ranked_list)
            seen_in_list: set[str] = set()
            for rank, item in enumerate(ranked_list, start=1):
                chunk_id = str(item.get("chunk_id", ""))
                if not chunk_id:
                    continue
                seen_in_list.add(chunk_id)
                ranks_by_chunk[chunk_id].append(rank)

            # Penalise chunks missing from this list with worst-rank + 1.
            missing_rank = list_len + 1
            for chunk_id in all_chunk_ids:
                if chunk_id not in seen_in_list:
                    ranks_by_chunk[chunk_id].append(missing_rank)

        num_lists = len(result_lists)
        fused = [
            AARResult(
                chunk_id=chunk_id,
                aar_score=round(sum(ranks) / num_lists, 4),
                metadata=metadata[chunk_id],
            )
            for chunk_id, ranks in ranks_by_chunk.items()
        ]
        return sorted(fused, key=lambda r: (r.aar_score, r.chunk_id))
