"""Unit tests for Average Average Rank (AAR) fusion."""

from __future__ import annotations

import pytest

from src.domain.services.aar_fusion_service import AARFusionService


def _make_list_a() -> list[dict]:
    return [
        {"chunk_id": "c1", "score": 0.9},
        {"chunk_id": "c2", "score": 0.8},
        {"chunk_id": "c3", "score": 0.7},
    ]


def _make_list_b() -> list[dict]:
    return [
        {"chunk_id": "c2", "score": 0.95},
        {"chunk_id": "c1", "score": 0.85},
        {"chunk_id": "c4", "score": 0.75},
    ]


def test_fusion_prefers_consistent_low_ranks() -> None:
    service = AARFusionService()
    fused = service.fuse(_make_list_a(), _make_list_b())

    # list_a: c1(1), c2(2), c3(3)
    # list_b: c2(1), c1(2), c4(3)
    # Missing penalties use len(list) + 1.
    # c1 ranks: 1 + 2 = avg 1.5
    # c2 ranks: 2 + 1 = avg 1.5 (tie broken by chunk_id)
    # c3 ranks: 3 + 4 = avg 3.5
    # c4 ranks: 4 + 3 = avg 3.5 (tie broken by chunk_id: c3 < c4)
    ids = [r.chunk_id for r in fused]
    assert ids == ["c1", "c2", "c3", "c4"]


def test_aar_score_is_average_rank() -> None:
    service = AARFusionService()
    fused = service.fuse(_make_list_a(), _make_list_b())

    by_id = {r.chunk_id: r for r in fused}
    assert by_id["c1"].aar_score == 1.5
    assert by_id["c2"].aar_score == 1.5
    assert by_id["c3"].aar_score == 3.5
    assert by_id["c4"].aar_score == 3.5


def test_single_list_preserves_order() -> None:
    service = AARFusionService()
    fused = service.fuse(_make_list_a())

    assert [r.chunk_id for r in fused] == ["c1", "c2", "c3"]
    assert fused[0].aar_score == 1.0


def test_empty_input_returns_empty() -> None:
    service = AARFusionService()
    assert service.fuse() == []


def test_empty_list_is_skipped() -> None:
    service = AARFusionService()
    fused = service.fuse(_make_list_a(), [])
    # Empty list contributes no ranks; result is the same as a single list.
    assert [r.chunk_id for r in fused] == ["c1", "c2", "c3"]


def test_missing_chunk_id_ignored() -> None:
    service = AARFusionService()
    fused = service.fuse(
        [{"chunk_id": "c1", "score": 0.9}, {"score": 0.8}],
        [{"chunk_id": "c2", "score": 0.95}],
    )

    ids = [r.chunk_id for r in fused]
    assert "c1" in ids
    assert "c2" in ids
