"""BM25 search over data/processed/chunks.json."""

import json
import math
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "chunks.json"
RESULT_FIELDS = (
    "chunk_id",
    "score",
    "title",
    "course",
    "doc_type",
    "source_file",
    "text",
)
INPUT_FIELDS = (
    "chunk_id",
    "title",
    "course",
    "doc_type",
    "source_file",
    "text",
)

_chunks: list[dict[str, Any]] | None = None
_bm25: BM25Okapi | None = None
_bm25_k1: float | None = None
_bm25_b: float | None = None


def _load_index(k1: float, b: float) -> tuple[list[dict[str, Any]], BM25Okapi]:
    global _chunks, _bm25, _bm25_k1, _bm25_b

    if (
        _chunks is not None
        and _bm25 is not None
        and _bm25_k1 == k1
        and _bm25_b == b
    ):
        return _chunks, _bm25

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"BM25 data file not found: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8") as data_file:
        loaded_chunks = json.load(data_file)

    if not isinstance(loaded_chunks, list):
        raise ValueError("BM25 data must be a JSON list of chunks")

    validated_chunks: list[dict[str, Any]] = []
    tokenized_chunks: list[list[str]] = []
    for index, chunk in enumerate(loaded_chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk at index {index} must be a JSON object")

        tokens = chunk.get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(token, str) for token in tokens
        ):
            raise ValueError(f"Chunk at index {index} must contain string tokens")

        missing_fields = [field for field in INPUT_FIELDS if field not in chunk]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"Chunk at index {index} is missing: {missing}")

        validated_chunks.append(chunk)
        tokenized_chunks.append(tokens)

    if not tokenized_chunks:
        raise ValueError("BM25 data must contain at least one chunk")

    _chunks = validated_chunks
    _bm25 = BM25Okapi(tokenized_chunks, k1=k1, b=b)
    _bm25_k1 = k1
    _bm25_b = b
    return _chunks, _bm25


def search_bm25(
    query: str,
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict[str, Any]]:
    """Return the highest-scoring chunks for a query."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if k1 < 0:
        raise ValueError("k1 must be non-negative")
    if b < 0 or b > 1:
        raise ValueError("b must be between 0 and 1")

    chunks, bm25 = _load_index(k1=k1, b=b)
    query_tokens = query.lower().split()
    scores = bm25.get_scores(query_tokens)
    ranked_indexes = sorted(
        range(len(chunks)),
        key=lambda index: scores[index],
        reverse=True,
    )[:top_k]

    return [
        {
            "chunk_id": chunk["chunk_id"],
            "score": float(scores[index]),
            "title": chunk["title"],
            "course": chunk["course"],
            "doc_type": chunk["doc_type"],
            "source_file": chunk["source_file"],
            "text": chunk["text"],
        }
        for index in ranked_indexes
        for chunk in [chunks[index]]
    ]


def evaluate_bm25(
    judgments: list[dict[str, Any]],
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
) -> dict[str, float]:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    if not isinstance(judgments, list):
        raise ValueError("judgments must be a JSON list")

    ndcg_scores: list[float] = []
    precision_scores: list[float] = []
    average_precision_scores: list[float] = []

    for judgment in judgments:
        if not isinstance(judgment, dict):
            raise ValueError("Each judgment must be a JSON object")
        query = judgment.get("query")
        relevant_chunks = judgment.get("relevant_chunks")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Each judgment must contain a non-empty query")
        if not isinstance(relevant_chunks, list) or not all(
            isinstance(chunk_id, str) for chunk_id in relevant_chunks
        ):
            raise ValueError("Each judgment must contain relevant_chunks")

        ranked_chunk_ids = [
            result["chunk_id"]
            for result in search_bm25(query, top_k, k1=k1, b=b)
        ]
        relevant_set = set(relevant_chunks)
        relevance_labels = {
            chunk_id: 1 for chunk_id in relevant_chunks
        }

        retrieved = [
            relevance_labels.get(chunk_id, 0)
            for chunk_id in ranked_chunk_ids
        ]
        ideal = sorted(relevance_labels.values(), reverse=True)[:top_k]
        dcg = sum(
            relevance / math.log2(rank + 2)
            for rank, relevance in enumerate(retrieved[:top_k])
        )
        ideal_dcg = sum(
            relevance / math.log2(rank + 2)
            for rank, relevance in enumerate(ideal)
        )
        ndcg_scores.append(dcg / ideal_dcg if ideal_dcg else 0.0)

        relevant_retrieved = sum(
            chunk_id in relevant_set
            for chunk_id in ranked_chunk_ids[:top_k]
        )
        precision_scores.append(relevant_retrieved / top_k)

        relevant_count = 0
        precision_sum = 0.0
        for rank, chunk_id in enumerate(ranked_chunk_ids, start=1):
            if chunk_id in relevant_set:
                relevant_count += 1
                precision_sum += relevant_count / rank
        average_precision_scores.append(
            precision_sum / len(relevant_set) if relevant_set else 0.0
        )

    if not judgments:
        return {"ndcg": 0.0, "precision": 0.0, "map": 0.0}

    count = len(judgments)
    return {
        "ndcg": sum(ndcg_scores) / count,
        "precision": sum(precision_scores) / count,
        "map": sum(average_precision_scores) / count,
    }


def format_bm25_metrics(
    metrics: dict[str, float],
    top_k: int = 10,
    k1: float = 1.5,
    b: float = 0.75,
) -> str:
    return (
        f"k1={k1:.1f}; b={b:.1f}; "
        f"NDCG@{top_k}: {metrics['ndcg']:.4f}; "
        f"Precision@{top_k}: {metrics['precision']:.4f}; "
        f"MAP: {metrics['map']:.4f}"
    )
