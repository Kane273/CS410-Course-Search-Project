"""TF-IDF search over data/processed/chunks.json."""

import json
import math
from pathlib import Path
from typing import Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "chunks.json"
INPUT_FIELDS = (
    "chunk_id",
    "title",
    "course",
    "doc_type",
    "source_file",
    "text",
)

_chunks: list[dict[str, Any]] | None = None
_vectorizer: TfidfVectorizer | None = None
_document_matrix: Any | None = None


def _load_index() -> tuple[
    list[dict[str, Any]],
    TfidfVectorizer,
    Any,
]:
    global _chunks, _vectorizer, _document_matrix

    if (
        _chunks is not None
        and _vectorizer is not None
        and _document_matrix is not None
    ):
        return _chunks, _vectorizer, _document_matrix

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"TF-IDF data file not found: {DATA_PATH}")

    with DATA_PATH.open("r", encoding="utf-8") as data_file:
        loaded_chunks = json.load(data_file)

    if not isinstance(loaded_chunks, list):
        raise ValueError("TF-IDF data must be a JSON list of chunks")

    validated_chunks: list[dict[str, Any]] = []
    texts: list[str] = []
    for index, chunk in enumerate(loaded_chunks):
        if not isinstance(chunk, dict):
            raise ValueError(f"Chunk at index {index} must be a JSON object")

        missing_fields = [field for field in INPUT_FIELDS if field not in chunk]
        if missing_fields:
            missing = ", ".join(missing_fields)
            raise ValueError(f"Chunk at index {index} is missing: {missing}")

        if not isinstance(chunk["text"], str):
            raise ValueError(f"Chunk at index {index} must contain string text")

        validated_chunks.append(chunk)
        texts.append(chunk["text"])

    if not texts:
        raise ValueError("TF-IDF data must contain at least one chunk")

    vectorizer = TfidfVectorizer()
    document_matrix = vectorizer.fit_transform(texts)

    _chunks = validated_chunks
    _vectorizer = vectorizer
    _document_matrix = document_matrix
    return _chunks, _vectorizer, _document_matrix


def search_tfidf(query: str, top_k: int = 10) -> list[dict[str, Any]]:
    """Return the highest-cosine-similarity chunks for a query."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")

    chunks, vectorizer, document_matrix = _load_index()
    query_matrix = vectorizer.transform([query])
    scores = cosine_similarity(query_matrix, document_matrix).flatten()
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


def evaluate_tfidf(
    judgments: list[dict[str, Any]],
    top_k: int = 10,
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
            for result in search_tfidf(query, top_k)
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


def format_tfidf_metrics(
    metrics: dict[str, float],
    top_k: int = 10,
) -> str:
    return (
        f"NDCG@{top_k}: {metrics['ndcg']:.4f}; "
        f"Precision@{top_k}: {metrics['precision']:.4f}; "
        f"MAP: {metrics['map']:.4f}"
    )