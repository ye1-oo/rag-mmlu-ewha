# src/retrieval/retriever.py
# LLM Router와 B가 직접 호출하는 핵심 검색 엔진
# Ewha/MMLU 검색 로직
"""
Retrieval 모듈
- 이화 학칙 corpus 검색
- 외부 MMLU KB 검색
- dense reranking
"""

from typing import List, Dict
from .embeddings import Embedder
from .vector_store import FaissIndex


def retrieve_ewha_context(
    question: str,
    embedder: Embedder,
    ewha_index: FaissIndex,
    top_k=5
) -> List[Dict]:
    """
    이화 학칙 전용 검색 함수
    입력: question(text)
    출력: [{"text":..., "score":..., ...}, ...]
    """

    q_emb = embedder.encode(question)[0]   # (D,)
    results = ewha_index.search(q_emb, top_k=top_k)
    return results


def search_external_kb(
    question: str,
    domain: str,
    embedder: Embedder,
    kb_index: FaissIndex,
    top_k=10
) -> List[Dict]:
    """
    외부 MMLU KB 검색
    (domain이 주어지면 filtering, 아니면 전체에서 검색)
    """

    q_emb = embedder.encode(question)[0]
    results = kb_index.search(q_emb, top_k=top_k)

    # 도메인 필터링 옵션 (라우터가 domain 판별해줄 때)
    if domain:
        results = [r for r in results if r.get("domain") == domain]

    return results


def rerank_by_embedding(
    question: str,
    candidates: List[Dict],
    embedder: Embedder,
    top_k=6
) -> List[Dict]:
    """
    후보 문장을 다시 query와 임베딩 비교하여 rerank
    """

    if len(candidates) == 0:
        return []

    q_emb = embedder.encode(question)[0]

    scores = []
    for c in candidates:
        c_emb = embedder.encode(c["text"])[0]
        score = float((q_emb * c_emb).sum())
        scores.append(score)

    # score 높은 순 정렬
    sorted_items = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    reranked = []
    for item, sc in sorted_items[:top_k]:
        new_item = item.copy()
        new_item["rerank_score"] = sc
        reranked.append(new_item)

    return reranked
