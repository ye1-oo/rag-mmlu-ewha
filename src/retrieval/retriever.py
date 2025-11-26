# src/retrieval/retriever.py

"""
Retrieval 모듈 - 최종 버전 (wikipediaapi 전용)

기능:
1. Ewha 학칙 전용 검색 (FAISS 기반)
2. External MMLU KB 검색 (FAISS 기반)
3. Wikipedia 검색 (wikipediaapi 기반, 타이틀 추론 방식)
4. Dense reranking (BGE-M3)
5. Ewha / MMLU용 Context Builder

주의:
- Wikipedia는 교수님이 지정한 wikipedia-api (wikipediaapi)만 사용한다.
"""

from typing import List, Dict
import re

import wikipediaapi

from .embeddings import Embedder
from .vector_store import FaissIndex


# ---------------------------------------------------------
# Wikipedia API 설정 (wikipediaapi)
# ---------------------------------------------------------

wiki = wikipediaapi.Wikipedia(
    language="en",
    user_agent="rag-mmlu-ewha/1.0 (https://github.com/ye1-oo)"
)


# ---------------------------------------------------------
# Ewha Retrieval
# ---------------------------------------------------------

def retrieve_ewha_context(
    question: str,
    embedder: Embedder,
    ewha_index: FaissIndex,
    top_k: int = 5
) -> List[Dict]:
    """
    이화 학칙 전용 검색 함수
    """
    q_emb = embedder.encode(question)[0]   # (D,)
    results = ewha_index.search(q_emb, top_k=top_k)
    return results


# ---------------------------------------------------------
# External KB 검색 (MMLU)
# ---------------------------------------------------------

def search_external_kb(
    question: str,
    domain: str,
    embedder: Embedder,
    kb_index: FaissIndex,
    top_k: int = 15  # 검색 범위를 좀 더 넓힘 (10 -> 15)
) -> List[Dict]:
    """
    외부 MMLU KB 검색 (FAISS)
    [수정 사항]
    - 도메인 필터링(if domain == ...)을 제거했습니다.
    - 질문과 의미적으로 가장 유사한 문서를 전체 DB에서 찾습니다.
    """
    q_emb = embedder.encode(question)[0]
    results = kb_index.search(q_emb, top_k=top_k)

    # [삭제됨] 도메인 강제 필터링 로직
    # if domain:
    #     results = [r for r in results if r.get("domain") == domain]

    return results


# ---------------------------------------------------------
# Wikipedia 타이틀 후보 추출기 (강화 버전)
# ---------------------------------------------------------

_STOPWORDS = {
    "what", "which", "that", "this", "these", "those",
    "who", "whom", "whose", "where", "when", "why", "how",
    "does", "do", "did", "is", "are", "was", "were", "be",
    "an", "a", "the", "of", "and", "or", "in", "on", "for",
    "to", "from", "with", "as", "by", "about"
}


def _tokenize_en(text: str) -> List[str]:
    """영어 알파벳/하이픈/공백으로 이루어진 토큰 시퀀스 추출."""
    # 예: "law of effect", "operant conditioning"
    tokens = re.findall(r"[A-Za-z][A-Za-z\-']*", text)
    return [t for t in tokens if t.strip()]


def extract_candidate_titles(question: str, domain: str | None = None,
                             max_candidates: int = 8) -> List[str]:
    """
    질문(및 도메인)에서 Wikipedia 페이지 타이틀로 사용할 만한 후보를 추출한다.

    전략:
    - 단어 토큰 → bigram / trigram 생성
    - 길이가 충분히 길고(stopword만으로 구성되지 않은) n-gram 우선 사용
    - 남는 자리는 중요한 단일 토큰으로 채움
    """

    text = question
    if domain:
        text = f"{question} {domain}"

    tokens = _tokenize_en(text)
    if not tokens:
        return []

    tokens_lower = [t.lower() for t in tokens]

    # 단일 토큰 후보 (길이>3 & not stopword)
    unigram_candidates: List[str] = []
    for t, t_low in zip(tokens, tokens_lower):
        if len(t) <= 3:
            continue
        if t_low in _STOPWORDS:
            continue
        unigram_candidates.append(t.title())

    # bigram / trigram 생성
    ngram_candidates: List[str] = []

    # bigram
    for i in range(len(tokens) - 1):
        gram = f"{tokens[i]} {tokens[i+1]}"
        gram_low = [tokens_lower[i], tokens_lower[i+1]]
        # 전부 stopword면 제외
        if all(g in _STOPWORDS for g in gram_low):
            continue
        if len(gram.replace(" ", "")) <= 4:
            continue
        ngram_candidates.append(gram.title())

    # trigram
    for i in range(len(tokens) - 2):
        gram = f"{tokens[i]} {tokens[i+1]} {tokens[i+2]}"
        gram_low = [tokens_lower[i], tokens_lower[i+1], tokens_lower[i+2]]
        if all(g in _STOPWORDS for g in gram_low):
            continue
        if len(gram.replace(" ", "")) <= 5:
            continue
        ngram_candidates.append(gram.title())

    # 우선순위: trigram/bigram → unigram
    all_candidates = ngram_candidates + unigram_candidates

    # 중복 제거 (순서 유지)
    seen = set()
    ordered: List[str] = []
    for c in all_candidates:
        if c in seen:
            continue
        seen.add(c)
        ordered.append(c)

    # domain 자체도 타이틀 후보로 추가 (예: "Philosophy")
    if domain:
        dom_title = domain.title()
        if dom_title not in seen:
            ordered.append(dom_title)

    return ordered[:max_candidates]


# ---------------------------------------------------------
# Wikipedia 검색 (wikipediaapi 이용)
# ---------------------------------------------------------

def search_wikipedia_chunks(
    query: str,
    domain: str,
    max_pages: int = 4
) -> List[Dict]:
    """
    wikipediaapi만 사용하여 Wikipedia chunk를 얻는 함수.

    wikipediaapi는 검색 API가 없기 때문에,
    - 질문에서 위키 타이틀 후보를 추출한 뒤
    - 각 후보에 대해 wiki.page(title)를 직접 호출하는 방식으로 구현한다.
    """

    chunks: List[Dict] = []

    # 1) 질문 + 도메인 기반 타이틀 후보 추출
    candidates = extract_candidate_titles(query, domain=domain, max_candidates=max_pages)

    if not candidates:
        return chunks

    # 2) 각 후보 타이틀에 대해 페이지 조회
    for title in candidates:
        page = wiki.page(title)

        if not page.exists():
            continue

        # 3) 페이지 텍스트를 문단 단위로 분할
        paragraphs = page.text.split("\n")

        for p in paragraphs:
            p = p.strip()

            # 너무 짧은 문장은 noise로 간주하고 제외
            if len(p) < 60:
                continue

            chunks.append({
                "text": p,
                "source": "wiki",
                "title": title
            })

    return chunks


# ---------------------------------------------------------
# Dense Reranking
# ---------------------------------------------------------

def rerank_by_embedding(
    question: str,
    candidates: List[Dict],
    embedder: Embedder,
    top_k: int = 6
) -> List[Dict]:
    """
    후보 문장(candidates)을 query와 임베딩 비교해 rerank한다.
    score = dot-product (normalized cosine similarity)
    """
    if len(candidates) == 0:
        return []

    q_emb = embedder.encode(question)[0]

    items = []
    for c in candidates:
        c_emb = embedder.encode(c["text"])[0]
        score = float((q_emb * c_emb).sum())
        items.append((c, score))

    # 점수 높은 순으로 정렬
    sorted_items = sorted(items, key=lambda x: x[1], reverse=True)

    reranked = []
    for (c, sc) in sorted_items[:top_k]:
        new_item = c.copy()
        new_item["rerank_score"] = sc
        reranked.append(new_item)

    return reranked


# ---------------------------------------------------------
# Ewha Context Builder
# ---------------------------------------------------------

def build_ewha_context(
    example,
    embedder: Embedder,
    ewha_index: FaissIndex,
    top_k: int = 12 # 더 많은 힌트조각 제공 
) -> str:
    """
    Ewha 학칙용 context builder:
    - question + options → 하나의 query로 사용
    - FAISS 검색 결과 상위 top_k 문장 join
    """

    # query = example.question + "\n" + " ".join(
    #     f"({k}) {v}" for k, v in example.options.items()
    # )

    # [수정 후] 질문만 사용
    query = example.question

    results = retrieve_ewha_context(query, embedder, ewha_index, top_k=top_k)

    context_text = "\n".join(r["text"] for r in results)
    return context_text


# ---------------------------------------------------------
# MMLU Context Builder (Wikipedia + External KB + rerank)
# ---------------------------------------------------------

def build_mmlu_context(
    example,
    domain: str,
    embedder: Embedder,
    kb_index: FaissIndex,
    kb_top_k: int = 10,
    total_top_k: int = 8
) -> str:
    """
    MMLU용 RAG context builder:
    1) Wikipedia 검색 (wikipediaapi + 타이틀 추론)
    2) External KB 검색 (FAISS)
    3) 둘 합친 뒤 dense rerank
    4) top_k만 모아서 context 생성
    """

    question = example.question

    # 1) Wikipedia 검색
    wiki_chunks = search_wikipedia_chunks(
        query=question,
        domain=domain,
        max_pages=4
    )

    # 2) External KB 검색
    kb_chunks = search_external_kb(
        question=question,
        domain=domain,
        embedder=embedder,
        kb_index=kb_index,
        top_k=kb_top_k
    )

    # candidates 합치기
    candidates = kb_chunks + wiki_chunks

    # 3) Rerank
    reranked = rerank_by_embedding(
        question=question,
        candidates=candidates,
        embedder=embedder,
        top_k=total_top_k
    )

    # 4) 묶어서 context 문자열 생성
    context = "\n".join(c["text"] for c in reranked)
    return context