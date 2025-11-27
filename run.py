# run.py
"""
최종 제출용 실행 스크립트
- testset.csv → 2_final.csv 생성
- Ewha / MMLU 라우팅
- Retrieval (Ewha, External KB, Wikipedia)
- Multi-agent LLM Solver 호출
"""

import os
import csv
from tqdm import tqdm

from src.parsing.parse_testset import load_examples_from_csv, Example
from src.routing.router import route_source, classify_domain
from src.retrieval.embeddings import Embedder
from src.retrieval.vector_store import FaissIndex
from src.retrieval.retriever import (
    build_ewha_context,
    build_mmlu_context,
)
from src.llm.solver import solve_with_ensemble


# ---------------------------------------------------
# 1. 환경 설정
# ---------------------------------------------------

TESTSET_PATH = "data/testset.csv"    # 혹은 private.csv
# TESTSET_PATH = "data/private_test.csv" 
OUTPUT_PATH = "2_final.csv"          
# OUTPUT_PATH = "2_final_private.csv"  

EWHA_INDEX_PATH = "data/ewha_index_final.faiss" #최종 파일로 수정
# EWHA_META_PATH = "data/ewha_corpus.jsonl" 
EWHA_META_PATH ="data/ewha_corpus_final.jsonl" #최종 파일로 수정
MMLU_INDEX_PATH = "data/mmlu_kb_index.faiss"
MMLU_META_PATH = "data/mmlu_kb.jsonl"


# ---------------------------------------------------
# 2. 메인 실행 함수
# ---------------------------------------------------

def run():
    print("Loading embedder & indices...")
    embedder = Embedder()

    ewha_index = FaissIndex(EWHA_INDEX_PATH, EWHA_META_PATH)
    mmlu_index = FaissIndex(MMLU_INDEX_PATH, MMLU_META_PATH)

    print("Loading testset...")
    examples = load_examples_from_csv(TESTSET_PATH)

    outputs = []

    print("Running inference...")
    for ex in tqdm(examples):

        # 1) 라우팅 (Ewha / MMLU)
        source_mode = route_source(ex)
        # ex.question 은 Example 객체의 질문 텍스트

        # --------------------------------------------------------
        # Ewha 모드
        # --------------------------------------------------------
        if source_mode == "ewha":
            context = build_ewha_context(ex, embedder, ewha_index, top_k=8)

            result = solve_with_ensemble(
                example=ex,
                context=context,
                mode="ewha"
            )

        # --------------------------------------------------------
        # MMLU 모드
        # --------------------------------------------------------
        else:
            # 1) domain 분류
            domain = classify_domain(ex)

            # 2) retrieval
            context = build_mmlu_context(
                ex,
                domain,
                embedder,
                mmlu_index,
                kb_top_k=10,      # External KB에서 먼저 10개 정도
                total_top_k=8     # wiki+kb 합쳐서 최종 8개 context만 사용
            )

            # 3) solve
            result = solve_with_ensemble(
                example=ex,
                context=context,
                mode="mmlu",
                domain=domain,
            )

        final_choice = result["choice"]

        # 저장 포맷: (qid, your_answer)
        outputs.append([ex.qid, f"({final_choice})"])


    # --------------------------------------------------------
    # 3. 최종 CSV 저장
    # --------------------------------------------------------
    with open(OUTPUT_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["qid", "your_answer"])
        writer.writerows(outputs)

    print(f"Done! Output saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    run()