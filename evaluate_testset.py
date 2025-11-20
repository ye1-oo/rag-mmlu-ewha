# evaluate_testset.py
"""
testset.csv + n_final.csv 평가 스크립트

- testset.csv: gold 정답 포함 (prompts, answers)
- n_final.csv: run.py가 생성한 예측 (qid, your_answer)

출력:
- 전체 정확도
- Ewha / MMLU 별 정확도
- (옵션) MMLU 도메인별 정확도
"""

import re
from collections import defaultdict

import pandas as pd

from src.parsing.parse_testset import load_examples_from_csv
from src.routing.router import route_source, classify_domain


TESTSET_PATH = "data/testset.csv"
PRED_PATH = "n_final.csv"


def normalize_choice(s):
    """
    '(D)', 'D', '**[ANSWER]: (C) ...**' 등에서
    선택지 문자 하나(A~J)만 안전하게 뽑아내는 유틸.
    """
    if not isinstance(s, str):
        return None

    s = s.strip().upper()
    # 괄호 안의 문자 우선
    m = re.search(r"\(([A-J])\)", s)
    if m:
        return m.group(1)

    # 그냥 문자 하나만 있는 경우
    m2 = re.search(r"\b([A-J])\b", s)
    if m2:
        return m2.group(1)

    return None


def main():
    print("Loading testset & predictions...")

    # 1) gold 포함 Example 리스트
    examples = load_examples_from_csv(TESTSET_PATH)

    # 2) 예측 파일 (run.py 결과)
    pred_df = pd.read_csv(PRED_PATH)
    # qid -> predicted choice 매핑 딕셔너리
    pred_map = {
        row["qid"]: normalize_choice(row["your_answer"])
        for _, row in pred_df.iterrows()
    }

    total = 0
    correct = 0

    # 소스별(Ewha / MMLU) 통계
    src_stats = {
        "ewha": {"total": 0, "correct": 0},
        "mmlu": {"total": 0, "correct": 0},
    }

    # 도메인별 통계 (MMLU용)
    domain_stats = defaultdict(lambda: {"total": 0, "correct": 0})

    missing_preds = []

    for ex in examples:
        gold = normalize_choice(ex.gold)  # load_examples_from_csv에서 이미 "D" 형태일 것
        pred = pred_map.get(ex.qid, None)

        if gold is None:
            # gold가 없는 경우는 스킵 (정상적이라면 거의 없을 것)
            continue

        if pred is None:
            # 예측 자체가 없으면 miss로 기록
            missing_preds.append(ex.qid)
            # 틀린 것으로 간주 (total은 올림, correct는 안 올림)
            src = route_source(ex)
            src_stats[src]["total"] += 1
            total += 1
            continue

        total += 1
        src = route_source(ex)
        src_stats[src]["total"] += 1

        if pred == gold:
            correct += 1
            src_stats[src]["correct"] += 1

        # MMLU인 경우 도메인 단위 통계도 집계
        if src == "mmlu":
            domain = classify_domain(ex)
            domain_stats[domain]["total"] += 1
            if pred == gold:
                domain_stats[domain]["correct"] += 1

    # -------------------- 결과 출력 --------------------

    if total == 0:
        print("평가할 샘플이 없습니다. testset/gold를 확인하세요.")
        return

    overall_acc = correct / total * 100.0

    print("\n================ EVALUATION RESULT ================")
    print(f"전체 샘플 수: {total}")
    print(f"전체 정답 수: {correct}")
    print(f"➡ 전체 정확도: {overall_acc:.2f}%")

    print("\n[소스별(Ewha / MMLU) 정확도]")
    for src, st in src_stats.items():
        if st["total"] == 0:
            print(f"- {src}: 샘플 없음")
            continue
        acc = st["correct"] / st["total"] * 100.0
        print(f"- {src}: {st['correct']} / {st['total']}  ({acc:.2f}%)")

    print("\n[MMLU 도메인별 정확도]")
    if not domain_stats:
        print("- MMLU 샘플 없음")
    else:
        for d, st in domain_stats.items():
            if st["total"] == 0:
                continue
            acc = st["correct"] / st["total"] * 100.0
            print(f"- {d}: {st['correct']} / {st['total']}  ({acc:.2f}%)")

    if missing_preds:
        print(f"\n⚠ 예측이 비어 있는 qid 개수: {len(missing_preds)}")
        print(f"  예시 일부: {missing_preds[:5]}")

    print("\n===================================================\n")


if __name__ == "__main__":
    main()