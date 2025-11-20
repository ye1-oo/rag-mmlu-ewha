import re
from typing import Literal

from langdetect import detect, LangDetectException

from src.parsing.parse_testset import Example

# Ewha 관련 키워드 (조금 확장)
EWHA_KEYWORDS = [
    "이화", "이대", "여자대학교",
    "학점", "수강", "졸업", "이수",
    "입학", "정원", "학부", "전공", "교양",
    "복수전공", "부전공", "휴학", "복학",
    "성적", "평점", "재입학", "유급",
    "학기", "수강신청", "재학생", "신입생",
]

# MMLU 도메인 키워드(초기 간단 버전, 나중에 LLM 기반으로 교체 가능)
DOMAIN_KEYWORDS = {
    "law": [
        "law", "legal", "constitution", "court", "crime", "liability",
        "contract", "rights", "justice", "penalty", "statute", "tort"
    ],
    "psychology": [
        "psychology", "cognitive", "memory", "behavior", "experiment",
        "freud", "conditioning", "perception", "emotion", "personality"
    ],
    "business": [
        "market", "business", "economics", "finance", "company", "industry",
        "management", "marketing", "firm", "profit", "cost", "revenue"
    ],
    "philosophy": [
        "philosophy", "ethics", "reason", "logic", "kant", "aristotle",
        "morality", "metaphysics", "epistemology", "utilitarianism"
    ],
    "history": [
        "history", "war", "empire", "revolution", "ancient", "medieval",
        "treaty", "dynasty", "kingdom", "colonial", "independence"
    ],
}


# ---------- 공통 유틸 함수 ----------

def contains_korean(text: str) -> bool:
    """문자열에 한국어(한글) 포함 여부 확인."""
    return bool(re.search(r'[\uac00-\ud7a3]', text))


def contains_english(text: str) -> bool:
    """문자열에 알파벳 포함 여부 확인."""
    return bool(re.search(r'[A-Za-z]', text))


def detect_language_safe(text: str) -> str:
    """langdetect를 안전하게 감싸는 헬퍼."""
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
    except Exception:
        return "unknown"


# ---------- 소스 라우팅 ----------

def route_source(example: Example) -> Literal["ewha", "mmlu"]:
    """
    문제를 Ewha 학칙 문제인지, MMLU 문제인지 구분한다.

    규칙 우선순위:
    1) Ewha 관련 키워드 포함 → ewha
    2) 한글 포함 → ewha   (MMLU-Pro는 영어 기반이므로)
    3) langdetect 결과가 en → mmlu
    4) 그 외 애매하면 기본값 ewha (필요시 mmlu로 변경 가능)
    """
    text = example.raw
    q = example.question
    q_lower = q.lower()

    # 1) Ewha 키워드 우선
    if any(kw in text for kw in EWHA_KEYWORDS):
        return "ewha"

    # 2) 한글 포함 여부
    if contains_korean(text):
        # Ewha 학칙 문제일 확률이 매우 높음
        return "ewha"

    # 3) 영어 포함 여부 & langdetect
    if contains_english(text):
        lang = detect_language_safe(q)
        if lang == "en":
            return "mmlu"

    # 4) 기본값 (지금 데이터셋은 거의 Ewha이므로 ewha로 두는게 안전)
    return "ewha"


# ---------- MMLU 도메인 분류 ----------

def classify_domain(example: Example) -> str:
    """
    MMLU 문제의 도메인을 분류한다.
    초기 버전은 keyword 기반, 나중에 LLM 분류기로 교체 가능.

    반환: "law", "psychology", "business", "philosophy", "history" 중 하나
    """
    text = (example.question + " " + " ".join(example.options.values())).lower()

    # 1차: 키워드 매칭 기반
    domain_scores = {domain: 0 for domain in DOMAIN_KEYWORDS.keys()}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                domain_scores[domain] += 1

    # 어떤 도메인도 키워드를 못 찾으면 기본값 history
    if all(score == 0 for score in domain_scores.values()):
        return "history"

    # 점수가 가장 높은 도메인 선택
    best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
    return best_domain