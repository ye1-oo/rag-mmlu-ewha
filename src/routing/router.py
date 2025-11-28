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
        "invasion", "perpetrator", "law", "entering", "robbery", "burglary",
        "gun", "larceny", "apprehension", "possession", "monism", "dualism",
        "private", "jurisprudence", "morality"
    ],
    "psychology": [
        "child", "self-esteem", "mania", "symptoms", "therapist", "conflict", 
        "choosing", "Motivation", "disagreement"
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

def llm_classify_domain(question: str, options: dict) -> str:
    """
    Few-shot LLM 기반 도메인 분류
    """
    try:
        from src.llm.solver import call_solar
    except Exception:
        return None

    FEW_SHOT = """
### EXAMPLES ###

[Law Example 1]
Question: A municipality passed an ordinance requiring all privately owned drones to transmit real-time location data...
Domain: law

[Law Example 2]
Question: A company’s employment contract states that disputes must be resolved by arbitration...
Domain: law


[History Example 1]
Question: During the late 19th century, several Asian states pursued selective Westernization...
Domain: history

[History Example 2]
Question: A historian studying early trade networks discovers evidence that a coastal city used standardized clay tokens...
Domain: history


[Philosophy Example 1]
Question: A philosopher argues that moral responsibility requires the ability to have acted otherwise...
Domain: philosophy

[Philosophy Example 2]
Question: In a debate on personal identity, a theorist claims that identity persists because of psychological continuity...
Domain: philosophy


[Business Example 1]
Question: A startup uses surge-pricing algorithms that automatically increase the price of its service...
Domain: business

[Business Example 2]
Question: A multinational corporation wants to enter a market where consumers strongly prefer local brands...
Domain: business


[Psychology Example 1]
Question: A researcher exposes participants to emotionally charged images...
Domain: psychology

[Psychology Example 2]
Question: A clinical psychologist studies how children interpret ambiguous statements...
Domain: psychology
"""

    prompt = f"""
You are a domain classifier for multiple-choice academic questions.
Choose one domain from: law, psychology, business, philosophy, history.

Use the examples as reference.

{FEW_SHOT}

### TASK ###
Classify the following question:

Question:
{question}

Options:
{options}

Respond with only one domain name in lowercase.
"""

    try:
        output = call_solar(prompt).strip().lower()
        for d in ["law", "psychology", "business", "philosophy", "history"]:
            if output.startswith(d):
                return d
        return None
    except:
        return None



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
    순서:
      1) LLM 기반 domain 분류 시도
      2) 실패 시 → keyword 기반 fallback
    """

    # ---- 1) LLM 기반 분류 먼저 시도 ----
    llm_domain = llm_classify_domain(example.question, example.options)
    if llm_domain is not None:
        return llm_domain

    # ---- 2) fallback: keyword 기반 기존 로직 ----
    text = (example.question + " " + " ".join(example.options.values())).lower()

    domain_scores = {domain: 0 for domain in DOMAIN_KEYWORDS.keys()}

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                domain_scores[domain] += 1

    # fallback 기본값: history
    if all(score == 0 for score in domain_scores.values()):
        return "history"

    best_domain = max(domain_scores.items(), key=lambda x: x[1])[0]
    return best_domain
