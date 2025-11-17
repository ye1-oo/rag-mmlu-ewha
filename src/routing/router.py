import re
from langdetect import detect

# Ewha 관련 키워드 리스트
EWHA_KWS = [
    "이화", "이대", "여자대학교",
    "학점", "수강", "졸업", "이수",
    "입학", "정원", "학부", "전공", "교양",
    "복수전공", "부전공", "휴학", "복학",
]

def contains_korean(text: str) -> bool:
    """문자열에 한국어 포함 여부 확인"""
    return bool(re.search(r'[\uac00-\ud7a3]', text))

def select_data_source(question: str) -> str:
    """
    질문이 이화대 관련인지, Wikipedia 검색해야 하는지 결정하는 라우터
    규칙:
    1) 이화 키워드 포함 → ewha
    2) 한국어로 된 질문 → ewha
    3) 영어 포함 질문 → wikipedia
    """

    q_lower = question.lower()

    # 1) Ewha 키워드 포함 여부
    if any(kw in q_lower for kw in EWHA_KWS):
        return "ewha"

    # 2) 한국어 포함 여부
    if contains_korean(question):
        return "ewha"

    # 3) 언어 감지 후 영어면 wikipedia로
    try:
        lang = detect(question)
    except:
        lang = "unknown"

    if lang == "en":
        return "wikipedia"

    # 기본값 wikipedia
    return "wikipedia"
