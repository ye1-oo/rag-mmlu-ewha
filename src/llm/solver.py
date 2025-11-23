# src/llm/solver.py

"""
LLM Solver 모듈
- Upstage Chat 모델 호출 (Solar Pro 2 등)
- 정답 파싱 (슬라이드의 parseable 포맷)
- Multi-agent + self-consistency 앙상블
"""

import os
import re
from typing import Optional, Dict, Any, List

import requests
import yaml
from dotenv import load_dotenv

from src.parsing.parse_testset import Example
from src.llm.prompts import (
    EWHA_AGENT_STRICT,
    EWHA_AGENT_COMPARE,
    EWHA_AGENT_SHORT,
    MMLU_AGENT_MAIN,
    MMLU_AGENT_ALT,
)

# .env 로드 (UPSTAGE_API_KEY, USER_AGENT 등)
load_dotenv()


# ------------------------------------------------
# 1. Upstage API 설정
# ------------------------------------------------

CONFIG_PATH = "configs.yaml"

if os.path.exists(CONFIG_PATH):
    # with open(CONFIG_PATH, "r") as f:
    #     _cfg = yaml.safe_load(f)
    with open("configs.yaml", "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)
else:
    # 기본값: 필요하면 프로젝트에서 수정
    _cfg = {
        "upstage": {
            "base_url": "https://api.upstage.ai/v1/chat/completions",
            # reasoning 모델만 아니라면 자유롭게 변경 가능
            "model_name": "solar-pro2",
        }
    }

UPSTAGE_API_KEY = os.getenv("UPSTAGE_API_KEY")
USER_AGENT = os.getenv("USER_AGENT", "rag-mmlu-ewha/0.1")
BASE_URL = _cfg["upstage"]["base_url"]
MODEL_NAME = _cfg["upstage"]["model_name"]  # configs.yaml에서 바꾸면 됨


def call_solar(prompt: str, temperature: float = 0.0, max_tokens: int = 512) -> str:
    """
    Upstage Chat 모델 호출 래퍼.
    - reasoning_effort 등의 reasoning 옵션은 사용하지 않는다 (슬라이드 규칙).
    - 모델 이름은 configs.yaml에서 자유롭게 지정 가능.
    """
    if UPSTAGE_API_KEY is None:
        raise RuntimeError("UPSTAGE_API_KEY is not set. Check your .env file.")

    headers = {
        "Authorization": f"Bearer {UPSTAGE_API_KEY}",
        "Content-Type": "application/json",
        "User-Agent": USER_AGENT,
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        # ⚠ reasoning 관련 파라미터는 사용하지 않음
    }

    resp = requests.post(BASE_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except Exception as e:
        raise RuntimeError(f"Unexpected response format from Upstage: {data}") from e

    return content


# ------------------------------------------------
# 2. 정답 파싱 (extract 전략 강화)
# ------------------------------------------------

ANSWER_LINE_PATTERN = re.compile(
    r"\*\*\[ANSWER\]\s*:\s*\(([A-J])\)", re.IGNORECASE
)
CHOICE_PATTERN = re.compile(r"\(([A-J])\)")


def parse_choice(text: str) -> Optional[str]:
    """
    LLM 출력에서 최종 선택지 문자(A~J)를 robust하게 파싱.
    1) [ANSWER]: (X) 줄을 먼저 찾고,
    2) 없으면 전체 텍스트에서 마지막 (X)를 사용,
    3) 그래도 안 되면 'answer: X' 패턴을 찾는다.
    """

    # 1) [ANSWER]: (X) 라인 우선
    m = ANSWER_LINE_PATTERN.search(text)
    if m:
        return m.group(1).upper()

    # 2) fallback: 전체 텍스트에서 (A)~(J) 패턴 중 마지막 것
    matches = CHOICE_PATTERN.findall(text)
    if matches:
        return matches[-1].upper()

    # 3) 그래도 실패하면 "answer: X" 패턴 탐색
    m2 = re.search(r"answer[^A-Za-z0-9]*([A-J])", text, flags=re.IGNORECASE)
    if m2:
        return m2.group(1).upper()

    return None


# ------------------------------------------------
# 3. 프롬프트 렌더링 유틸
# ------------------------------------------------

def format_options(example: Example) -> str:
    return "\n".join(
        f"({k}) {v}" for k, v in example.options.items()
    )


def render_prompt_ewha(example: Example, context: str, template: str) -> str:
    return template.format(
        context=context,
        question=example.question,
        options=format_options(example),
    )


def render_prompt_mmlu(example: Example, context: str, template: str) -> str:
    return template.format(
        context=context,
        question=example.question,
        options=format_options(example),
    )


# ------------------------------------------------
# 4. Multi-agent + self-consistency 앙상블
# ------------------------------------------------

def solve_with_ensemble(
    example: Example,
    context: str,
    mode: str,
    domain: Optional[str] = None,
    temperatures: List[float] = (0.0, 0.3, 0.7),
) -> Dict[str, Any]:
    """
    mode: "ewha" 또는 "mmlu"
    domain: mmlu인 경우 "law", "psychology", "business", "philosophy", "history" 중 하나

    반환:
        {
          "choice": "C",
          "votes": {"C": 3, "B": 1, ...},
          "raw_preds": [... 각 에이전트/온도별 원본 응답 ...]
        }
    """

    preds: List[Dict[str, Any]] = []

    if mode == "ewha":
        templates = [
            ("ewha_strict", EWHA_AGENT_STRICT),
            ("ewha_compare", EWHA_AGENT_COMPARE),
            ("ewha_short", EWHA_AGENT_SHORT),
        ]
        render_fn = render_prompt_ewha
    else:
        assert domain is not None, "MMLU 모드에서는 domain이 필요합니다."
        templates = [
            (f"mmlu_prof_{domain}", MMLU_AGENT_MAIN[domain]),
            (f"mmlu_solve_{domain}", MMLU_AGENT_ALT[domain]),
        ]
        render_fn = render_prompt_mmlu

    # 각 템플릿 × 여러 temperature 조합으로 호출
    for tmpl_id, tmpl in templates:
        for temp in temperatures:
            prompt = render_fn(example, context, tmpl)
            try:
                answer_text = call_solar(prompt, temperature=temp)
            except Exception as e:
                # API 에러 시에도 파이프라인이 완전히 죽지 않도록
                preds.append(
                    {
                        "choice": None,
                        "answer_text": f"[ERROR] {e}",
                        "temperature": temp,
                        "template_id": tmpl_id,
                    }
                )
                continue

            choice = parse_choice(answer_text)

            preds.append(
                {
                    "choice": choice,
                    "answer_text": answer_text,
                    "temperature": temp,
                    "template_id": tmpl_id,
                }
            )

    # 다수결 투표
    vote_counter: Dict[str, int] = {}
    for p in preds:
        c = p["choice"]
        if c is None:
            continue
        vote_counter[c] = vote_counter.get(c, 0) + 1

    if not vote_counter:
        # 모든 파싱 실패 시, 첫 번째로 파싱된 choice를 사용
        fallback_choice = next(
            (p["choice"] for p in preds if p["choice"] is not None),
            None,
        )
        final_choice = fallback_choice
    else:
        final_choice = max(vote_counter.items(), key=lambda x: x[1])[0]

    return {
        "choice": final_choice,
        "votes": vote_counter,
        "raw_preds": preds,
    }