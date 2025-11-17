# src/corpora/build_ewha_corpus.py
"""
Ewha 학칙 PDF(ewha.pdf)를 조문 단위로 파싱하여
JSONL 코퍼스를 생성하는 버전 (조문 전체 → 하나의 Document)
"""

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

try:
    import pdfplumber
except ImportError:
    pdfplumber = None


# -----------------------------
# 패턴 정의
# -----------------------------

# 조문 헤더 예:
#   제26조(휴학)
ARTICLE_RE = re.compile(r"^제\s*(\d+)조\s*\((.+?)\)")

# ANNEX (별표 / 부칙)
ANNEX_RE = re.compile(r"^(별표\s*\d+|부칙)")

@dataclass
class EwhaArticle:
    """조문 전체 단위 Document"""
    doc_id: str          # "제26조"
    section: str         # "제26조(휴학)"
    article_no: int      # 26
    text: str            # 조문 전체 텍스트(항/번호 포함)


# -----------------------------
# PDF → 줄 단위 텍스트
# -----------------------------
def extract_text_lines(pdf_path: Path) -> List[str]:
    if pdfplumber is None:
        raise ImportError("pdfplumber가 필요합니다. pip install pdfplumber")

    lines = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            for raw in text.splitlines():
                line = raw.replace("\u00a0", " ").strip()
                if line:
                    lines.append(line)
    return lines


# -----------------------------
# 조문 단위 파싱
# -----------------------------
def parse_articles(lines: List[str]) -> List[EwhaArticle]:
    docs: List[EwhaArticle] = []

    current_article_no: Optional[int] = None
    current_section_title: Optional[str] = None
    current_buffer: List[str] = []

    def flush_article():
        """현재 조문 전체를 하나의 Document로 저장."""
        nonlocal current_article_no, current_section_title, current_buffer

        if (
            current_article_no is None
            or current_section_title is None
            or not current_buffer
        ):
            return

        text = "\n".join(current_buffer).strip()
        if not text:
            return

        doc_id = f"제{current_article_no}조"
        docs.append(
            EwhaArticle(
                doc_id=doc_id,
                section=current_section_title,
                article_no=current_article_no,
                text=text,
            )
        )
        current_buffer = []

    for line in lines:

        # 1) 조문 헤더인지 확인
        m_article = ARTICLE_RE.match(line)
        if m_article:
            # 이전 조문 flush
            flush_article()

            # 새 조문 시작
            current_article_no = int(m_article.group(1))
            article_name = m_article.group(2).strip()
            current_section_title = f"제{current_article_no}조({article_name})"
            current_buffer = []
            continue

        # 2) 별표 / 부칙 → 원하는 경우 조문으로 넣거나 스킵
        if ANNEX_RE.match(line):
            # 일단 조문으로 처리하지 않고 건너뛰는 방식
            continue

        # 3) 조문 내부 텍스트 누적
        if current_article_no is not None:
            current_buffer.append(line)

    # 마지막 조문 flush
    flush_article()

    return docs


# -----------------------------
# 메인 빌드 함수
# -----------------------------
def build_ewha_corpus(pdf_path: Path, output_path: Path):
    print(f"[INFO] PDF 읽는 중: {pdf_path}")
    lines = extract_text_lines(pdf_path)
    print(f"[INFO] 총 {len(lines)}줄 추출")

    print("[INFO] 조문 단위 파싱 시작")
    docs = parse_articles(lines)
    print(f"[INFO] 총 {len(docs)}개 조문 생성")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(asdict(d), ensure_ascii=False) + "\n")

    print(f"[DONE] 코퍼스 저장 완료 → {output_path}")


# -----------------------------
# 실행 (CLI)
# -----------------------------
if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]  # 프로젝트 루트
    pdf_path = ROOT / "data" / "ewha.pdf"
    output_path = ROOT / "data" / "ewha_corpus.jsonl"

    build_ewha_corpus(pdf_path, output_path)
