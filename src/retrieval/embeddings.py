# src/retrieval/embeddings.py
"""
BGE-M3 기반 임베딩 생성기
- query/text 임베딩 모두 지원
- retrieval.py와 vector_store.py에서 공통적으로 사용
"""

from typing import List
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class Embedder:
    """
    BGE-M3 임베딩 모델 래퍼
    - encode(): 여러 문장을 임베딩하여 numpy array로 반환
    """

    def __init__(self, model_name="BAAI/bge-m3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 토크나이저 + 모델 로딩
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

    @torch.no_grad()
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        texts: List[str] → 2D numpy array (N x D)
        """

        if isinstance(texts, str):
            texts = [texts]

        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(self.device)

        outputs = self.model(**inputs)
        # BGE-M3의 CLS 토큰 사용
        dense = outputs.last_hidden_state[:, 0]

        # L2 normalize
        dense = dense / dense.norm(dim=1, keepdim=True)
        return dense.cpu().numpy()
