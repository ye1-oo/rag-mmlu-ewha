# rag-mmlu/embedding.py

"""
임베딩 관련 유틸 모듈
- BGEM3EmbeddingFunction 을 한 번만 로드해서 재사용
- 텍스트 리스트 -> dense / sparse 임베딩 벡터 반환
"""

import numpy as np
import torch
from pymilvus import model

# 기본 디바이스 설정 (GPU 있으면 GPU 사용)
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# 전역 임베딩 모델 캐시
_embedding_model = None
_embedding_device = None
_embedding_use_fp16 = None


def get_embedding_model(
    device: str = DEFAULT_DEVICE,
    use_fp16: bool = True,
):
    """
    BGEM3EmbeddingFunction 모델을 전역으로 하나만 로딩해서 재사용.
    """
    global _embedding_model, _embedding_device, _embedding_use_fp16

    # 이미 로드된 모델이 있고 설정이 같으면 그대로 사용
    if (
        _embedding_model is not None
        and _embedding_device == device
        and _embedding_use_fp16 == use_fp16
    ):
        return _embedding_model

    # 새로 로드
    print(f"🔁 Loading BGEM3EmbeddingFunction on {device} (fp16={use_fp16})")
    _embedding_model = model.hybrid.BGEM3EmbeddingFunction(
        use_fp16=use_fp16,
        device=device,
    )
    _embedding_device = device
    _embedding_use_fp16 = use_fp16
    return _embedding_model


def generate_embeddings(
    texts,
    device: str = DEFAULT_DEVICE,
    use_fp16: bool = True,
):
    """
    텍스트(문자열 리스트)를 받아 dense / sparse 임베딩을 생성한다.

    반환:
        dense_embeddings: np.ndarray, shape = (N, D)
        sparse_embeddings: BGEM3에서 반환하는 sparse 표현 (Milvus에 그대로 넣기용)
    """
    if isinstance(texts, str):
        texts = [texts]

    emb_model = get_embedding_model(device=device, use_fp16=use_fp16)
    embeddings = emb_model(texts)

    dense_embeddings = np.array(embeddings["dense"], dtype=np.float32)
    sparse_embeddings = embeddings["sparse"]  # Milvus에서 그대로 사용 가능

    return dense_embeddings, sparse_embeddings
