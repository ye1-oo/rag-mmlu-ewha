## 사용 예시 코드! 
from src.retrieval.embeddings import Embedder
from src.retrieval.vector_store import FaissIndex
from src.retrieval.retriever import retrieve_ewha_context

embedder = Embedder()

ewha_index = FaissIndex(
    index_path="data/ewha_index.faiss",
    meta_path="data/ewha_corpus.jsonl"
)

res = retrieve_ewha_context(
    "휴학 신청 가능 기간은?",
    embedder,
    ewha_index,
    top_k=5
)

for r in res:
    print(r["score"], r["text"][:200])