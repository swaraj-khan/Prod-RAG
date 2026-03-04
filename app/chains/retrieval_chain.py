from app.retrievers.bm25 import build_bm25
from app.retrievers.vector import build_vector_retriever
from app.retrievers.hybrid import rrf_fusion


class RetrievalChain:
    def __init__(self, documents):
        self.bm25 = build_bm25(documents)
        self.vector = build_vector_retriever()

    def retrieve(self, query: str, top_k: int = 10):
        bm25_docs = self.bm25.invoke(query)
        vector_docs = self.vector.invoke(query)
        hybrid_docs = rrf_fusion(bm25_docs, vector_docs)

        return hybrid_docs[:top_k]