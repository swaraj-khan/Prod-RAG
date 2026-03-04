from langchain_community.retrievers import BM25Retriever

def build_bm25(docs):
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = 10
    return retriever