import cohere
from app.config import COHERE_API_KEY

co = cohere.Client(COHERE_API_KEY)

def rerank(query, documents, top_n=5):
    response = co.rerank(
        query=query,
        documents=[d.page_content for d in documents],
        model="rerank-english-v3.0",
        top_n=top_n,
    )

    return [documents[r.index] for r in response.results]