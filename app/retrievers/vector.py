from app.ingestion.embed import get_vectorstore

def build_vector_retriever():
    vectordb = get_vectorstore()
    return vectordb.as_retriever(search_kwargs={"k": 10})