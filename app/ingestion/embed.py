from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from app.config import GOOGLE_API_KEY, EMBEDDING_MODEL, CHROMA_DIR, COLLECTION_NAME

def get_embeddings():
    return GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY
    )

def get_vectorstore():
    embeddings = get_embeddings()
    vectordb = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )

    return vectordb

def ingest_documents(documents):
    vectordb = get_vectorstore()
    vectordb.add_documents(documents)
    vectordb.persist()