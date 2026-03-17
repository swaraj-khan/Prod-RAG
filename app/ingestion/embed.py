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
    valid_docs = []
    filtered = 0
    for doc in documents:
        content = doc.page_content.strip()
        if content and len(content) >= 50:
            valid_docs.append(doc)
        else:
            filtered += 1
    if filtered > 0:
        print(f"Filtered {filtered} invalid documents before embedding")
    if valid_docs:
        vectordb = get_vectorstore()
        vectordb.add_documents(valid_docs)
        vectordb.persist()
        print(f"Ingested {len(valid_docs)} valid documents")
    else:
        print("No valid documents to ingest")
