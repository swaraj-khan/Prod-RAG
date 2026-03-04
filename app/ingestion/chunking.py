from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunked_documents = splitter.split_documents(documents)

    # Add a unique chunk_id to each chunk
    for i, doc in enumerate(chunked_documents):
        doc.metadata["chunk_id"] = i

    return chunked_documents