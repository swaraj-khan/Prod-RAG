from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List

def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunked_documents = splitter.split_documents(documents)
    
    valid_chunks = []
    filtered = 0
    for doc in chunked_documents:
        content = doc.page_content.strip()
        if len(content) >= 100:
            valid_chunks.append(doc)
        else:
            filtered += 1
    if filtered > 0:
        print(f"Filtered {filtered} tiny chunks")

    for i, doc in enumerate(valid_chunks):
        doc.metadata["chunk_id"] = i

    return valid_chunks
