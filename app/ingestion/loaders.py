import os
from typing import List
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_core.documents import Document


def load_document(file_path: str) -> List[Document]:
    _, extension = os.path.splitext(file_path)
    extension = extension.lower()

    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
    elif extension == ".md":
        loader = UnstructuredMarkdownLoader(file_path)
    elif extension == ".txt":
        loader = TextLoader(file_path)
    else:
        print(f"Unsupported file type: {extension}")
        return []

    return loader.load()