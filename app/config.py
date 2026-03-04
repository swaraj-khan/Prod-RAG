import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
COHERE_API_KEY = os.getenv("COHERE_API_KEY")

CHROMA_DIR = "./chroma"
COLLECTION_NAME = "prod_rag_docs"

EMBEDDING_MODEL = "gemini-embedding-001"
LLM_MODEL = "gemini-2.5-pro"