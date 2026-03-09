import os
import tempfile
import streamlit as st
from langchain_core.documents import Document
from datasets import Dataset

from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
)

from app.ingestion.loaders import load_document
from app.ingestion.chunking import chunk_documents
from app.ingestion.embed import ingest_documents, get_vectorstore, get_embeddings
from app.chains.retrieval_chain import RetrievalChain
from app.chains.rerank_chain import rerank
from app.chains.generation_chain import generate_answer, llm, langfuse

st.set_page_config(page_title="Prod-RAG", layout="wide")


def initialize_session_state():
    if "retrieval_chain" not in st.session_state:
        st.session_state.retrieval_chain = None
    if "messages" not in st.session_state:
        st.session_state.messages = []


def is_vectorstore_empty():
    vectordb = get_vectorstore()
    return vectordb._collection.count() == 0


def load_all_documents_from_vectorstore():
    vectordb = get_vectorstore()
    results = vectordb._collection.get(include=["documents", "metadatas"])

    docs = [
        Document(page_content=text, metadata=metadata)
        for text, metadata in zip(results["documents"], results["metadatas"])
    ]

    return docs


def process_uploaded_files(uploaded_files):
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_docs = []

        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)

            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            raw_docs.extend(load_document(file_path))

        if raw_docs:
            with st.spinner("Chunking and ingesting documents..."):
                chunked_docs = chunk_documents(raw_docs)
                ingest_documents(chunked_docs)

                st.success("Documents ingested successfully!")

                all_docs = load_all_documents_from_vectorstore()
                st.session_state.retrieval_chain = RetrievalChain(all_docs)


st.title("📄 Prod-RAG: Production-Ready RAG System")

initialize_session_state()

with st.sidebar:
    st.header("📚 Document Ingestion")

    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Ingest Documents"):
            process_uploaded_files(uploaded_files)


if not is_vectorstore_empty() and not st.session_state.retrieval_chain:
    with st.spinner("Loading documents and initializing retrieval chain..."):
        all_docs = load_all_documents_from_vectorstore()
        st.session_state.retrieval_chain = RetrievalChain(all_docs)


if st.session_state.retrieval_chain:
    st.info("Retrieval chain is ready. Ask your question below.")
else:
    st.warning("Please ingest documents to begin.")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if query := st.chat_input("Ask a question about your documents"):

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    if not st.session_state.retrieval_chain:
        st.error("The retrieval chain is not initialized. Please ingest documents first.")
        st.stop()

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            retrieved_docs = st.session_state.retrieval_chain.retrieve(query)

            reranked_docs = rerank(query, retrieved_docs)

            answer = generate_answer(query, reranked_docs)

            # Flush Langfuse traces to ensure they're sent
            langfuse.flush()

            st.markdown(answer)

            with st.expander("Show Evaluation Metrics"):

                st.write("Running RAGAS evaluation...")

                contexts = [d.page_content for d in reranked_docs]

                eval_data = {
                    "question": [query],
                    "answer": [answer],
                    "contexts": [contexts],
                }

                dataset = Dataset.from_dict(eval_data)

                metrics = [
                    Faithfulness(),
                    AnswerRelevancy(),
                ]

                embeddings = get_embeddings()

                result = evaluate(
                    dataset,
                    metrics=metrics,
                    llm=llm,
                    embeddings=embeddings,
                )

                st.dataframe(result.to_pandas())

    st.session_state.messages.append({"role": "assistant", "content": answer})