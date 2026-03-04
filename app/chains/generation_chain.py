import os
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import GOOGLE_API_KEY, LLM_MODEL
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.output_parsers import StrOutputParser

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL, google_api_key=GOOGLE_API_KEY, temperature=0
)


def load_system_prompt():
    script_dir = os.path.dirname(__file__)
    prompt_path = os.path.join(script_dir, "..", "prompts", "v1.yaml")
    with open(prompt_path, "r") as f:
        config = yaml.safe_load(f)
    return config["system_prompt"]


SYSTEM_PROMPT = load_system_prompt()

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion:\n{query}"),
    ]
)

generation_chain: Runnable = prompt_template | llm | StrOutputParser()


def generate_answer(query, documents):
    context = "\n\n".join(
        f"{doc.page_content}\n[source: {doc.metadata['source']} | {doc.metadata['chunk_id']}]"
        for doc in documents
    )

    return generation_chain.invoke({"context": context, "query": query})