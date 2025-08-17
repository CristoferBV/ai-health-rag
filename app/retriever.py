import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

DB_DIR = os.environ.get("CHROMA_DB_DIR", "chroma_db")

def get_embeddings():
    from os import getenv
    if getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddings(model="text-embedding-3-small")
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def build_retriever(k: int = 5):
    vs = Chroma(persist_directory=DB_DIR, embedding_function=get_embeddings())
    return vs.as_retriever(search_kwargs={"k": k})
