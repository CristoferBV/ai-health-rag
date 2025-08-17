import os, glob
from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from sentence_transformers import SentenceTransformer

load_dotenv()
DB_DIR = os.environ.get("CHROMA_DB_DIR", "chroma_db")
DATA_DIR = os.environ.get("DATA_DIR", "data")

def load_docs():
    docs = []
    # PDFs
    for fp in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
        docs += PyPDFLoader(fp).load()
    # TXT/MD
    docs += DirectoryLoader(DATA_DIR, glob="*.txt", loader_cls=TextLoader).load()
    docs += DirectoryLoader(DATA_DIR, glob="*.md", loader_cls=TextLoader).load()
    return docs

def get_embeddings():
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return OpenAIEmbeddings(model="text-embedding-3-small")
    # Fallback local (sin API key)
    from langchain_community.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

if __name__ == "__main__":
    docs = load_docs()
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
    chunks = splitter.split_documents(docs)
    print(f"Loaded {len(docs)} docs → {len(chunks)} chunks")
    vs = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=DB_DIR,
    )
    vs.persist()
    print("Vector DB ready at", DB_DIR)
