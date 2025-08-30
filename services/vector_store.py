# services/vector_store.py
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def create_vector_store(chunked_docs):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunked_docs, embeddings)
    return vectorstore

