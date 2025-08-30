# services/document_processor.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def process_pdf(files):
    all_chunked_docs = []
    for file in files:
        loader = PyPDFLoader(file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

        chunked_docs = text_splitter.split_documents(documents)

        # Add metadata for source filename
        for doc in chunked_docs:
            doc.metadata["source"] = file

        all_chunked_docs.extend(chunked_docs)

    return all_chunked_docs
