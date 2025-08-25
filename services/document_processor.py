# services/document_processor.py
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def process_pdf(files):
    all_chunked_docs = []
    for file in files:
        loader = PyPDFLoader(file)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        chunked_doc =  text_splitter.split_documents(documents)
        all_chunked_docs.extend(chunked_doc)

    return all_chunked_docs