# app/ingest.py

import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import Milvus
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


#  TXT loader
def load_txt(path: str) -> list[str]:
    """Load text from a TXT file and return as list of strings"""
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def load_documents(path: str) -> list[str]:
    """Load documents from TXT file"""
    if path.endswith('.txt'):
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file format: {path}")


#  PDF loader
def load_pdf(path: str) -> list[str]:
    """Load text from a PDF file and return as list of strings"""
    loader = PyPDFLoader(path)
    docs = loader.load()
    
    # Extract text content from Document objects
    return [doc.page_content for doc in docs]


def ingest_to_milvus():
    #  file paths
    txt_path = r"data\input.txt"
    pdf_path = r"data\doc.pdf"

    docs = []
    docs.extend(load_txt(txt_path))
    docs.extend(load_pdf(pdf_path))

    # chunking
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=os.getenv("MILVUS_COLLECTION", "rag_collection"),
        connection_args={
            "uri": os.getenv("MILVUS_URI"),
            "token": os.getenv("MILVUS_TOKEN"),
            "db_name": os.getenv("MILVUS_DB_NAME", "default"),
        },
    )

    return vectorstore
