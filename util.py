import re
import time
from io import BytesIO
from typing import Any, Dict, List
import openai
import streamlit as st
from pypdf import PdfReader
from langchain import LLMChain, OpenAI
from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import RetrievalQA
from langchain.chains.question_answering import load_qa_chain
from langchain.docstore.document import Document
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS


OPENAI_API_KEY = "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"


@st.cache_data
def pdf_parser(file: BytesIO) -> List[str]:
    """ Extract text from a pdf file object
        Clean, remove specific symbols such as hyphenated word, fixing newlines
        and return a list of string for a page of PDF
    """
    pdf_reader = PdfReader(file)

    output_str = []
    for page in pdf_reader.pages:
        txt = page.extract_text()                              # extract text from each page in the pdf file
        txt = re.sub(r"(\w+)-\n(\w+)", r"\1\2", txt)           # Merge hyphenated words
        txt = re.sub(r"(?<!\n\s)\n(?!\s\n)", " ", txt.strip()) # Fix newlines in the middles of sentences
        txt = re.sub(r"\n\s*\n", "\n\n", txt)                  # Remove multiple newlines
        output_str.append(txt)

    return output_str


@st.cache_data
def page_chunker(text: str) -> list[Document]:
    """ Converts a list of strings to a list of LangChain Document objects.
        Each Document represents a chunk of text of up to 4000 characters with metadata (configurable)
    """
    if isinstance(text, str):
        text = [text]
    page_docs = [Document(page_content=page) for page in text] # Convert the text into a Document object

    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1                           # Add page numbers as metadata
    
    text_chunks = []
    for doc in page_docs:
        # Split text from the page
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0
            )
        page_text = text_splitter.split_text(doc.page_content)

        # Convert page text into a Document object with metadata (page number & index)
        for i, p_t in enumerate(page_text):
            doc = Document(page_content=p_t, metadata={"page":doc.metadata["page"], "chunk":i})
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            text_chunks.append(doc)

    return text_chunks
