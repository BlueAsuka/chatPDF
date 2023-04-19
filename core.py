import gui
import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI


# Set the api and embed the text 
api = gui.set_openai_api()


def embed(pages):
    """ Embed the pages using LLM """
    emb = OpenAIEmbeddings(openai_api_key=api)

    with st.spinner("Text Embedding..."):
        index = FAISS.from_documents(pages, emb)

    return index


def retrieval(index):
    """ Use RetrievalQA for text retrieval """
    return RetrievalQA.from_chain_type(
        llm=OpenAI(openai_api_key=api),
        chain_type="stuff",
        retriever=index.as_retriever()
    )


def tool():
    pass
