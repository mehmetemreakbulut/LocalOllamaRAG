from langchain_community.llms import Ollama
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.embeddings import HuggingFaceEmbeddings
from bs4 import BeautifulSoup as Soup
from langchain.utils.html import (PREFIXES_TO_IGNORE_REGEX,
                                  SUFFIXES_TO_IGNORE_REGEX)
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import *
import logging
import sys

logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


global conversation
conversation = None

global vectordb
vectordb = None


def init_index():
    global vectordb
    if not INIT_INDEX:
        logging.info("continue without initializing index")
        return

    # Load, chunk and index the contents of the blog.
    loader = WebBaseLoader(
        web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        ),
    )
    docs = loader.load()
    # split text
    # this chunk_size and chunk_overlap effects to the prompt size
    # execeed promt size causes error `prompt size exceeds the context window size and cannot be processed`
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(docs)

    # create embeddings with huggingface embedding model `all-MiniLM-L6-v2`
    # then persist the vector index on vector db
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(documents=documents, embedding=embeddings)


def init_conversation():
    global conversation
    global vectordb

    # phi llm which runs with ollama
    # ollama expose an api for the llam in `localhost:11434`
    llm = Ollama(
        model="phi",
        base_url="http://localhost:11434",
    )
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    # create conversation
    '''
    conversation = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        verbose=True,
    )
    '''

    from langchain_core.runnables import RunnableMap

    # Create a RunnableMap to handle the input processing
    conversation = (
        RunnableMap({
            "context":  vectordb.as_retriever() | format_docs,
            "question": RunnablePassthrough()
        })
        | (lambda inputs: f"Answer the following question based on the context provided:\n\nContext:\n{inputs['context']}\n\nQuestion: {inputs['question']}")
        | llm
        | StrOutputParser()
    )



def chat(question, user_id):
    global conversation

    chat_history = []
    answer = conversation.invoke(question)
 
    

    logging.info("got response from llm - %s", answer)

    # TODO save history

    return answer
