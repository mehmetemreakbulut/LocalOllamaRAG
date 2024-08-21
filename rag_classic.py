import time
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama

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

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()

llm = Ollama(
        model="phi",
        base_url="http://localhost:11434",
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


from langchain_core.runnables import RunnableMap

# Create a RunnableMap to handle the input processing
rag_chain = (
    RunnableMap({
        "context": retriever | format_docs,
        "question": RunnablePassthrough()
    })
    | (lambda inputs: f"Answer the following question based on the context provided:\n\nContext:\n{inputs['context']}\n\nQuestion: {inputs['question']}")
    | llm
    | StrOutputParser()
)



# Now you can call the rag_chain with different questions dynamically, getting question from user
'''
for i in range(10):
    question = input("Enter your question: ")

    for chunk in rag_chain.stream(question):
        print(chunk, end="", flush=True)
'''

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

from langchain_core.messages import HumanMessage

chat_history = []

question = input("Enter your question: ")
ai_msg_1 = ''
for chunk in rag_chain.stream({"input": question, "chat_history": chat_history}): 
    if 'answer' not in chunk:
        continue
    print(chunk['answer'], end="", flush=True)
    if isinstance(chunk, str):
        ai_msg_1 = ai_msg_1 + chunk["answer"]

#ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})

chat_history.extend([HumanMessage(content=question), ai_msg_1])

for i in range(10):
    question = input("Enter your question: ")
    ai_msg_2 = ''
    for chunk in rag_chain.stream({"input": question, "chat_history": chat_history}):
        if 'answer' not in chunk:
            continue
        print(chunk['answer'], end="", flush=True)
        if isinstance(chunk, str):
            ai_msg_2 = ai_msg_2 + chunk["answer"]
    chat_history.extend([HumanMessage(content=question), ai_msg_2])
    time.sleep(1)