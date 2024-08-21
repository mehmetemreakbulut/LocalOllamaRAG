import bs4
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.embeddings import HuggingFaceEmbeddings

llm = Ollama(
        model="phi",
        base_url="http://localhost:11434",
    )


### Construct retriever ###
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
retriever = vectorstore.as_retriever()


### Contextualize question ###
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


### Answer question ###
qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""


qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),  # Context only, should not appear in the final answer
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


### Statefully manage chat history ###
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def refine_history(history: BaseChatMessageHistory):
    # Extract only necessary parts of the history for context
    # For example, use only the last N turns, or preprocess the history
    refined_history = []
    for message in history.messages[-5:]:  # Use only the last 5 messages
        refined_history.append(message)
    return refined_history


conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

chat_history=[]
answer1=""
for chunk in rag_chain.stream(
    {"input": "What is Task Decomposition?","chat_history": chat_history},
):
    if "answer" not in chunk:
        continue
    if chunk['answer'] == "Human:":
        break
    print(chunk['answer'], end="", flush=True)
    answer1=answer1+chunk['answer']

chat_history.append(answer1)
print("chat_history:",chat_history)
answer2=""
for chunk in rag_chain.stream(
    {"input": "What are common ways of doing it?","chat_history": chat_history},
):
    if "answer" not in chunk:
        continue
    if chunk['answer'] == "Human:":
        break
    print(chunk['answer'], end="", flush=True)
    answer2=answer2+chunk['answer']

chat_history.append(answer2)
print("chat_history:",chat_history)

for i in range(10):
    question = input("Enter your question: ")

    answerx = ""
    for chunk in rag_chain.stream(
        {"input": question, "chat_history": chat_history},
    ):
        if "answer" not in chunk:
            continue
        if chunk['answer'] == "Human:":
            break
        print(chunk['answer'], end="", flush=True)
        answerx = answerx + chunk['answer']

    chat_history.append(answerx)