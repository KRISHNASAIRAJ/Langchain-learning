import os
from dotenv import load_dotenv
import chainlit as cl
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.environ.get("GROQ_API_KEY")
)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.environ["INDEX_NAME"])
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a portfolio agent. Answer honestly and without bias.
    Answer based only on the following context:
    {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@cl.on_chat_start                        # ← runs once when user opens the chat
async def start():
    cl.user_session.set("history", ChatMessageHistory())  # ← session per user
    await cl.Message(content="👋 Hi! Ask me anything about your portfolio.").send()

@cl.on_message                           # ← runs every time user sends a message
async def main(message: cl.Message):
    history = cl.user_session.get("history")

    docs = retriever.invoke(message.content)
    context = format_docs(docs)

    messages = prompt_template.format_messages(
        context=context,
        history=history.messages,
        question=message.content
    )

    response = llm.invoke(messages)
    answer = response.content

    history.add_user_message(message.content)
    history.add_ai_message(answer)

    await cl.Message(content=answer).send()  # ← send reply