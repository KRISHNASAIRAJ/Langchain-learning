import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import chainlit as cl

load_dotenv()

print("Initializing components...")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.environ["INDEX_NAME"])

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# NEW — prompt now has a {history} slot for past messages
prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a portfolio agent. Answer honestly and without bias.
    Answer based only on the following context:
    {context}"""),
    MessagesPlaceholder(variable_name="history"),  # ← chat history goes here
    ("human", "{question}")                         # ← current question goes here
])

# NEW — stores the conversation history
chat_history = ChatMessageHistory()

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def chat(query):
    # 1. retrieve relevant chunks
    docs = retriever.invoke(query)
    context = format_docs(docs)

    # 2. build prompt with history
    messages = prompt_template.format_messages(
        context=context,
        history=chat_history.messages,  # ← pass full history
        question=query
    )

    # 3. get response
    response = llm.invoke(messages)
    answer = response.content

    # 4. save this turn to history
    chat_history.add_user_message(query)      # ← save question
    chat_history.add_ai_message(answer)        # ← save answer

    return answer

if __name__ == "__main__":
    print("Chatbot ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        answer = chat(query)
        print(f"\nBot: {answer}")