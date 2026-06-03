import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

load_dotenv()

print("Initializing components...")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

vectorstore = PineconeVectorStore(embedding=embeddings, index_name=os.environ["INDEX_NAME"]) #ingesting the chunks into Pinecone vector store using the created embedding model and the index name from the environment variable

retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) #k specifies the number of relevant documents to retrieve

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context: 
    {context}
    Question: {question}
    provide a detailed answer:"""
    )

def format_docs(docs):
    """Format retrieved documents into a single string."""
    return "\n\n".join(doc.page_content for doc in docs)

if __name__ == "__main__":
    query = "What is the Total Occupancy percentage of BIRET?"
    docs = retriever.invoke(query) #retrieving relevant documents from the vector store based on the query
    context = format_docs(docs) #formatting the retrieved documents into a single string to be used as context for the LLM
    messages = prompt_template.format_messages(context=context, question=query) #formatting the prompt with the retrieved context and the original query
    response = llm.invoke(messages) #getting the response from the LLM based on the formatted prompt
    print(response.content)