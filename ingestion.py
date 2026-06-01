from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Ingesting the source data...")
    loader=TextLoader("biretinvestor.txt",encoding="UTF-8") #path of your document
    document = loader.load() #loading the source as a document by langchain
    
    print("Splitting the document into chunks...")
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0) #splitting the document into chunks of 1000 characters with no overlap
    texts=text_splitter.split_documents(document)
    
    # embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") #creating the embedding model using HuggingFace
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")  # 1024 dims

    print("Ingesting the chunks into Pinecone...")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"]) #ingesting the chunks into Pinecone vector store using the created embedding model and the index name from the environment variable
    print("Ingestion complete!")