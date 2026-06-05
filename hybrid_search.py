from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

load_dotenv()

# ── Load and chunk PDF ──
print("Loading PDF...")
loader = PyPDFLoader("BIRET.pdf")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(documents)
texts = [chunk.page_content for chunk in chunks]
print(f"Total chunks: {len(chunks)}")

# ── Dense embeddings (BAAI) ──
print("Loading dense embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# ── BM25 sparse encoder ──
print("Fitting BM25 encoder on chunks...")
bm25 = BM25Encoder()
bm25.fit(texts)          # learns term frequencies from your document
bm25.dump("bm25.json")   # save so we don't refit every time
print("BM25 fitted and saved ✅")

# ── Connect to Pinecone ──
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
sparse_index = pc.Index(os.environ["SPARSE_INDEX_NAME"])

# ── Ingest into sparse index ──
print("Ingesting into sparse index...")
for i, (chunk, text) in enumerate(zip(chunks, texts)):
    sparse_vector = bm25.encode_documents([text])[0]

    sparse_index.upsert(vectors=[{
        "id": f"chunk-{i}",
        "sparse_values": sparse_vector,  # sparse only ← removed dense values
        "metadata": {"text": text}
    }])

    if i % 10 == 0:
        print(f"Ingested {i}/{len(chunks)} chunks...")

print("Sparse index ingestion complete ✅")