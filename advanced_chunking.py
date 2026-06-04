from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.stores import InMemoryStore

load_dotenv()

# ── Load PDF ──
print("Loading PDF...")
loader = PyPDFLoader("BIRET.pdf")
documents = loader.load()

# ── Embeddings (shared across all strategies) ──
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# ═══════════════════════════════════════════
# STRATEGY 1 — RecursiveCharacterTextSplitter
# ═══════════════════════════════════════════
print("\n--- Strategy 1: Recursive Chunking ---")

recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\n\n", "\n", ".", " ", ""]
    # tries each separator in order ↑
    # paragraphs first → lines → sentences → words → chars
)

recursive_chunks = recursive_splitter.split_documents(documents)
print(f"Total chunks: {len(recursive_chunks)}")
print(f"Sample chunk:\n{recursive_chunks[0].page_content[:300]}")

# store in pinecone
PineconeVectorStore.from_documents(
    recursive_chunks,
    embeddings,
    index_name=os.environ["INDEX_NAME"]
)
print("Strategy 1 ingested into Pinecone ✅")


# ═══════════════════════════════════════════
# STRATEGY 2 — Semantic Chunking
# ═══════════════════════════════════════════
print("\n--- Strategy 2: Semantic Chunking ---")

semantic_splitter = SemanticChunker(
    embeddings=embeddings,
    breakpoint_threshold_type="percentile",  # split when meaning shifts beyond 95th percentile
    breakpoint_threshold_amount=95
)

semantic_chunks = semantic_splitter.split_documents(documents)
print(f"Total chunks: {len(semantic_chunks)}")
print(f"Sample chunk:\n{semantic_chunks[0].page_content[:300]}")

# notice — chunk sizes vary based on meaning shifts
sizes = [len(c.page_content) for c in semantic_chunks]
print(f"Smallest chunk: {min(sizes)} chars")
print(f"Largest chunk:  {max(sizes)} chars")
print(f"Average chunk:  {sum(sizes)//len(sizes)} chars")


# ═══════════════════════════════════════════
# STRATEGY 3 — Parent Document Retrieval (manual)
# ═══════════════════════════════════════════
print("\n--- Strategy 3: Parent Document Retrieval ---")

# small chunks for precise search
child_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

# large chunks for rich context
parent_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=200
)

# create both sets of chunks
child_chunks = child_splitter.split_documents(documents)
parent_chunks = parent_splitter.split_documents(documents)

print(f"Child chunks (stored in Pinecone): {len(child_chunks)}")
print(f"Parent chunks (returned to LLM):   {len(parent_chunks)}")
print(f"\nChild chunk size: ~{len(child_chunks[0].page_content)} chars")
print(f"Parent chunk size: ~{len(parent_chunks[0].page_content)} chars")
print(f"\nSample child chunk:\n{child_chunks[0].page_content}")
print(f"\nSample parent chunk:\n{parent_chunks[0].page_content[:400]}...")
print("\nStrategy 3 chunks created ✅")

# test it
query = "What is the total occupancy of BIRET?"
retrieved = parent_retriever.invoke(query)
print(f"\nQuery: {query}")
print(f"Retrieved {len(retrieved)} parent chunks")
print(f"First parent chunk ({len(retrieved[0].page_content)} chars):\n{retrieved[0].page_content[:400]}")