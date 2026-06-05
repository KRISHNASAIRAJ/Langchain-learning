from dotenv import load_dotenv
import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder

load_dotenv()

# ── Initialize components ──
print("Initializing...")

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=os.environ.get("GROQ_API_KEY")
)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")

# load BM25 encoder fitted during ingestion
bm25 = BM25Encoder()
bm25.load("bm25.json")  # loads the saved encoder from ingestion

# connect to both indexes
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
dense_index  = pc.Index(os.environ["INDEX_NAME"])         # semantic
sparse_index = pc.Index(os.environ["SPARSE_INDEX_NAME"])  # keyword

prompt_template = ChatPromptTemplate.from_messages([
    ("system", """You are a portfolio agent. Answer honestly and without bias.
    Answer based only on the following context:
    {context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}")
])

chat_history = ChatMessageHistory()

# ── Hybrid retrieval function ──
def hybrid_retrieve(query, k=3, alpha=0.5):
    """
    alpha=1.0 → pure semantic
    alpha=0.0 → pure keyword
    alpha=0.5 → equal blend
    """

    # 1. embed query for dense search
    dense_vector = embeddings.embed_query(query)

    # 2. encode query for sparse search
    sparse_vector = bm25.encode_queries([query])[0]

    # 3. query dense index
    dense_results = dense_index.query(
        vector=dense_vector,
        top_k=k * 2,          # fetch more, RRF will rerank
        include_metadata=True
    )

    # 4. query sparse index
    sparse_results = sparse_index.query(
        sparse_vector=sparse_vector,
        top_k=k * 2,
        include_metadata=True
    )

    # 5. RRF merge
    rrf_scores = {}

    for rank, match in enumerate(dense_results["matches"]):
        chunk_id = match["id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + alpha * (1 / (rank + 60))

    for rank, match in enumerate(sparse_results["matches"]):
        chunk_id = match["id"]
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + (1 - alpha) * (1 / (rank + 60))

    # 6. sort by RRF score and get top-k
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:k]

    # 7. fetch metadata for top-k chunks
    all_matches = {m["id"]: m for m in dense_results["matches"]}
    all_matches.update({m["id"]: m for m in sparse_results["matches"]})

    top_chunks = [
        all_matches[id]["metadata"]["text"]
        for id in sorted_ids
        if id in all_matches
    ]

    return top_chunks

def format_context(chunks):
    return "\n\n".join(chunks)

def chat(query):
    # retrieve using hybrid search
    chunks = hybrid_retrieve(query, k=3, alpha=0.5)
    context = format_context(chunks)

    messages = prompt_template.format_messages(
        context=context,
        history=chat_history.messages,
        question=query
    )

    response = llm.invoke(messages)
    answer = response.content

    chat_history.add_user_message(query)
    chat_history.add_ai_message(answer)

    return answer

if __name__ == "__main__":
    print("Hybrid Search Chatbot ready! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            print("Goodbye!")
            break
        answer = chat(query)
        print(f"\nBot: {answer}")