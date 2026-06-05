import faiss
import pickle
import numpy as np

from sentence_transformers import SentenceTransformer
from ollama import chat

# =====================================
# LOAD INDEX
# =====================================

print("Loading Index...")

index = faiss.read_index(
    "vectorstore/faiss.index"
)

print("Index Loaded")

# =====================================
# LOAD METADATA
# =====================================

with open(
    "vectorstore/chunks.pkl",
    "rb"
) as f:

    documents_data = pickle.load(f)

print("Metadata Loaded")

# =====================================
# EMBEDDING MODEL
# =====================================

print("Loading Embedding Model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding Model Ready")

# =====================================
# SETTINGS
# =====================================

TOP_K = 5

MAX_DISTANCE = 1.5

# =====================================
# CHAT LOOP
# =====================================

while True:

    query = input(
        "\nAsk a question (or exit): "
    )

    if query.lower() == "exit":
        break

    # =====================================
    # QUERY EMBEDDING
    # =====================================

    query_embedding = embedding_model.encode(
        [query]
    )

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    # =====================================
    # SEARCH
    # =====================================

    distances, indices = index.search(
        query_embedding,
        TOP_K
    )

    print("\n" + "=" * 60)
    print("SIMILARITY SCORES")
    print("=" * 60)

    context = ""

    sources = []

    for rank, idx in enumerate(indices[0]):

        score = distances[0][rank]

        print(
            f"Chunk {idx} -> Distance: {score:.4f}"
        )

        # FILTER BAD CHUNKS

        if score <= MAX_DISTANCE:

            chunk_data = documents_data[idx]

            context += (
                chunk_data["text"]
                + "\n\n"
            )

            sources.append(
                (
                    chunk_data["source"],
                    chunk_data["page"]
                )
            )

    # =====================================
    # SAFETY CHECK
    # =====================================

    if not context:

        print(
            "\nNo relevant information found."
        )

        continue

    # =====================================
    # SHOW CONTEXT
    # =====================================

    print("\n" + "=" * 60)
    print("RETRIEVED CONTEXT")
    print("=" * 60)

    print(context)

    # =====================================
    # PROMPT
    # =====================================

    prompt = f"""
You are a professional AI assistant.

Answer ONLY using the context below.

If the answer is not found,
reply exactly:

"I could not find that information in the documents."

Context:
{context}

Question:
{query}

Detailed Answer:
"""

    # =====================================
    # MISTRAL
    # =====================================

    response = chat(
        model="mistral",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    # =====================================
    # DISPLAY ANSWER
    # =====================================

    print("\n" + "=" * 60)
    print("AI ANSWER")
    print("=" * 60)

    print(answer)

    # =====================================
    # DISPLAY SOURCES
    # =====================================

    print("\n" + "=" * 60)
    print("SOURCES")
    print("=" * 60)

    for source, page in sorted(set(sources)):

        print(
            f"{source} — Page {page}"
        )