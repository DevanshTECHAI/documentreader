from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Documents

chunks = [
    "Payment must be completed within 30 days.",
    "The contract can be terminated with notice.",
    "Taxes must be filed before March 31."
]

# Load model

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# Create embeddings

embeddings = model.encode(chunks)

embeddings = np.array(
    embeddings
).astype("float32")

# Create FAISS index

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(embeddings)

# User question

query = "What are the payment terms?"

query_embedding = model.encode(
    [query]
)

query_embedding = np.array(
    query_embedding
).astype("float32")

# Search

k = 2

distances, indices = index.search(
    query_embedding,
    k
)

print("\nRelevant Chunks:\n")

for i in indices[0]:
    print(chunks[i])