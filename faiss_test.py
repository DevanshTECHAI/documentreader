from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Sample chunks
chunks = [
    "Payment must be completed within 30 days.",
    "The contract can be terminated with notice.",
    "Taxes must be filed before March 31."
]

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Convert chunks to vectors
embeddings = model.encode(chunks)

# Convert to float32 (FAISS requirement)
embeddings = np.array(embeddings).astype("float32")

# Create FAISS index
dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

# Add vectors
index.add(embeddings)

print("Vectors stored:", index.ntotal)