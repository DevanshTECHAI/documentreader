from sentence_transformers import SentenceTransformer

# Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

text = "Payment must be completed within 30 days."

embedding = model.encode(text)

print("Embedding Length:", len(embedding))

print("\nFirst 10 Numbers:\n")

print(embedding[:10])