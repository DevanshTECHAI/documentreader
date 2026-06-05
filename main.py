from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from ollama import chat
import faiss
import numpy as np


# STEP 1: READ PDF


pdf_path = "documents/contract.pdf"

reader = PdfReader(pdf_path)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text

print("✅ PDF Loaded Successfully")
print("Total Characters:", len(text))


# STEP 2: CHUNKING


splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_text(text)

print("✅ Chunks Created:", len(chunks))


# STEP 3: EMBEDDINGS


print("\nLoading Embedding Model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = embedding_model.encode(chunks)

embeddings = np.array(embeddings).astype("float32")

print("✅ Embeddings Created")


# STEP 4: FAISS


dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("✅ FAISS Ready")
print("Vectors Stored:", index.ntotal)


# STEP 5: QUESTION LOOP


while True:

    query = input("\nAsk a question (or type 'exit'): ")

    if query.lower() == "exit":
        break

    
    # Create Query Embedding
    

    query_embedding = embedding_model.encode([query])

    query_embedding = np.array(
        query_embedding
    ).astype("float32")

    
    # Search FAISS
    

    k = 5

    distances, indices = index.search(
        query_embedding,
        k
    )

    
    # Build Context
    

    context = ""

    for i in indices[0]:
        context += chunks[i]
        context += "\n\n"

    print("\n" + "=" * 60)
    print("RETRIEVED CONTEXT")
    print("=" * 60)
    print(context)

    
    # Build Prompt
    

    prompt = f"""
You are an expert document assistant.

Use ONLY the information provided in the context.

If the answer is not present in the context,
reply:

"I could not find the answer in the document."

Context:
{context}

Question:
{query}

Provide a detailed answer based only on the context.

Answer:
"""

    
    # Ask Mistral
    

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

    print("\n" + "=" * 60)
    print("AI ANSWER")
    print("=" * 60)
    print(answer)