import os
import pickle
import faiss
import numpy as np

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

# =====================================
# SETTINGS
# =====================================

PDF_FOLDER = "documents"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# =====================================
# TEXT SPLITTER
# =====================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# =====================================
# STORAGE
# =====================================

all_chunks = []

documents_data = []

# =====================================
# LOAD PDFs
# =====================================

for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        pdf_path = os.path.join(
            PDF_FOLDER,
            file
        )

        print(f"\nLoading: {file}")

        reader = PdfReader(pdf_path)

        # -----------------------------
        # PAGE LOOP
        # -----------------------------

        for page_number, page in enumerate(reader.pages):

            page_text = page.extract_text()

            if not page_text:
                continue

            chunks = splitter.split_text(
                page_text
            )

            # -----------------------------
            # CHUNK LOOP
            # -----------------------------

            for chunk in chunks:

                all_chunks.append(chunk)

                documents_data.append(
                    {
                        "text": chunk,
                        "source": file,
                        "page": page_number + 1
                    }
                )

print("\nTotal Chunks:", len(all_chunks))

# =====================================
# EMBEDDINGS
# =====================================

print("\nLoading Embedding Model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

embeddings = model.encode(
    all_chunks
)

embeddings = np.array(
    embeddings
).astype("float32")

print("Embeddings Created")

# =====================================
# FAISS
# =====================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(
    dimension
)

index.add(embeddings)

print("FAISS Index Ready")

# =====================================
# SAVE INDEX
# =====================================

faiss.write_index(
    index,
    "vectorstore/faiss.index"
)

# =====================================
# SAVE METADATA
# =====================================

with open(
    "vectorstore/chunks.pkl",
    "wb"
) as f:

    pickle.dump(
        documents_data,
        f
    )

print("\nIndex Saved Successfully")