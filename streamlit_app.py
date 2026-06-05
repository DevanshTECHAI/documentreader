import streamlit as st
import faiss
import pickle
import ollama
import os

from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="AI RAG Assistant",
    page_icon="🤖",
    layout="wide"
)

# =====================================================
# TITLE
# =====================================================

st.title("🤖 AI RAG Assistant")

st.markdown(
    """
Upload PDFs and chat with your documents using AI.
"""
)

# =====================================================
# SESSION MEMORY
# =====================================================

if "chat_history" not in st.session_state:

    st.session_state.chat_history = []

# =====================================================
# CREATE FOLDERS
# =====================================================

os.makedirs("documents", exist_ok=True)

os.makedirs("vectorstore", exist_ok=True)

# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("⚙️ Settings")

    model_name = st.selectbox(
        "Choose AI Model",
        [
            "phi3",
            "mistral"
        ]
    )

    top_k = st.slider(
        "Retrieved Chunks",
        min_value=1,
        max_value=10,
        value=4
    )

    if st.button("Clear Chat"):

        st.session_state.chat_history = []

        st.success("Chat Cleared")

# =====================================================
# PDF UPLOAD
# =====================================================

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)

# =====================================================
# PROCESS PDFs
# =====================================================

if uploaded_files:

    all_chunks = []

    metadata = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    for uploaded_file in uploaded_files:

        pdf_path = os.path.join(
            "documents",
            uploaded_file.name
        )

        # Save PDF
        with open(pdf_path, "wb") as f:

            f.write(
                uploaded_file.getbuffer()
            )

        st.success(
            f"Uploaded: {uploaded_file.name}"
        )

        # Read PDF
        reader = PdfReader(pdf_path)

        for page_number, page in enumerate(reader.pages):

            page_text = page.extract_text()

            if page_text:

                chunks = splitter.split_text(
                    page_text
                )

                for chunk in chunks:

                    all_chunks.append(chunk)

                    metadata.append({
                        "source": uploaded_file.name,
                        "page": page_number + 1
                    })

    st.info(
        f"Total Chunks Created: {len(all_chunks)}"
    )

    # =================================================
    # CREATE EMBEDDINGS
    # =================================================

    with st.spinner(
        "Creating Embeddings..."
    ):

        embedding_model = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        embeddings = embedding_model.encode(
            all_chunks
        )

    st.success(
        "Embeddings Created"
    )

    # =================================================
    # CREATE FAISS INDEX
    # =================================================

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(
        dimension
    )

    index.add(embeddings)

    # Save FAISS
    faiss.write_index(
        index,
        "vectorstore/faiss.index"
    )

    # Save Chunks
    with open(
        "vectorstore/chunks.pkl",
        "wb"
    ) as f:

        pickle.dump(
            all_chunks,
            f
        )

    # Save Metadata
    with open(
        "vectorstore/metadata.pkl",
        "wb"
    ) as f:

        pickle.dump(
            metadata,
            f
        )

    st.success(
        "AI Knowledge Base Created"
    )

# =====================================================
# LOAD DATABASE
# =====================================================

if os.path.exists(
    "vectorstore/faiss.index"
):

    index = faiss.read_index(
        "vectorstore/faiss.index"
    )

    with open(
        "vectorstore/chunks.pkl",
        "rb"
    ) as f:

        all_chunks = pickle.load(f)

    with open(
        "vectorstore/metadata.pkl",
        "rb"
    ) as f:

        metadata = pickle.load(f)

    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    st.success(
        "✅ AI System Ready"
    )

    # =================================================
    # SHOW CHAT HISTORY
    # =================================================

    for chat in st.session_state.chat_history:

        with st.chat_message(
            chat["role"]
        ):

            st.markdown(
                chat["content"]
            )

    # =================================================
    # CHAT INPUT
    # =================================================

    question = st.chat_input(
        "Ask something from your PDFs..."
    )

    # =================================================
    # PROCESS QUESTION
    # =================================================

    if question:

        # Save User Message
        st.session_state.chat_history.append({
            "role": "user",
            "content": question
        })

        # Show User Message
        with st.chat_message("user"):

            st.markdown(question)

        # =============================================
        # ASSISTANT RESPONSE
        # =============================================

        with st.chat_message("assistant"):

            with st.spinner(
                "Thinking..."
            ):

                # =====================================
                # CREATE QUESTION EMBEDDING
                # =====================================

                question_embedding = embedding_model.encode(
                    [question]
                )

                # =====================================
                # SEARCH FAISS
                # =====================================

                distances, indices = index.search(
                    question_embedding,
                    k=top_k
                )

                retrieved_chunks = []

                retrieved_sources = []

                for idx in indices[0]:

                    retrieved_chunks.append(
                        all_chunks[idx]
                    )

                    retrieved_sources.append(
                        metadata[idx]
                    )

                # =====================================
                # CONTEXT
                # =====================================

                context = "\n\n".join(
                    retrieved_chunks
                )

                # =====================================
                # MEMORY
                # =====================================

                history = ""

                for msg in st.session_state.chat_history[-6:]:

                    history += (
                        f"{msg['role']}: "
                        f"{msg['content']}\n"
                    )

                # =====================================
                # ADVANCED PROMPT
                # =====================================

                prompt = f"""
You are an advanced AI document assistant.

Your task is to answer questions ONLY
from the provided document context.

STRICT RULES:

1. Use ONLY provided context
2. NEVER hallucinate
3. NEVER make up answers
4. If answer is missing say:
   "I could not find that information in the documents."
5. Keep answers professional
6. Keep answers concise
7. Use bullet points when useful
8. Use previous conversation for context

------------------------------------------------

PREVIOUS CONVERSATION:
{history}

------------------------------------------------

DOCUMENT CONTEXT:
{context}

------------------------------------------------

QUESTION:
{question}

------------------------------------------------

PROFESSIONAL ANSWER:
"""

                # =====================================
                # OLLAMA LLM
                # =====================================

                response = ollama.chat(
                    model=model_name,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                answer = response[
                    "message"
                ]["content"]

                # =====================================
                # SHOW ANSWER
                # =====================================

                st.markdown(answer)

                st.divider()

                # Save AI Message
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer
                })

                # =====================================
                # SHOW SOURCES
                # =====================================

                with st.expander(
                    "📚 View Sources"
                ):

                    shown = set()

                    for source in retrieved_sources:

                        source_text = (
                            f"{source['source']} "
                            f"(Page {source['page']})"
                        )

                        if source_text not in shown:

                            st.info(
                                source_text
                            )

                            shown.add(
                                source_text
                            )