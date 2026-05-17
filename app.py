import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from pypdf import PdfReader
import time

# =========================================================
# FIX TOKENIZER WARNING
# =========================================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Study Buddy RAG",
    page_icon="📚",
    layout="wide"
)

# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.stButton button {
    width: 100%;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    font-weight: bold;
    background-color: #4CAF50;
    color: white;
}

.stTextInput input {
    border-radius: 10px;
}

.answer-box {
    background-color: #1e1e1e;
    padding: 20px;
    border-radius: 12px;
    color: white;
    margin-top: 20px;
    border: 1px solid #444;
}

.context-box {
    background-color: #262730;
    padding: 15px;
    border-radius: 10px;
    color: white;
    margin-bottom: 10px;
    border: 1px solid #444;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TITLE
# =========================================================

st.title("📚 Study Buddy RAG with Gemini")
st.caption("Upload PDF → Ask Questions → Get AI Answers")

# =========================================================
# GEMINI API CONFIG
# =========================================================

try:

    # Get API Key from Streamlit Secrets
    api_key = st.secrets["GOOGLE_API_KEY"]

    # Configure Gemini
    genai.configure(api_key=api_key)

    # Load Gemini Model
    llm = genai.GenerativeModel(
        model_name="gemini-1.5-flash"
    )

    # Test API
    test_response = llm.generate_content("Hello")

    if test_response:
        st.sidebar.success("✅ Gemini Connected")

except KeyError:

    st.error("""
❌ GOOGLE_API_KEY not found.

Add this inside Streamlit Secrets:

GOOGLE_API_KEY = "your_api_key"
""")

    st.stop()

except Exception as e:

    st.error(f"""
❌ Gemini API Error

{e}
""")

    st.info("""
Possible reasons:
- Invalid API key
- Billing issue
- Wrong Gemini model
- API quota exceeded
""")

    st.stop()

# =========================================================
# SIDEBAR SETTINGS
# =========================================================

st.sidebar.header("⚙️ Settings")

chunk_size = st.sidebar.slider(
    "Chunk Size",
    200,
    1000,
    500,
    100
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    0,
    200,
    50,
    10
)

top_k = st.sidebar.slider(
    "Top Chunks",
    1,
    10,
    4
)

# =========================================================
# LOAD EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

try:

    with st.spinner("🔄 Loading embedding model..."):

        embedding_model = load_embedding_model()

except Exception as e:

    st.error(f"""
❌ Failed to load embedding model

{e}
""")

    st.stop()

# =========================================================
# FILE UPLOAD
# =========================================================

uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type=["pdf"]
)

# =========================================================
# CHUNKING FUNCTION
# =========================================================

def chunk_text(
    text,
    chunk_size=500,
    overlap=50
):

    chunks = []

    start = 0

    while start < len(text):

        end = min(
            start + chunk_size,
            len(text)
        )

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# =========================================================
# MAIN APP
# =========================================================

if uploaded_file:

    st.success(
        f"✅ Uploaded: {uploaded_file.name}"
    )

    tmp_path = None

    try:

        # =================================================
        # SAVE TEMP FILE
        # =================================================

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())

            tmp_path = tmp.name

        # =================================================
        # EXTRACT PDF TEXT
        # =================================================

        with st.spinner("📖 Extracting PDF text..."):

            reader = PdfReader(tmp_path)

            full_text = ""

            for page in reader.pages:

                try:

                    text = page.extract_text()

                    if text:
                        full_text += text + "\n"

                except Exception:
                    pass

            time.sleep(1)

        # =================================================
        # EMPTY CHECK
        # =================================================

        if not full_text.strip():

            st.error(
                "❌ No readable text found in PDF"
            )

            st.stop()

        # =================================================
        # CREATE CHUNKS
        # =================================================

        texts = chunk_text(
            full_text,
            chunk_size,
            chunk_overlap
        )

        if len(texts) == 0:

            st.error(
                "❌ Failed to create chunks"
            )

            st.stop()

        # =================================================
        # CREATE EMBEDDINGS
        # =================================================

        with st.spinner("🧠 Creating embeddings..."):

            embeddings = embedding_model.encode(
                texts,
                convert_to_numpy=True,
                show_progress_bar=False
            )

            embeddings = np.array(
                embeddings,
                dtype=np.float32
            )

        # =================================================
        # CREATE FAISS INDEX
        # =================================================

        with st.spinner(
            "⚡ Building vector database..."
        ):

            dimension = embeddings.shape[1]

            index = faiss.IndexFlatL2(
                dimension
            )

            index.add(embeddings)

        # =================================================
        # SUCCESS MESSAGE
        # =================================================

        st.success(f"""
✅ PDF processed successfully!

📄 Total Chunks: {len(texts)}
""")

        # =================================================
        # QUESTION INPUT
        # =================================================

        st.subheader("💬 Ask Questions")

        question = st.text_input(
            "Enter your question"
        )

        # =================================================
        # GENERATE ANSWER
        # =================================================

        if st.button("🚀 Generate Answer"):

            if not question.strip():

                st.warning(
                    "⚠️ Please enter a question"
                )

            else:

                # =============================================
                # SEARCH RELEVANT CHUNKS
                # =============================================

                with st.spinner(
                    "🔍 Searching context..."
                ):

                    q_embedding = embedding_model.encode(
                        [question],
                        convert_to_numpy=True
                    )

                    q_embedding = np.array(
                        q_embedding,
                        dtype=np.float32
                    )

                    distances, indices = index.search(
                        q_embedding,
                        k=min(top_k, len(texts))
                    )

                # =============================================
                # RETRIEVE CHUNKS
                # =============================================

                retrieved_chunks = []

                for idx in indices[0]:

                    if idx < len(texts):

                        retrieved_chunks.append(
                            texts[idx]
                        )

                context = "\n\n".join(
                    retrieved_chunks
                )

                # =============================================
                # PROMPT
                # =============================================

                prompt = f"""
You are an intelligent AI Study Assistant.

Answer ONLY from the provided context.

If the answer is not present in the context,
reply exactly:

"I could not find the answer in the uploaded PDF."

================ CONTEXT ================

{context}

================ QUESTION ================

{question}

================ ANSWER ================
"""

                # =============================================
                # GEMINI RESPONSE
                # =============================================

                try:

                    with st.spinner(
                        "🤖 Gemini is thinking..."
                    ):

                        response = llm.generate_content(
                            prompt
                        )

                    # =========================================
                    # SAFE RESPONSE EXTRACTION
                    # =========================================

                    answer = "No response generated."

                    try:

                        if response.text:

                            answer = response.text

                    except Exception:

                        pass

                    # =========================================
                    # DISPLAY ANSWER
                    # =========================================

                    st.subheader("📌 Answer")

                    st.markdown(f"""
<div class="answer-box">

{answer}

</div>
""", unsafe_allow_html=True)

                    # =========================================
                    # SHOW CONTEXT
                    # =========================================

                    with st.expander(
                        "📚 Retrieved Context"
                    ):

                        for i, chunk in enumerate(
                            retrieved_chunks
                        ):

                            st.markdown(
                                f"### Chunk {i+1}"
                            )

                            st.markdown(f"""
<div class="context-box">

{chunk}

</div>
""", unsafe_allow_html=True)

                    # =========================================
                    # STATS
                    # =========================================

                    st.subheader("📊 Stats")

                    col1, col2, col3 = st.columns(3)

                    col1.metric(
                        "Chunks",
                        len(texts)
                    )

                    col2.metric(
                        "Top Matches",
                        top_k
                    )

                    col3.metric(
                        "Characters",
                        len(full_text)
                    )

                except Exception as e:

                    st.error(f"""
❌ Gemini Response Error

{e}
""")

    except Exception as e:

        st.error(f"""
❌ Error Processing PDF

{e}
""")

    finally:

        # =================================================
        # DELETE TEMP FILE
        # =================================================

        if (
            tmp_path
            and os.path.exists(tmp_path)
        ):

            os.unlink(tmp_path)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.caption(
    "🚀 Built with Streamlit + Gemini + FAISS + SentenceTransformers"
)
