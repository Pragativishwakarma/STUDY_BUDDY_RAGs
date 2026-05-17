import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from pypdf import PdfReader
import time

# ---------------- SETTINGS ----------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="Study Buddy RAG",
    page_icon="📚",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------

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

# ---------------- TITLE ----------------

st.title("📚 Study Buddy RAG")
st.caption("Upload PDF → Ask Questions → Get AI Answers")

# ---------------- GEMINI CONFIG ----------------

try:

    # STREAMLIT SECRETS
    api_key = st.secrets["GOOGLE_API_KEY"]

    # CONFIGURE GEMINI
    genai.configure(api_key=api_key)

    # LOAD MODEL
    llm = genai.GenerativeModel(
        model_name="gemini-1.5-flash"
    )

    # TEST API
    test = llm.generate_content("Hello")

    if test:
        st.sidebar.success("✅ Gemini Connected")

except Exception as e:

    st.error(f"""
❌ Gemini API Error

{e}
""")

    st.info("""
Possible fixes:
- Wrong API key
- Billing issue
- Expired key
- Invalid Gemini model
""")

    st.stop()

# ---------------- SIDEBAR ----------------

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
    "Top Context Chunks",
    1,
    10,
    4
)

# ---------------- LOAD EMBEDDING MODEL ----------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

try:

    with st.spinner("🔄 Loading embedding model..."):

        embedding_model = load_embedding_model()

except Exception as e:

    st.error(f"❌ Embedding Model Error\n\n{e}")

    st.stop()

# ---------------- FILE UPLOADER ----------------

uploaded_file = st.file_uploader(
    "📄 Upload PDF",
    type=["pdf"]
)

# ---------------- CHUNK FUNCTION ----------------

def chunk_text(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

# ---------------- MAIN APP ----------------

if uploaded_file:

    st.success(f"✅ Uploaded: {uploaded_file.name}")

    tmp_path = None

    try:

        # SAVE TEMP FILE
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())

            tmp_path = tmp.name

        # EXTRACT PDF TEXT
        with st.spinner("📖 Reading PDF..."):

            reader = PdfReader(tmp_path)

            full_text = ""

            for page in reader.pages:

                text = page.extract_text()

                if text:
                    full_text += text + "\n"

        if not full_text.strip():

            st.error("❌ No readable text found.")

            st.stop()

        # CHUNKING
        texts = chunk_text(
            full_text,
            chunk_size,
            chunk_overlap
        )

        st.success(f"✅ Created {len(texts)} chunks")

        # CREATE EMBEDDINGS
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

        # CREATE FAISS INDEX
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        # ---------------- QUESTION INPUT ----------------

        st.subheader("💬 Ask Questions")

        question = st.text_input(
            "Ask anything from the PDF"
        )

        if st.button("🚀 Generate Answer"):

            if not question.strip():

                st.warning("⚠️ Enter a question")

            else:

                # QUESTION EMBEDDING
                q_embedding = embedding_model.encode(
                    [question],
                    convert_to_numpy=True
                )

                q_embedding = np.array(
                    q_embedding,
                    dtype=np.float32
                )

                # SEARCH
                with st.spinner("🔍 Searching PDF..."):

                    distances, indices = index.search(
                        q_embedding,
                        k=min(top_k, len(texts))
                    )

                # RETRIEVE CONTEXT
                retrieved_chunks = []

                for idx in indices[0]:

                    if idx < len(texts):

                        retrieved_chunks.append(
                            texts[idx]
                        )

                context = "\n\n".join(
                    retrieved_chunks
                )

                # PROMPT
                prompt = f"""
You are an AI Study Assistant.

Answer ONLY from the provided context.

If answer is unavailable, say:
"I could not find the answer in the uploaded PDF."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

                # GEMINI RESPONSE
                try:

                    with st.spinner("🤖 Gemini Thinking..."):

                        response = llm.generate_content(
                            prompt
                        )

                    answer = response.text

                    # OUTPUT
                    st.subheader("📌 Answer")

                    st.markdown(f"""
<div class="answer-box">

{answer}

</div>
""", unsafe_allow_html=True)

                    # CONTEXT
                    with st.expander("📚 Retrieved Context"):

                        for i, chunk in enumerate(retrieved_chunks):

                            st.markdown(f"### Chunk {i+1}")

                            st.markdown(f"""
<div class="context-box">

{chunk}

</div>
""", unsafe_allow_html=True)

                    # STATS
                    st.subheader("📊 Stats")

                    c1, c2, c3 = st.columns(3)

                    c1.metric(
                        "Chunks",
                        len(texts)
                    )

                    c2.metric(
                        "Top Matches",
                        top_k
                    )

                    c3.metric(
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
❌ PDF Processing Error

{e}
""")

    finally:

        if tmp_path and os.path.exists(tmp_path):

            os.unlink(tmp_path)

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "🚀 Built with Streamlit + Gemini + FAISS"
)
