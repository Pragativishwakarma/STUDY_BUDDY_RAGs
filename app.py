import streamlit as st
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from pypdf import PdfReader

# ---------------- SETTINGS ----------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="Study Buddy RAG",
    page_icon="📚",
    layout="wide"
)

# ---------------- PAGE STYLE ----------------

st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.stButton button {
    width: 100%;
    border-radius: 10px;
    height: 3em;
    font-size: 16px;
    font-weight: bold;
    background-color: #4CAF50;
    color: white;
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
st.caption("Upload PDF → Ask Questions → AI Answers")

# ---------------- OPENAI API KEY ----------------

# ✅ ADD YOUR OPENAI API KEY HERE
LOCAL_API_KEY = "YOUR_OPENAI_API_KEY"

try:

    # USE STREAMLIT SECRET IF AVAILABLE
    api_key = st.secrets.get(
        "OPENAI_API_KEY",
        LOCAL_API_KEY
    )

    if not api_key:

        st.error("❌ OpenAI API Key Missing")

        st.code("""
OPENAI_API_KEY = "your_key_here"
""")

        st.stop()

    # OPENAI CLIENT
    client = OpenAI(api_key=api_key)

    # TEST CONNECTION
    client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "user", "content": "hello"}
        ],
        max_tokens=5
    )

    st.sidebar.success("✅ OpenAI Connected")

except Exception as e:

    st.error(f"""
❌ OpenAI API Error

{e}
""")

    st.info("""
Possible Fixes:
- Invalid API key
- Billing issue
- Quota exceeded
- Wrong model name
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
    "Top Chunks",
    1,
    10,
    4
)

# ---------------- EMBEDDING MODEL ----------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

with st.spinner("🔄 Loading embedding model..."):

    embedding_model = load_embedding_model()

# ---------------- FILE UPLOAD ----------------

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

    temp_path = None

    try:

        # SAVE TEMP FILE
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as temp_file:

            temp_file.write(uploaded_file.read())

            temp_path = temp_file.name

        # READ PDF
        with st.spinner("📖 Reading PDF..."):

            reader = PdfReader(temp_path)

            full_text = ""

            for page in reader.pages:

                text = page.extract_text()

                if text:

                    full_text += text + "\n"

        if not full_text.strip():

            st.error("❌ No readable text found")

            st.stop()

        # CREATE CHUNKS
        texts = chunk_text(
            full_text,
            chunk_size,
            chunk_overlap
        )

        st.success(f"✅ {len(texts)} chunks created")

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

        # BUILD FAISS INDEX
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatL2(dimension)

        index.add(embeddings)

        # ---------------- QUESTION ----------------

        st.subheader("💬 Ask Question")

        question = st.text_input(
            "Enter your question"
        )

        if st.button("🚀 Generate Answer"):

            if not question.strip():

                st.warning("⚠️ Please enter a question")

            else:

                # EMBED QUESTION
                q_embedding = embedding_model.encode(
                    [question],
                    convert_to_numpy=True
                )

                q_embedding = np.array(
                    q_embedding,
                    dtype=np.float32
                )

                # SEARCH
                with st.spinner("🔍 Searching..."):

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
You are a Study Assistant AI.

Answer ONLY from the given PDF context.

If answer is unavailable, say:
"I could not find the answer in the uploaded PDF."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

                # OPENAI RESPONSE
                try:

                    with st.spinner("🤖 OpenAI Thinking..."):

                        response = client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are a helpful study assistant."
                                },
                                {
                                    "role": "user",
                                    "content": prompt
                                }
                            ],
                            temperature=0.3,
                            max_tokens=500
                        )

                    answer = response.choices[0].message.content

                    # OUTPUT
                    st.subheader("📌 Answer")

                    st.markdown(f"""
<div class="answer-box">

{answer}

</div>
""", unsafe_allow_html=True)

                    # CONTEXT VIEWER
                    with st.expander("📚 Retrieved Context"):

                        for i, chunk in enumerate(retrieved_chunks):

                            st.markdown(f"### Chunk {i+1}")

                            st.markdown(f"""
<div class="context-box">

{chunk}

</div>
""", unsafe_allow_html=True)

                except Exception as e:

                    st.error(f"""
❌ OpenAI Response Error

{e}
""")

    except Exception as e:

        st.error(f"""
❌ PDF Processing Error

{e}
""")

    finally:

        if temp_path and os.path.exists(temp_path):

            os.unlink(temp_path)

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "🚀 Built with Streamlit + OpenAI + FAISS"
)
