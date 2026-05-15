import streamlit as st
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import tempfile
import os
from pypdf import PdfReader
import time

# ---------------- FIX TOKENIZER WARNING ----------------

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------- PAGE CONFIG ----------------

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
    border-radius: 10px;
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

st.title("📚 Study Buddy RAG with Gemini")
st.caption("Upload PDF → Ask Questions → Get AI Answers")

# ---------------- GEMINI API CONFIG ----------------

try:

    # SAFE SECRET ACCESS
    api_key = st.secrets.get("GOOGLE_API_KEY")

    if not api_key:
        st.error("❌ Gemini API Key not found in Streamlit Secrets")

        st.code("""
GOOGLE_API_KEY = "your_api_key_here"
""")

        st.info("""
📌 Add this inside Streamlit Cloud:

Settings → Secrets
""")

        st.stop()

    # CONFIGURE GEMINI
    genai.configure(api_key=api_key)

    # TEST CONNECTION
    try:
        test_model = genai.GenerativeModel(
            "gemini-1.5-flash"
        )

        test_response = test_model.generate_content(
            "Hello"
        )

        if test_response:
            st.sidebar.success("✅ Gemini API Connected")

    except Exception as gemini_error:

        st.error(f"""
❌ Gemini API Error:

{gemini_error}
""")

        st.info("""
Possible reasons:
- Invalid API Key
- Expired API Key
- Billing/Quota issue
- Wrong Gemini model
""")

        st.stop()

except Exception as e:

    st.error(f"❌ API Setup Error: {e}")

    st.stop()

# ---------------- SIDEBAR ----------------

st.sidebar.header("⚙️ Settings")

chunk_size = st.sidebar.slider(
    "Chunk Size",
    min_value=200,
    max_value=1000,
    value=500,
    step=100
)

chunk_overlap = st.sidebar.slider(
    "Chunk Overlap",
    min_value=0,
    max_value=200,
    value=50,
    step=10
)

top_k = st.sidebar.slider(
    "Top Context Chunks",
    min_value=1,
    max_value=10,
    value=4
)

# ---------------- EMBEDDING MODEL CACHE ----------------

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

# ---------------- LOAD MODEL ----------------

try:

    with st.spinner("🔄 Loading embedding model..."):

        embedding_model = load_embedding_model()

except Exception as e:

    st.error(f"""
❌ Failed to load embedding model.

Error:
{e}
""")

    st.stop()

# ---------------- FILE UPLOAD ----------------

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type="pdf"
)

# ---------------- TEXT CHUNK FUNCTION ----------------

def chunk_text(text, chunk_size=500, overlap=50):

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

# ---------------- MAIN PROCESS ----------------

if uploaded_file:

    st.success(
        f"✅ Uploaded: {uploaded_file.name}"
    )

    tmp_path = None

    try:

        # ---------------- SAVE TEMP PDF ----------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp:

            tmp.write(uploaded_file.read())

            tmp_path = tmp.name

        # ---------------- PDF EXTRACTION ----------------

        with st.spinner("📖 Extracting text from PDF..."):

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

        # ---------------- EMPTY PDF CHECK ----------------

        if full_text.strip() == "":

            st.error(
                "❌ No readable text found in PDF"
            )

            st.stop()

        # ---------------- CHUNKING ----------------

        texts = chunk_text(
            full_text,
            chunk_size=chunk_size,
            overlap=chunk_overlap
        )

        if len(texts) == 0:

            st.error(
                "❌ Failed to create text chunks"
            )

            st.stop()

        # ---------------- EMBEDDINGS ----------------

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

            time.sleep(1)

        # ---------------- FAISS INDEX ----------------

        with st.spinner(
            "⚡ Building vector database..."
        ):

            dimension = embeddings.shape[1]

            index = faiss.IndexFlatL2(
                dimension
            )

            index.add(embeddings)

            time.sleep(1)

        # ---------------- SUCCESS ----------------

        st.success(
            f"""
✅ PDF processed successfully!

📄 Chunks Created: {len(texts)}
"""
        )

        # ---------------- QUESTION SECTION ----------------

        st.subheader("💬 Ask Questions")

        question = st.text_input(
            "Enter your question"
        )

        # ---------------- GENERATE BUTTON ----------------

        if st.button("🚀 Generate Answer"):

            if question.strip() == "":

                st.warning(
                    "⚠️ Please enter a question"
                )

            else:

                # ---------------- SEARCH CONTEXT ----------------

                with st.spinner(
                    "🔍 Searching relevant context..."
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

                # ---------------- RETRIEVE CHUNKS ----------------

                retrieved_chunks = []

                for i in indices[0]:

                    if i < len(texts):
                        retrieved_chunks.append(
                            texts[i]
                        )

                context = "\n\n".join(
                    retrieved_chunks
                )

                # ---------------- PROMPT ----------------

                prompt = f"""
You are an intelligent AI Study Assistant.

Answer the user's question ONLY using the provided context.

If the answer is not available in the context,
reply exactly:

"I could not find the answer in the uploaded PDF."

---------------- CONTEXT ----------------

{context}

---------------- QUESTION ----------------

{question}

---------------- ANSWER ----------------
"""

                # ---------------- GEMINI RESPONSE ----------------

                try:

                    llm = genai.GenerativeModel(
                        "gemini-1.5-flash"
                    )

                    with st.spinner(
                        "🤖 Gemini is thinking..."
                    ):

                        response = llm.generate_content(
                            prompt
                        )

                    # ---------------- SAFE RESPONSE ----------------

                    answer = "No response generated."

                    if response:

                        if hasattr(response, "text"):

                            answer = response.text

                        elif (
                            hasattr(response, "candidates")
                            and response.candidates
                        ):

                            answer = response.candidates[
                                0
                            ].content.parts[0].text

                    # ---------------- OUTPUT ----------------

                    st.subheader("📌 Answer")

                    st.markdown(f"""
<div class="answer-box">

{answer}

</div>
""", unsafe_allow_html=True)

                    # ---------------- CONTEXT VIEWER ----------------

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

                    # ---------------- METRICS ----------------

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
                        "PDF Size",
                        f"{round(len(full_text)/1000, 2)}K chars"
                    )

                except Exception as e:

                    st.error(f"""
❌ Gemini Error

{e}
""")

                    st.info("""
Fixes:
- Check API Key
- Use latest Gemini model
- Verify internet connection
- Verify billing/quota
""")

    except Exception as e:

        st.error(f"""
❌ Error Processing PDF

{e}
""")

    finally:

        # ---------------- DELETE TEMP FILE ----------------

        if (
            tmp_path
            and os.path.exists(tmp_path)
        ):

            os.unlink(tmp_path)

# ---------------- FOOTER ----------------

st.markdown("---")

st.caption(
    "🚀 Built with Streamlit + Gemini + FAISS + Sentence Transformers"
)
