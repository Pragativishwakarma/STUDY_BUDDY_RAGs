
# 📄 PDF Q&A with Gemini

An AI-powered PDF Question Answering application built using **Streamlit**, **Google Gemini API**, **Sentence Transformers**, and **FAISS**.

This application allows users to upload PDF documents and ask questions related to the uploaded content. The system extracts text from the PDF, converts it into embeddings using Sentence Transformers, stores embeddings inside a FAISS vector database, retrieves the most relevant context, and generates answers using Google's Gemini AI model.

---
That error usually happens on GitHub when:

* The file is too large
* The browser editor failed
* You copied invalid markdown formatting
* Another tab/session changed the file
* GitHub web editor timed out

Try these fixes:

### ✅ Fix 1: Refresh and Edit Again

1. Refresh GitHub page
2. Open `README.md`
3. Click ✏️ Edit
4. Paste content again
5. Commit changes

---

### ✅ Fix 2: Create README.md Locally

Create a file named:

```bash
README.md
```

Paste the full content inside it.

Then run:

```bash
git add README.md
git commit -m "Added README file"
git push origin main
```

---

### ✅ Fix 3: Remove Extra Markdown Blocks

Sometimes nested triple backticks break GitHub editor.

Use this simplified structure:

```markdown
# Project Name

## Installation

pip install -r requirements.txt
```

Avoid very large nested code blocks.

---

### ✅ Fix 4: Use VS Code

1. Open project folder in Visual Studio Code
2. Create `README.md`
3. Paste content
4. Save
5. Push using Git

---

### ✅ Fix 5: GitHub Desktop

Use GitHub Desktop:

* Clone repository
* Add README.md
* Commit
* Push

---

### ✅ Quick Minimal README (Safe Version)

If GitHub still errors, first commit this smaller version:

```markdown
# PDF Q&A with Gemini

AI-powered PDF Question Answering app using:
- Streamlit
- Gemini API
- Sentence Transformers
- FAISS

## Installation

pip install -r requirements.txt

## Run

streamlit run app.py
```

Then expand it later.
