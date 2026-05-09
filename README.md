# AI Study Assistant – RAG-Based Q&A Application

## Live Demo
https://ai-study-assistant-using-rag-8jvcbvqfukdtb8fvqoyywf.streamlit.app/

---

## Overview

AI Study Assistant is a Retrieval-Augmented Generation (RAG) based web application designed to assist students in:

* Getting instant, context-aware answers directly from their uploaded PDFs and notes
* Summarizing lengthy study materials in multiple formats
* Auto-generating quiz questions to test their understanding

The project demonstrates a complete end-to-end AI pipeline including document processing, vector embeddings, semantic search, and LLM integration.

---

## Key Features

### RAG-Based Q&A
* Upload multiple PDFs simultaneously(upto 50 MB)
* Answers are retrieved directly from your documents with source citations
* Shows exactly which document and page the answer came from

### Hybrid Intelligence Mode
* If the question is not found in the uploaded documents, the app intelligently falls back to Groq AI general knowledge
* Clearly indicates whether the answer came from 📄 your documents or 🧠 AI knowledge

### Summary Generator
* Generates structured summaries of any uploaded document
* Three summary styles: Bullet Points, Detailed Paragraph, and Key Concepts Only

### Quiz Mode
* Auto-generates multiple choice questions from your study material
* Adjustable difficulty: Easy, Medium, Hard

---

## Technology Stack

* Python
* Streamlit
* LangChain
* FAISS (Facebook AI Similarity Search)
* Sentence Transformers
* Groq API (LLaMA 3.1 8B)
* PyMuPDF

---

## How It Works

```
User Uploads PDF
      ↓
Text Extraction & Chunking (PyMuPDF + LangChain)
      ↓
Vector Embeddings (Sentence Transformers)
      ↓
FAISS Vector Store
      ↓
User Asks a Question
      ↓
Semantic Search → Cosine Similarity Score
      ↓
Score High → 📄 RAG Mode (Answer from Documents)
Score Low  → 🧠 AI Mode (Groq General Knowledge)
```

---

## Project Structure

```bash
ai-study-assistant/
│
├── app.py
├── requirements.txt
├── utils/
│   ├── __init__.py
│   ├── pdf_processor.py
│   ├── vector_store.py
│   ├── llm.py
│   └── helpers.py
└── styles/
    └── main.css
```

---

## Running the Application Locally

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Add your Groq API key — create `.streamlit/secrets.toml`:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
4. Run the application:
   ```
   streamlit run app.py
   ```

Get a free Groq API key at https://console.groq.com

---

## Future Enhancements

* Add conversation memory across sessions
* Support for additional file formats (DOCX, TXT)
* Multi-language support for answers
* Export chat history as PDF

---

## Author

Ashwatha
