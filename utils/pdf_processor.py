import fitz  
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
import io


def process_pdfs(uploaded_files):
    """
    Extract text from uploaded PDFs, chunk them, and return:
    - chunks: list of LangChain Document objects with metadata
    - doc_texts: dict of {filename: full_text} for summarization/quiz
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "],
    )

    all_chunks = []
    doc_texts = {}

    for file in uploaded_files:
        pdf_bytes = file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        full_text = ""
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text()
            full_text += text

            # Create page-level chunks with metadata
            if text.strip():
                chunks = splitter.create_documents(
                    [text],
                    metadatas=[{"source": file.name, "page": page_num}],
                )
                all_chunks.extend(chunks)

        doc_texts[file.name] = full_text[:15000]  # cap for LLM context

    return all_chunks, doc_texts
