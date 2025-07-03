import os
from typing import List
from PyPDF2 import PdfReader
from docx import Document


def extract_text_from_pdf(file_path: str) -> str:
    """Extracts text from a PDF file."""
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return text.strip()


def extract_text_from_docx(file_path: str) -> str:
    """Extracts text from a Word DOCX file."""
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs]).strip()


def load_file(file_path: str) -> str:
    """Loads text from a PDF or DOCX file."""
    if file_path.lower().endswith(".pdf"):
        return extract_text_from_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        return extract_text_from_docx(file_path)
    else:
        raise ValueError("Unsupported file format. Use PDF or DOCX.")


def chunk_text(text: str, max_words: int = 200) -> List[str]:
    """
    Splits the input text into chunks of `max_words` words each.
    Adjust `max_words` based on model input length limits.
    """
    words = text.split()
    chunks = [
        " ".join(words[i:i + max_words])
        for i in range(0, len(words), max_words)
    ]
    return chunks


def ingest(file_path: str, chunk_size: int = 200) -> List[str]:
    """
    Full ingestion pipeline: load + chunk.
    Returns a list of text chunks.
    """
    print(f"[INFO] Loading file: {file_path}")
    text = load_file(file_path)
    print(f"[INFO] Document length: {len(text)} characters")

    chunks = chunk_text(text, max_words=chunk_size)
    print(f"[INFO] Split into {len(chunks)} chunks")
    return chunks