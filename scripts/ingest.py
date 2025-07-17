import os
from typing import List
from PyPDF2 import PdfReader
from docx import Document
import re

import nltk
nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))
from nltk.tokenize import sent_tokenize


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

def sliding_window_chunk(text: str, chunk_size: int = 300, overlap: int = 100) -> List[str]:
    """
    Chunking strategy that splits pdf into 300 word sizes with 100 word overlaps with
    adjacent chunks.
    """
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

def semantic_chunk(text: str, max_words: int = 200) -> List[str]:
    """
    Split text into semantic chunks by grouping sentences so each
    chunk is based on max words, preserving sentence boundaries.
    """
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = []
    current_length = 0

    for sentence in sentences:
        sentence_length = len(sentence.split()) 

        # Handle long sentences
        if sentence_length > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            words = sentence.split()
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i+max_words]))
            continue

        if current_length + sentence_length > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_sentences = current_chunk[-1:]
            current_chunk = overlap_sentences + [sentence]
            current_length = sum(len(s.split()) for s in current_chunk)
        else:
            current_chunk.append(sentence)
            current_length += sentence_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def ingest(file_path: str, chunk_size: int = 200) -> List[str]:
    """
    Full ingestion pipeline: load + chunk.
    Returns a list of text chunks.
    """
    print(f"[INFO] Loading file: {file_path}")
    text = load_file(file_path)
    print(f"[DEBUG] First 500 characters of raw text:\n{text[:500]}")
    print(f"[INFO] Document length: {len(text)} characters")

    chunks = semantic_chunk(text, max_words=chunk_size)
    print(f"[INFO] Split into {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    file_path = "data/notes.pdf"
    chunks = ingest(file_path)

    print(f"\n[INFO] {len(chunks)} chunks generated:")
    print("\n--- First Chunk ---\n")
    print(chunks[70])