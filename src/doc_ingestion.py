# src/doc_ingestion.py

from pathlib import Path
from typing import List
import io
from pypdf import PdfReader



def load_pdf_text(path: str | Path) -> str:
    """
    Read all text from a PDF file and return it as one big string.
    """
    path = Path(path)
    reader = PdfReader(str(path))
    pages_text: List[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    full_text = "\n".join(pages_text)
    return full_text


def chunk_text(
    text: str,
    chunk_size: int = 800,
    overlap: int = 200,
) -> List[str]:
    """
    Split text into overlapping chunks for RAG.
    """
    clean_text = " ".join(text.split())  # remove weird line breaks
    chunks: List[str] = []

    start = 0
    length = len(clean_text)

    while start < length:
        end = min(start + chunk_size, length)
        chunk = clean_text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == length:
            break
        start = max(0, end - overlap)

    return chunks
def load_pdf_text_from_bytes(data: bytes) -> str:
    """
    Read all text from a PDF given as raw bytes.
    """
    reader = PdfReader(io.BytesIO(data))
    pages_text: List[str] = []

    for page in reader.pages:
        text = page.extract_text() or ""
        pages_text.append(text)

    return "\n".join(pages_text)
