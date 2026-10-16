
from io import BytesIO
from pathlib import Path
import os

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import VECTORSTORE_DIR, EMBEDDING_MODEL


def _read_pdf(source):
    if isinstance(source, (str, Path)):
        return PdfReader(source)

    if isinstance(source, bytes):
        return PdfReader(BytesIO(source))

    return PdfReader(source)


def _get_pdf_bytes(source):
    """Read PDF bytes without unnecessarily changing stream position."""
    if isinstance(source, bytes):
        return source

    if isinstance(source, (str, Path)):
        return Path(source).read_bytes()

    if hasattr(source, "getvalue"):
        return source.getvalue()

    position = source.tell() if hasattr(source, "tell") else None

    try:
        if hasattr(source, "seek"):
            source.seek(0)
        return source.read()
    finally:
        if position is not None and hasattr(source, "seek"):
            source.seek(position)


def ingest_pdf(source) -> int:
    """
    Build a LangChain FAISS vector store from a PDF.

    Uses normal PDF text extraction first, then local handwriting
    OCR if no selectable text is found.
    """
    reader = _read_pdf(source)
    text_parts = []

    for page in reader.pages:
        text = page.extract_text()
        if text and text.strip():
            text_parts.append(text)

    full_text = "\n\n".join(text_parts).strip()

    # OCR fallback for fully scanned or handwritten PDFs.
    if not full_text:
        print("No selectable text found. Starting local handwriting OCR...")

        try:
            from handwriting_ocr import recognize_handwritten_pdf

            full_text = recognize_handwritten_pdf(_get_pdf_bytes(source))
        except Exception as exc:
            raise ValueError(
                f"Local handwriting recognition failed: {exc}"
            ) from exc

        if not full_text or not full_text.strip():
            raise ValueError(
                "No text could be recognized. Try a clearer scanned PDF."
            )

        print("Handwriting OCR completed successfully.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
    )

    chunks = splitter.split_text(full_text)

    if not chunks:
        raise ValueError("No usable text was found in the PDF.")

    print(f"Total Chunks: {len(chunks)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL
    )

    vectorstore = FAISS.from_texts(
        texts=chunks,
        embedding=embeddings,
    )

    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    vectorstore.save_local(VECTORSTORE_DIR)

    return len(chunks)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path-to-pdf>")
        sys.exit(1)

    count = ingest_pdf(sys.argv[1])
    print(f"Done! Ingested {count} chunks.")
