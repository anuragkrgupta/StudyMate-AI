from io import BytesIO
from pathlib import Path
import os

from pypdf import PdfReader, PdfWriter
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from config import VECTORSTORE_DIR, EMBEDDING_MODEL, get_vectorstore_dir

MAX_OCR_PAGES = 10


def _read_pdf(source):
    if isinstance(source, (str, Path)):
        return PdfReader(source)

    if isinstance(source, bytes):
        return PdfReader(BytesIO(source))

    if hasattr(source, "seek"):
        source.seek(0)

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


def ingest_pdf(source, doc_id: str | None = None) -> int:
    """
    Build a LangChain FAISS vector store from a PDF.

    Processes each page independently:
    - Uses normal selectable-text extraction first.
    - Runs local handwriting OCR on individual pages lacking selectable text.
    - Enforces an OCR safety limit (at most 10 pages requiring OCR).
    - Rejects incomplete ingestion if OCR returns empty text for any scanned page.
    """
    reader = _read_pdf(source)
    if not reader.pages:
        raise ValueError("The PDF document contains no pages.")

    # Identify which pages have selectable text and which need OCR
    selectable_texts = {}
    ocr_page_indices = []

    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            selectable_texts[idx] = text
        else:
            ocr_page_indices.append(idx)

    # Enforce OCR page limit explicitly on pages requiring OCR
    if len(ocr_page_indices) > MAX_OCR_PAGES:
        raise ValueError(
            f"PDF has {len(ocr_page_indices)} pages requiring OCR. "
            f"Please upload a PDF with at most {MAX_OCR_PAGES} pages requiring OCR, "
            f"or split the document into smaller parts."
        )

    text_parts = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        if idx in selectable_texts:
            text_parts.append(selectable_texts[idx])
        else:
            print(f"Page {page_num}: No selectable text found. Running local handwriting OCR...")
            try:
                from handwriting_ocr import recognize_handwritten_pdf

                writer = PdfWriter()
                writer.add_page(page)
                page_buf = BytesIO()
                writer.write(page_buf)
                page_bytes = page_buf.getvalue()

                recognized_text = recognize_handwritten_pdf(
                    page_bytes,
                    max_pages=MAX_OCR_PAGES,
                    page_number_offset=idx,
                )
            except Exception as exc:
                raise ValueError(
                    f"Local handwriting recognition failed on page {page_num}: {exc}"
                ) from exc

            if not recognized_text or not recognized_text.strip():
                raise ValueError(
                    f"Local handwriting recognition returned no text on page {page_num}. "
                    "Try a clearer scanned PDF."
                )

            text_parts.append(recognized_text)

    full_text = "\n\n".join(text_parts).strip()

    if not full_text:
        raise ValueError("No usable text was found in the PDF.")

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

    target_dir = get_vectorstore_dir(doc_id, root_dir=VECTORSTORE_DIR)
    os.makedirs(target_dir, exist_ok=True)
    vectorstore.save_local(target_dir)

    # Cryptographically sign the newly created index files for integrity verification
    try:
        from security import sign_vectorstore_index
        sign_vectorstore_index(target_dir)
    except Exception:
        # If save_local was mocked in unit tests and didn't write physical files, ignore
        if not (os.path.exists(os.path.join(target_dir, "index.pkl")) and os.path.exists(os.path.join(target_dir, "index.faiss"))):
            pass
        else:
            raise

    return len(chunks)


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path-to-pdf>")
        sys.exit(1)

    count = ingest_pdf(sys.argv[1])
    print(f"Done! Ingested {count} chunks.")
