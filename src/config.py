import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

try:
    from security import load_env_file
    load_env_file(os.path.join(BASE_DIR, ".env"))
except Exception:
    pass

VECTORSTORE_DIR = os.path.join(BASE_DIR, "vectorstore")
CHUNKS_PATH = os.path.join(VECTORSTORE_DIR, "chunks.pkl")
FAISS_INDEX_PATH = os.path.join(VECTORSTORE_DIR, "faiss_index.bin")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"    
RETRIEVAL_K = 4


def sanitize_doc_id(doc_id: str) -> str:
    """Return a filesystem-safe directory name for a document identifier."""
    if not doc_id:
        return ""
    if ":" in doc_id:
        name_part, digest_part = doc_id.rsplit(":", 1)
        safe_name = "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in name_part
        ).strip("_")
        safe_digest = "".join(c for c in digest_part if c.isalnum())
        if safe_name and safe_digest:
            return f"{safe_name}_{safe_digest}"
        if safe_digest:
            return safe_digest

    import hashlib
    safe = "".join(
        c if c.isalnum() or c in ("-", "_") else "_"
        for c in doc_id
    ).strip("_")
    if not safe:
        return hashlib.sha256(doc_id.encode("utf-8")).hexdigest()[:16]
    return safe


def get_vectorstore_dir(
    doc_id: str | None = None,
    root_dir: str | None = None,
) -> str:
    """Resolve the vector store directory for a specific document or default root."""
    base = root_dir if root_dir is not None else VECTORSTORE_DIR
    if not doc_id:
        return base
    safe_id = sanitize_doc_id(doc_id)
    return os.path.join(base, safe_id)
