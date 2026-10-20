import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
    get_vectorstore_dir,
)

_vectorstore = None
_vectorstores = {}
_embeddings = None


def vectorstore_exists(doc_id: str | None = None):
    target_dir = get_vectorstore_dir(doc_id, root_dir=VECTORSTORE_DIR)
    return os.path.exists(
        os.path.join(
            target_dir,
            "index.faiss",
        )
    )


def _get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL
        )

    return _embeddings


def _load(doc_id: str | None = None):
    global _vectorstore

    target_dir = get_vectorstore_dir(doc_id, root_dir=VECTORSTORE_DIR)
    cache_key = target_dir

    if doc_id is None and _vectorstore is not None:
        return _vectorstore

    if cache_key in _vectorstores:
        if doc_id is None:
            _vectorstore = _vectorstores[cache_key]
        return _vectorstores[cache_key]

    if not vectorstore_exists(doc_id):
        doc_msg = f" for document '{doc_id}'" if doc_id else ""
        raise FileNotFoundError(
            f"No vector store found{doc_msg}. Upload a PDF first."
        )

    # Cryptographically verify index integrity and authenticity BEFORE deserializing pickle
    from security import verify_vectorstore_index
    verify_vectorstore_index(target_dir)

    vs = FAISS.load_local(
        target_dir,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )
    _vectorstores[cache_key] = vs
    if doc_id is None:
        _vectorstore = vs
    return vs


def reload(doc_id: str | None = None):
    global _vectorstore

    target_dir = get_vectorstore_dir(doc_id, root_dir=VECTORSTORE_DIR)
    cache_key = target_dir

    _vectorstores.pop(cache_key, None)
    if doc_id is None:
        _vectorstore = None

    return _load(doc_id)


def get_all_documents(doc_id: str | None = None) -> list[Document]:
    """Retrieve all Document objects from the vector store in insertion order."""
    vs = _load(doc_id)

    if vs is None:
        return []

    # FAISS maintains index_to_docstore_id where integer keys correspond to insertion order
    if hasattr(vs, "index_to_docstore_id") and vs.index_to_docstore_id:
        try:
            sorted_indices = sorted(
                vs.index_to_docstore_id.keys(),
                key=lambda x: int(x) if str(x).isdigit() else str(x),
            )
        except Exception:
            sorted_indices = list(vs.index_to_docstore_id.keys())

        ordered_ids = [
            vs.index_to_docstore_id[idx]
            for idx in sorted_indices
        ]

        if hasattr(vs, "get_by_ids"):
            raw_docs = vs.get_by_ids(ordered_ids)
        else:
            raw_docs = [
                vs.docstore.search(doc_id_key)
                for doc_id_key in ordered_ids
            ]

        docs = []
        for doc in raw_docs:
            if isinstance(doc, Document):
                docs.append(doc)
            elif hasattr(doc, "page_content"):
                docs.append(Document(page_content=doc.page_content, metadata=getattr(doc, "metadata", {}) or {}))
            elif isinstance(doc, str) and not doc.startswith("ID "):
                docs.append(Document(page_content=doc, metadata={}))
        return docs

    # Fallback to docstore values if index_to_docstore_id is not present or empty
    if hasattr(vs, "docstore") and hasattr(vs.docstore, "_dict"):
        docs = []
        for doc in vs.docstore._dict.values():
            if isinstance(doc, Document):
                docs.append(doc)
            elif hasattr(doc, "page_content"):
                docs.append(Document(page_content=doc.page_content, metadata=getattr(doc, "metadata", {}) or {}))
            elif isinstance(doc, str) and not doc.startswith("ID "):
                docs.append(Document(page_content=doc, metadata={}))
        return docs

    return []


def get_all_chunks(doc_id: str | None = None) -> list[str]:
    """Retrieve all chunk text strings belonging to the active document."""
    docs = get_all_documents(doc_id=doc_id)
    return [
        doc.page_content
        for doc in docs
        if hasattr(doc, "page_content")
    ]


def retrieve_with_metadata(
    query: str,
    doc_id: str | None = None,
    k: int | None = None,
) -> list[Document]:
    """Retrieve top-k Document objects with chunk content and page metadata."""
    # Handle backward-compatible positional k call: retrieve_with_metadata(query, k)
    if isinstance(doc_id, int) and k is None:
        k = doc_id
        doc_id = None
    if k is None:
        k = RETRIEVAL_K

    vs = _load(doc_id)

    docs = vs.similarity_search(
        query,
        k=k,
    )

    return docs


# Alias for metadata-aware retrieval
retrieve_documents = retrieve_with_metadata


def retrieve(
    query,
    doc_id=None,
    k=None,
):
    """Retrieve top-k chunk text strings for backward compatibility."""
    docs = retrieve_with_metadata(
        query,
        doc_id=doc_id,
        k=k,
    )

    return [
        doc.page_content if hasattr(doc, "page_content") else str(doc)
        for doc in docs
    ]