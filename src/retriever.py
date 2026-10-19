import os

from langchain_community.vectorstores import FAISS
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


def get_all_chunks(doc_id: str | None = None):
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
            docs = vs.get_by_ids(ordered_ids)
        else:
            docs = [
                vs.docstore.search(doc_id_key)
                for doc_id_key in ordered_ids
            ]

        chunks = []
        for doc in docs:
            if hasattr(doc, "page_content"):
                chunks.append(doc.page_content)
            elif isinstance(doc, str) and not doc.startswith("ID "):
                chunks.append(doc)
        return chunks

    # Fallback to docstore values if index_to_docstore_id is not present or empty
    if hasattr(vs, "docstore") and hasattr(vs.docstore, "_dict"):
        chunks = []
        for doc in vs.docstore._dict.values():
            if hasattr(doc, "page_content"):
                chunks.append(doc.page_content)
            elif isinstance(doc, str) and not doc.startswith("ID "):
                chunks.append(doc)
        return chunks

    return []


def retrieve(
    query,
    doc_id=None,
    k=None,
):
    # Handle backward-compatible positional k call: retrieve(query, k)
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

    return [
        doc.page_content
        for doc in docs
    ]