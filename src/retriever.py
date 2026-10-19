import os

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config import (
    VECTORSTORE_DIR,
    EMBEDDING_MODEL,
    RETRIEVAL_K,
)

_vectorstore = None
_embeddings = None


def vectorstore_exists():
    return os.path.exists(
        os.path.join(
            VECTORSTORE_DIR,
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


def _load():
    global _vectorstore

    if _vectorstore is not None:
        return

    if not vectorstore_exists():
        raise FileNotFoundError(
            "No vector store found. Upload a PDF first."
        )

    _vectorstore = FAISS.load_local(
        VECTORSTORE_DIR,
        _get_embeddings(),
        allow_dangerous_deserialization=True,
    )


def reload():
    global _vectorstore

    _vectorstore = None
    _load()


def get_all_chunks():
    _load()

    if _vectorstore is None:
        return []

    # FAISS maintains index_to_docstore_id where integer keys correspond to insertion order
    if hasattr(_vectorstore, "index_to_docstore_id") and _vectorstore.index_to_docstore_id:
        try:
            sorted_indices = sorted(
                _vectorstore.index_to_docstore_id.keys(),
                key=lambda x: int(x) if str(x).isdigit() else str(x),
            )
        except Exception:
            sorted_indices = list(_vectorstore.index_to_docstore_id.keys())

        ordered_ids = [
            _vectorstore.index_to_docstore_id[idx]
            for idx in sorted_indices
        ]

        if hasattr(_vectorstore, "get_by_ids"):
            docs = _vectorstore.get_by_ids(ordered_ids)
        else:
            docs = [
                _vectorstore.docstore.search(doc_id)
                for doc_id in ordered_ids
            ]

        chunks = []
        for doc in docs:
            if hasattr(doc, "page_content"):
                chunks.append(doc.page_content)
            elif isinstance(doc, str) and not doc.startswith("ID "):
                chunks.append(doc)
        return chunks

    # Fallback to docstore values if index_to_docstore_id is not present or empty
    if hasattr(_vectorstore, "docstore") and hasattr(_vectorstore.docstore, "_dict"):
        chunks = []
        for doc in _vectorstore.docstore._dict.values():
            if hasattr(doc, "page_content"):
                chunks.append(doc.page_content)
            elif isinstance(doc, str) and not doc.startswith("ID "):
                chunks.append(doc)
        return chunks

    return []


def retrieve(
    query,
    k=RETRIEVAL_K,
):
    _load()

    docs = _vectorstore.similarity_search(
        query,
        k=k,
    )

    return [
        doc.page_content
        for doc in docs
    ]