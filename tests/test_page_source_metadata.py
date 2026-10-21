import io
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pymupdf
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

import config
import ingest
import retriever
from app import _format_source_label, _render_source_attribution
from security import sign_vectorstore_index


def make_pdf(pages_content):
    """Create an in-memory PDF from a list of page text contents."""
    doc = pymupdf.open()
    for content in pages_content:
        page = doc.new_page(width=595, height=842)
        if content and content.strip():
            page.insert_text((50, 100), content.strip())
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestPageSourceMetadata(unittest.TestCase):
    """
    Comprehensive regression test suite for page-level source metadata in StudyMate AI.
    Covers Tests A through I as specified in the requirements.
    """

    def setUp(self):
        os.environ["STUDYMATE_INDEX_SIGNING_KEY"] = (
            "test-signing-key-for-automated-regression-tests-only-32chars"
        )
        self.temp_dir = tempfile.mkdtemp()
        self.fake_embeddings = FakeEmbeddings(size=10)

        # Patch VECTORSTORE_DIR in config, ingest, and retriever
        self.config_patcher = patch.object(config, "VECTORSTORE_DIR", self.temp_dir)
        self.ingest_patcher = patch.object(ingest, "VECTORSTORE_DIR", self.temp_dir)
        self.retriever_patcher = patch.object(retriever, "VECTORSTORE_DIR", self.temp_dir)

        self.config_patcher.start()
        self.ingest_patcher.start()
        self.retriever_patcher.start()

        # Patch embeddings in ingest and retriever for fast and deterministic unit testing
        self.ingest_emb_patcher = patch("ingest.HuggingFaceEmbeddings", return_value=self.fake_embeddings)
        self.retriever_emb_patcher = patch("retriever.HuggingFaceEmbeddings", return_value=self.fake_embeddings)
        self.ingest_emb_patcher.start()
        self.retriever_emb_patcher.start()

        # Clear retriever in-memory cache
        retriever._vectorstore = None
        retriever._vectorstores.clear()
        retriever._embeddings = self.fake_embeddings

    def tearDown(self):
        self.retriever_emb_patcher.stop()
        self.ingest_emb_patcher.stop()
        self.retriever_patcher.stop()
        self.ingest_patcher.stop()
        self.config_patcher.stop()

        retriever._vectorstore = None
        retriever._vectorstores.clear()
        retriever._embeddings = None

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test A — Text-only PDF metadata
    # -------------------------------------------------------------------------
    def test_a_text_only_pdf_metadata(self):
        """
        Ingest a small text-only PDF and verify that every produced chunk contains:
        - The correct source filename.
        - The correct 1-based page number.
        - The expected document ID when supplied.
        """
        pdf_bytes = make_pdf([
            "Page 1: Digital lecture on Data Structures and Memory Layouts.",
            "Page 2: Tree Traversals, Binary Search Trees, and Heaps.",
        ])

        doc_id = "algo_lecture.pdf:abc12345"
        source_name = "algo_lecture.pdf"

        chunk_count = ingest.ingest_pdf(
            pdf_bytes,
            doc_id=doc_id,
            source_name=source_name,
        )
        self.assertEqual(chunk_count, 2)

        docs = retriever.get_all_documents(doc_id=doc_id)
        self.assertEqual(len(docs), 2)

        # Page 1 chunk verification
        self.assertEqual(docs[0].metadata.get("source"), "algo_lecture.pdf")
        self.assertEqual(docs[0].metadata.get("page"), 1)
        self.assertEqual(docs[0].metadata.get("doc_id"), doc_id)
        self.assertIn("Data Structures", docs[0].page_content)

        # Page 2 chunk verification
        self.assertEqual(docs[1].metadata.get("source"), "algo_lecture.pdf")
        self.assertEqual(docs[1].metadata.get("page"), 2)
        self.assertEqual(docs[1].metadata.get("doc_id"), doc_id)
        self.assertIn("Tree Traversals", docs[1].page_content)

        # Verify page numbers are 1-based integers
        for d in docs:
            page = d.metadata.get("page")
            self.assertIsInstance(page, int)
            self.assertGreaterEqual(page, 1)

    # -------------------------------------------------------------------------
    # Test B — Mixed-PDF metadata
    # -------------------------------------------------------------------------
    def test_b_mixed_pdf_metadata(self):
        """
        Ingest a PDF containing both selectable-text and scanned pages,
        using mocked OCR. Verify that chunks from both page types retain the
        correct original page number and source filename.
        """
        pdf_bytes = make_pdf([
            "Page 1 digital text: Operating System Process Scheduling Algorithms.",
            "",  # Page 2 scanned diagram
            "Page 3 digital text: Virtual Memory and Demand Paging Architecture.",
        ])

        doc_id = "os_mixed.pdf:def67890"
        source_name = "os_mixed.pdf"

        def mock_ocr(page_bytes, max_pages=50, page_number_offset=0):
            return f"Page {page_number_offset + 1} OCR: Handwritten Notes on Round Robin Scheduling"

        with patch("handwriting_ocr.recognize_handwritten_pdf", side_effect=mock_ocr) as mock_ocr_fn:
            chunk_count = ingest.ingest_pdf(
                pdf_bytes,
                doc_id=doc_id,
                source_name=source_name,
            )
            self.assertEqual(chunk_count, 3)
            self.assertEqual(mock_ocr_fn.call_count, 1)

        docs = retriever.get_all_documents(doc_id=doc_id)
        self.assertEqual(len(docs), 3)

        # Page 1 (digital)
        self.assertEqual(docs[0].metadata["source"], source_name)
        self.assertEqual(docs[0].metadata["page"], 1)
        self.assertIn("Process Scheduling", docs[0].page_content)

        # Page 2 (OCR scanned)
        self.assertEqual(docs[1].metadata["source"], source_name)
        self.assertEqual(docs[1].metadata["page"], 2)
        self.assertIn("Handwritten Notes on Round Robin", docs[1].page_content)

        # Page 3 (digital)
        self.assertEqual(docs[2].metadata["source"], source_name)
        self.assertEqual(docs[2].metadata["page"], 3)
        self.assertIn("Virtual Memory", docs[2].page_content)

    # -------------------------------------------------------------------------
    # Test C — Multiple chunks from one page
    # -------------------------------------------------------------------------
    def test_c_multiple_chunks_from_one_page(self):
        """
        Use a page with enough text to create multiple chunks.
        Verify that all chunks retain the same source and page metadata
        without duplicating or corrupting content.
        """
        # Create a single page with ~3600 characters of distinct sentences
        long_paragraph = " ".join([f"Sentence number {i} exploring deep learning." for i in range(120)])
        pdf_bytes = make_pdf([long_paragraph])

        doc_id = "long_doc.pdf:999888"
        source_name = "long_doc.pdf"

        chunk_count = ingest.ingest_pdf(
            pdf_bytes,
            doc_id=doc_id,
            source_name=source_name,
        )
        self.assertGreaterEqual(chunk_count, 3)

        docs = retriever.get_all_documents(doc_id=doc_id)
        self.assertEqual(len(docs), chunk_count)

        for i, d in enumerate(docs):
            self.assertEqual(d.metadata["source"], source_name)
            self.assertEqual(d.metadata["page"], 1)
            self.assertEqual(d.metadata["doc_id"], doc_id)
            self.assertTrue(len(d.page_content) <= 1600)

        # Ensure chunks are not identical copies
        contents = [d.page_content for d in docs]
        self.assertEqual(len(set(contents)), len(contents))

    # -------------------------------------------------------------------------
    # Test D — Original order preserved
    # -------------------------------------------------------------------------
    def test_d_original_order_preserved(self):
        """
        Verify that chunks preserve page order and their order within each page.
        """
        pdf_bytes = make_pdf([
            "Alpha Section Page 1 content.",
            "Beta Section Page 2 content.",
            "Gamma Section Page 3 content.",
            "Delta Section Page 4 content.",
        ])

        doc_id = "order_test.pdf:1234"
        ingest.ingest_pdf(pdf_bytes, doc_id=doc_id, source_name="order_test.pdf")

        docs = retriever.get_all_documents(doc_id=doc_id)
        pages = [d.metadata["page"] for d in docs]
        self.assertEqual(pages, [1, 2, 3, 4])

        chunks = retriever.get_all_chunks(doc_id=doc_id)
        self.assertEqual(len(chunks), 4)
        self.assertIn("Alpha Section", chunks[0])
        self.assertIn("Beta Section", chunks[1])
        self.assertIn("Gamma Section", chunks[2])
        self.assertIn("Delta Section", chunks[3])

    # -------------------------------------------------------------------------
    # Test E — Retrieval exposes metadata
    # -------------------------------------------------------------------------
    def test_e_retrieval_exposes_metadata(self):
        """
        Verify that metadata-aware retrieval helper returns the correct content
        and metadata from the stored FAISS documents.
        """
        pdf_bytes = make_pdf([
            "Database Systems: Relational Algebra and Normal Forms.",
            "Computer Networks: OSI Model and TCP/IP Protocol Suite.",
        ])

        doc_id = "cs_overview.pdf:5555"
        source_name = "cs_overview.pdf"
        ingest.ingest_pdf(pdf_bytes, doc_id=doc_id, source_name=source_name)

        # Test retrieve_with_metadata
        docs = retriever.retrieve_with_metadata(
            "Relational Algebra",
            doc_id=doc_id,
            k=2,
        )

        self.assertIsInstance(docs, list)
        self.assertGreater(len(docs), 0)
        top_doc = docs[0]
        self.assertIsInstance(top_doc, Document)
        self.assertIn("source", top_doc.metadata)
        self.assertEqual(top_doc.metadata["source"], source_name)
        self.assertIn("page", top_doc.metadata)
        self.assertIsInstance(top_doc.metadata["page"], int)
        self.assertEqual(top_doc.metadata["doc_id"], doc_id)

    # -------------------------------------------------------------------------
    # Test F — Existing retrieval compatibility
    # -------------------------------------------------------------------------
    def test_f_existing_retrieval_compatibility(self):
        """
        Verify that existing callers expecting a list of strings continue to
        work seamlessly with retrieve() and get_all_chunks().
        """
        pdf_bytes = make_pdf([
            "First page content for compatibility test.",
            "Second page content for compatibility test.",
        ])

        doc_id = "compat.pdf:7777"
        ingest.ingest_pdf(pdf_bytes, doc_id=doc_id, source_name="compat.pdf")

        # retrieve(query, doc_id) returns list of strings
        chunks = retriever.retrieve("compatibility", doc_id=doc_id, k=2)
        self.assertIsInstance(chunks, list)
        for c in chunks:
            self.assertIsInstance(c, str)
            self.assertNotIsInstance(c, Document)

        # Also ingest into default root index (doc_id=None) to test legacy positional retrieve(query, k)
        ingest.ingest_pdf(pdf_bytes, doc_id=None, source_name="default_compat.pdf")
        chunks_pos = retriever.retrieve("compatibility", 1)
        self.assertIsInstance(chunks_pos, list)
        self.assertEqual(len(chunks_pos), 1)
        for c in chunks_pos:
            self.assertIsInstance(c, str)
            self.assertNotIsInstance(c, Document)

        # get_all_chunks returns list of strings
        all_chunks = retriever.get_all_chunks(doc_id=doc_id)
        self.assertIsInstance(all_chunks, list)
        for c in all_chunks:
            self.assertIsInstance(c, str)
            self.assertNotIsInstance(c, Document)

    # -------------------------------------------------------------------------
    # Test G — Chat source display and deduplication
    # -------------------------------------------------------------------------
    def test_g_chat_source_display_and_deduplication(self):
        """
        Verify that source labels are cleanly formatted, deduplicated across
        multiple chunks from the same page, and handle missing metadata gracefully.
        """
        # 1. Test _format_source_label formatting
        self.assertEqual(
            _format_source_label("lecture.pdf", 3),
            "lecture.pdf (Page 3)",
        )
        self.assertEqual(
            _format_source_label("lecture.pdf", None),
            "lecture.pdf",
        )
        self.assertEqual(
            _format_source_label(None, 4),
            "Page 4",
        )
        self.assertEqual(
            _format_source_label(None, None),
            "Unknown source",
        )

        # 2. Test deduplication behavior
        # Simulate 4 retrieved chunks: two from page 1, two from page 2
        mock_sources = [
            {"text": "Chunk 1", "source": "notes.pdf", "page": 1},
            {"text": "Chunk 2", "source": "notes.pdf", "page": 1},
            {"text": "Chunk 3", "source": "notes.pdf", "page": 2},
            {"text": "Chunk 4", "source": "notes.pdf", "page": 2},
        ]

        seen = set()
        unique_labels = []
        for s in mock_sources:
            src = s.get("source")
            pg = s.get("page")
            key = (src, pg)
            if key not in seen:
                seen.add(key)
                unique_labels.append(_format_source_label(src, pg))

        self.assertEqual(
            unique_labels,
            ["notes.pdf (Page 1)", "notes.pdf (Page 2)"],
        )

        # 3. Test missing metadata handling gracefully
        missing_meta_sources = [
            {"text": "Legacy chunk without metadata", "source": None, "page": None},
        ]
        seen_missing = set()
        missing_labels = []
        for s in missing_meta_sources:
            src = s.get("source")
            pg = s.get("page")
            key = (src, pg)
            if key not in seen_missing:
                seen_missing.add(key)
                missing_labels.append(_format_source_label(src, pg))

        self.assertEqual(missing_labels, ["Unknown source"])

    # -------------------------------------------------------------------------
    # Test H — Document isolation
    # -------------------------------------------------------------------------
    def test_h_document_isolation(self):
        """
        Verify that a query for one document cannot surface source metadata
        or chunks belonging to another document.
        """
        pdf_a = make_pdf(["Document A: Quantum Computing Principles and Qubits."])
        pdf_b = make_pdf(["Document B: Renaissance Art and Italian Sculptures."])

        doc_id_a = "quantum.pdf:1111"
        doc_id_b = "art.pdf:2222"

        ingest.ingest_pdf(pdf_a, doc_id=doc_id_a, source_name="quantum.pdf")
        ingest.ingest_pdf(pdf_b, doc_id=doc_id_b, source_name="art.pdf")

        # Query Document A
        docs_a = retriever.retrieve_with_metadata("principles", doc_id=doc_id_a)
        for doc in docs_a:
            self.assertEqual(doc.metadata["source"], "quantum.pdf")
            self.assertEqual(doc.metadata["doc_id"], doc_id_a)
            self.assertNotIn("Renaissance", doc.page_content)

        # Query Document B
        docs_b = retriever.retrieve_with_metadata("principles", doc_id=doc_id_b)
        for doc in docs_b:
            self.assertEqual(doc.metadata["source"], "art.pdf")
            self.assertEqual(doc.metadata["doc_id"], doc_id_b)
            self.assertNotIn("Quantum", doc.page_content)

    # -------------------------------------------------------------------------
    # Test I — Index security
    # -------------------------------------------------------------------------
    def test_i_index_security(self):
        """
        Verify that metadata-bearing indexes are signed and pass the existing
        integrity-validation boundary. Tampered or unsigned indexes must remain
        rejected before deserialization.
        """
        pdf_bytes = make_pdf(["Security test page on HMAC-SHA256 signatures."])
        doc_id = "security.pdf:8888"
        source_name = "security.pdf"

        ingest.ingest_pdf(pdf_bytes, doc_id=doc_id, source_name=source_name)

        target_dir = config.get_vectorstore_dir(doc_id, root_dir=self.temp_dir)
        self.assertTrue(os.path.exists(os.path.join(target_dir, "index.sig")))
        self.assertTrue(os.path.exists(os.path.join(target_dir, "index.pkl")))
        self.assertTrue(os.path.exists(os.path.join(target_dir, "index.faiss")))

        # Verified load succeeds
        docs = retriever.retrieve_with_metadata("Security", doc_id=doc_id)
        self.assertGreater(len(docs), 0)

        # Clear retriever cache so reload attempts deserialization
        retriever._vectorstores.clear()
        retriever._vectorstore = None

        # Tamper with index.pkl
        pkl_path = os.path.join(target_dir, "index.pkl")
        with open(pkl_path, "ab") as f:
            f.write(b"\x00CORRUPT")

        # Loading tampered index must raise SecurityError or ValueError
        with self.assertRaises(Exception) as cm:
            retriever._load(doc_id=doc_id)
        self.assertIn("tampered", str(cm.exception).lower())

    # -------------------------------------------------------------------------
    # Test J — App chat source attribution end-to-end with AppTest
    # -------------------------------------------------------------------------
    def test_j_app_chat_source_attribution_e2e(self):
        """
        Verify end-to-end in Streamlit AppTest that an answer displays
        source attribution (filename and page), retains sources in session state,
        and continues rendering attribution across script reruns.
        """
        from streamlit.testing.v1 import AppTest

        app_path = str(SRC_DIR / "app.py")
        retrieved_test_docs = [
            Document(page_content="Demand paging brings pages into RAM on reference.", metadata={"source": "os_textbook.pdf", "page": 42}),
            Document(page_content="Page tables translate virtual addresses to physical frames.", metadata={"source": "os_textbook.pdf", "page": 42}),
            Document(page_content="Thrashing occurs when the working set exceeds physical memory.", metadata={"source": "os_textbook.pdf", "page": 45}),
        ]

        with patch("langchain_groq.ChatGroq.invoke") as mock_invoke, \
             patch("retriever.retrieve_with_metadata", return_value=retrieved_test_docs):
            mock_invoke.return_value = MagicMock(content="Paging loads frames into RAM when referenced.")

            at = AppTest.from_file(app_path, default_timeout=10)
            at.session_state["document_ready"] = True
            at.session_state["processed_file_id"] = "os_textbook.pdf:hash111"
            at.session_state["current_doc_name"] = "os_textbook.pdf"
            at.run()

            # Chat input
            at.chat_input[0].set_value("How does demand paging work?").run()

            # Verify assistant message has sources
            messages = at.session_state["messages"]
            self.assertEqual(len(messages), 2)
            assistant_msg = messages[1]
            self.assertEqual(assistant_msg["role"], "assistant")
            self.assertIn("sources", assistant_msg)
            self.assertEqual(len(assistant_msg["sources"]), 3)

            # Verify rendered caption contains deduplicated sources
            caption_texts = [c.value for c in at.caption]
            self.assertTrue(
                any("os_textbook.pdf (Page 42)" in c and "os_textbook.pdf (Page 45)" in c for c in caption_texts),
                f"Expected source caption not found in captions: {caption_texts}"
            )

            # Verify rerun preserves sources and caption
            at.run()
            caption_texts_rerun = [c.value for c in at.caption]
            self.assertTrue(
                any("os_textbook.pdf (Page 42)" in c and "os_textbook.pdf (Page 45)" in c for c in caption_texts_rerun),
                f"Source attribution lost after rerun: {caption_texts_rerun}"
            )


if __name__ == "__main__":
    unittest.main()
