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
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import FakeEmbeddings

import config
import ingest
import retriever
from config import get_vectorstore_dir, sanitize_doc_id
from ingest import ingest_pdf
from retriever import get_all_chunks, retrieve, reload, vectorstore_exists


def make_pdf(text_lines):
    """Helper to create a single-page in-memory PDF with the provided text lines."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    y = 100
    for line in text_lines:
        page.insert_text((50, y), line)
        y += 30
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestVectorstoreIsolation(unittest.TestCase):
    """
    Regression test suite for vector store isolation across documents and sessions.
    Covers Tests A through G as specified in Task: Prevent Cross-Document Vector Store Interference.
    """

    def setUp(self):
        os.environ.setdefault(
            "STUDYMATE_INDEX_SIGNING_KEY",
            "test-signing-key-for-automated-regression-tests-only-32chars",
        )
        self.temp_dir = tempfile.mkdtemp()
        self.fake_embeddings = FakeEmbeddings(size=10)

        # Patch VECTORSTORE_DIR across config, ingest, and retriever
        self.config_patcher = patch.object(config, "VECTORSTORE_DIR", self.temp_dir)
        self.ingest_patcher = patch.object(ingest, "VECTORSTORE_DIR", self.temp_dir)
        self.retriever_patcher = patch.object(retriever, "VECTORSTORE_DIR", self.temp_dir)

        self.config_patcher.start()
        self.ingest_patcher.start()
        self.retriever_patcher.start()

        # Reset retriever module state
        retriever._vectorstore = None
        retriever._vectorstores.clear()
        retriever._embeddings = self.fake_embeddings

    def tearDown(self):
        self.config_patcher.stop()
        self.ingest_patcher.stop()
        self.retriever_patcher.stop()

        retriever._vectorstore = None
        retriever._vectorstores.clear()
        retriever._embeddings = None

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test A — Document-specific persistence
    # -------------------------------------------------------------------------
    def test_a_document_specific_persistence(self):
        """
        Ingest two different PDFs with distinct doc_ids and verify that their
        vector-store directories are distinct on the filesystem.
        """
        pdf_a = make_pdf(["Operating Systems: Process Scheduling and Virtual Memory"])
        pdf_b = make_pdf(["Microbiology: Bacterial Cell Wall Synthesis and Penicillin"])

        doc_a_id = "os_lecture.pdf:1111222233334444"
        doc_b_id = "bio_notes.pdf:5555666677778888"

        dir_a = get_vectorstore_dir(doc_a_id, root_dir=self.temp_dir)
        dir_b = get_vectorstore_dir(doc_b_id, root_dir=self.temp_dir)

        # Vector store paths must be distinct
        self.assertNotEqual(dir_a, dir_b)

        with patch("ingest.HuggingFaceEmbeddings", return_value=self.fake_embeddings):
            count_a = ingest_pdf(pdf_a, doc_id=doc_a_id)
            count_b = ingest_pdf(pdf_b, doc_id=doc_b_id)

        self.assertGreater(count_a, 0)
        self.assertGreater(count_b, 0)

        # Both separate directories must exist and contain their own index.faiss
        self.assertTrue(os.path.exists(os.path.join(dir_a, "index.faiss")))
        self.assertTrue(os.path.exists(os.path.join(dir_a, "index.pkl")))
        self.assertTrue(os.path.exists(os.path.join(dir_b, "index.faiss")))
        self.assertTrue(os.path.exists(os.path.join(dir_b, "index.pkl")))

    # -------------------------------------------------------------------------
    # Test B — Content isolation
    # -------------------------------------------------------------------------
    def test_b_content_isolation(self):
        """
        Verify that document A retrieves only document A's chunks and
        document B retrieves only document B's chunks.
        """
        pdf_a = make_pdf([
            "Topic A: Relational Database Normalization and Third Normal Form BCNF."
        ])
        pdf_b = make_pdf([
            "Topic B: Astrophysics and Black Hole Event Horizon Singularity."
        ])

        doc_a_id = "db_notes.pdf:aaaa1111"
        doc_b_id = "astro.pdf:bbbb2222"

        with patch("ingest.HuggingFaceEmbeddings", return_value=self.fake_embeddings):
            ingest_pdf(pdf_a, doc_id=doc_a_id)
            ingest_pdf(pdf_b, doc_id=doc_b_id)

        # Retrieve all chunks for each document
        chunks_a = get_all_chunks(doc_id=doc_a_id)
        chunks_b = get_all_chunks(doc_id=doc_b_id)

        # Verify Doc A contains Relational Database and no Astrophysics
        content_a = " ".join(chunks_a)
        self.assertIn("Relational Database", content_a)
        self.assertNotIn("Astrophysics", content_a)

        # Verify Doc B contains Astrophysics and no Relational Database
        content_b = " ".join(chunks_b)
        self.assertIn("Astrophysics", content_b)
        self.assertNotIn("Relational Database", content_b)

    # -------------------------------------------------------------------------
    # Test C — Interleaved session behavior
    # -------------------------------------------------------------------------
    def test_c_interleaved_session_behavior(self):
        """
        Simulate the interleaved multi-session sequence:
        1. Session A ingests document A.
        2. Session B ingests document B.
        3. Session A retrieves chunks again.
        4. Session B retrieves chunks again.
        Verify that both sessions resolve strictly to their own document content.
        """
        pdf_a = make_pdf(["Session A Chemistry Notes: Thermodynamics and Gibbs Free Energy"])
        pdf_b = make_pdf(["Session B Literature Notes: Shakespearean Tragedies and Hamlet"])

        session_a_doc = "chem.pdf:sessionA123"
        session_b_doc = "lit.pdf:sessionB456"

        with patch("ingest.HuggingFaceEmbeddings", return_value=self.fake_embeddings):
            # Step 1: Session A ingests and reloads Document A
            ingest_pdf(pdf_a, doc_id=session_a_doc)
            reload(doc_id=session_a_doc)

            # Step 2: Session B ingests and reloads Document B
            ingest_pdf(pdf_b, doc_id=session_b_doc)
            reload(doc_id=session_b_doc)

        # Step 3: Session A retrieves chunks again
        res_a = retrieve("Thermodynamics", doc_id=session_a_doc)
        all_a = get_all_chunks(doc_id=session_a_doc)

        self.assertTrue(any("Thermodynamics" in c for c in res_a))
        self.assertTrue(all("Shakespeare" not in c for c in res_a))
        self.assertTrue(all("Shakespeare" not in c for c in all_a))

        # Step 4: Session B retrieves chunks again
        res_b = retrieve("Shakespeare", doc_id=session_b_doc)
        all_b = get_all_chunks(doc_id=session_b_doc)

        self.assertTrue(any("Shakespeare" in c for c in res_b))
        self.assertTrue(all("Thermodynamics" not in c for c in res_b))
        self.assertTrue(all("Thermodynamics" not in c for c in all_b))

    # -------------------------------------------------------------------------
    # Test D — Same-document rerun
    # -------------------------------------------------------------------------
    def test_d_same_document_rerun(self):
        """
        Verify that a rerun for the same document reuses the cached in-memory index
        without unnecessary disk reloads or re-ingestions.
        """
        pdf = make_pdf(["Linear Algebra: Eigenvalues, Eigenvectors, and Matrix Diagonalization."])
        doc_id = "math_linear.pdf:99990000"

        with patch("ingest.HuggingFaceEmbeddings", return_value=self.fake_embeddings):
            ingest_pdf(pdf, doc_id=doc_id)

        # First load
        vs_first = retriever._load(doc_id=doc_id)

        # Second load (simulating subsequent Streamlit rerun in the same session)
        with patch.object(FAISS, "load_local") as mock_load_local:
            vs_second = retriever._load(doc_id=doc_id)
            # Must not hit disk again because it is cached in _vectorstores
            mock_load_local.assert_not_called()

        self.assertIs(vs_first, vs_second)

    # -------------------------------------------------------------------------
    # Test E — Document switch
    # -------------------------------------------------------------------------
    def test_e_document_switch(self):
        """
        Verify that switching a session from document A to document B causes retrieval
        to switch to B without exposing document A's chunks.
        """
        pdf_a = make_pdf(["Document Alpha: First Document Content"])
        pdf_b = make_pdf(["Document Beta: Second Document Content"])

        doc_a = "alpha.pdf:1111"
        doc_b = "beta.pdf:2222"

        with patch("ingest.HuggingFaceEmbeddings", return_value=self.fake_embeddings):
            ingest_pdf(pdf_a, doc_id=doc_a)
            ingest_pdf(pdf_b, doc_id=doc_b)

        # Initial active document is A
        active_doc = doc_a
        chunks = get_all_chunks(doc_id=active_doc)
        self.assertIn("Document Alpha", " ".join(chunks))
        self.assertNotIn("Document Beta", " ".join(chunks))

        # Switch active document to B
        active_doc = doc_b
        chunks_switched = get_all_chunks(doc_id=active_doc)
        self.assertIn("Document Beta", " ".join(chunks_switched))
        self.assertNotIn("Document Alpha", " ".join(chunks_switched))

    # -------------------------------------------------------------------------
    # Test F — Failed ingestion
    # -------------------------------------------------------------------------
    def test_f_failed_ingestion_does_not_expose_incomplete_index(self):
        """
        Verify that if ingestion fails (e.g. invalid PDF), no index is created
        or marked as available for that document identifier.
        """
        invalid_pdf_bytes = b"not-a-valid-pdf-content-header"
        failed_doc_id = "corrupted.pdf:fail1234"

        # Verify vector store does not exist initially
        self.assertFalse(vectorstore_exists(doc_id=failed_doc_id))

        with self.assertRaises(Exception):
            ingest_pdf(invalid_pdf_bytes, doc_id=failed_doc_id)

        # Failed document should have no persisted index
        self.assertFalse(vectorstore_exists(doc_id=failed_doc_id))

        # Attempting to retrieve from the failed document must raise FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            get_all_chunks(doc_id=failed_doc_id)

    # -------------------------------------------------------------------------
    # Test G — All-chunks retrieval regression under isolation
    # -------------------------------------------------------------------------
    def test_g_all_chunks_retrieval_regression(self):
        """
        Verify that document-isolated get_all_chunks() retrieves all chunks
        without truncation on a document with more than 100 chunks (120 chunks).
        """
        doc_id = "large_doc.pdf:large120"
        target_dir = get_vectorstore_dir(doc_id, root_dir=self.temp_dir)

        chunks_120 = [f"Isolated section {i}: Advanced Neural Networks" for i in range(120)]
        vs = FAISS.from_texts(chunks_120, self.fake_embeddings)
        os.makedirs(target_dir, exist_ok=True)
        vs.save_local(target_dir)
        from security import sign_vectorstore_index
        sign_vectorstore_index(target_dir)

        retrieved = get_all_chunks(doc_id=doc_id)

        self.assertEqual(len(retrieved), 120)
        self.assertEqual(retrieved, chunks_120)


if __name__ == "__main__":
    unittest.main()
