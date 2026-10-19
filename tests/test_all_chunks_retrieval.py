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

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import FakeEmbeddings

import retriever
from retriever import get_all_chunks, vectorstore_exists


class TestAllChunksRetrieval(unittest.TestCase):
    """
    Regression test suite for complete and reliable chunk retrieval in StudyMate AI.
    Tests A through G cover all required behavior for get_all_chunks().
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fake_embeddings = FakeEmbeddings(size=10)

        # Patch retriever's VECTORSTORE_DIR to point to isolated temp directory
        self.dir_patcher = patch.object(
            retriever, "VECTORSTORE_DIR", self.temp_dir
        )
        self.dir_patcher.start()

        # Reset global singleton state in retriever
        retriever._vectorstore = None
        retriever._embeddings = self.fake_embeddings

    def tearDown(self):
        self.dir_patcher.stop()
        retriever._vectorstore = None
        retriever._embeddings = None
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_and_save_vectorstore(self, chunks):
        """Helper to create and persist a FAISS index with the given text chunks."""
        vs = FAISS.from_texts(chunks, self.fake_embeddings)
        vs.save_local(self.temp_dir)
        return vs

    # -------------------------------------------------------------------------
    # Test A — Retrieve fewer than 100 chunks
    # -------------------------------------------------------------------------
    def test_a_retrieve_fewer_than_100_chunks(self):
        """
        Create a vector store containing a known set of fewer than 100 chunks (25 chunks).
        Verify that all chunks are returned exactly once with no duplicates.
        """
        known_chunks = [
            f"Lecture topic {i}: Fundamental Concepts in NLP"
            for i in range(25)
        ]
        self._create_and_save_vectorstore(known_chunks)

        retrieved = get_all_chunks()

        self.assertIsInstance(retrieved, list)
        self.assertEqual(len(retrieved), 25)
        self.assertEqual(len(set(retrieved)), 25)
        self.assertEqual(retrieved, known_chunks)

    # -------------------------------------------------------------------------
    # Test B — Retrieve more than 100 chunks
    # -------------------------------------------------------------------------
    def test_b_retrieve_more_than_100_chunks(self):
        """
        Create a vector store with at least 150 distinct chunks.
        Verify that all 150 chunks are returned and retrieval is not capped at 100.
        """
        known_chunks = [
            f"Chunk {i}: In-depth analysis of transformer attention mechanisms {i * 7}."
            for i in range(150)
        ]
        self._create_and_save_vectorstore(known_chunks)

        retrieved = get_all_chunks()

        self.assertIsInstance(retrieved, list)
        self.assertEqual(len(retrieved), 150)
        self.assertGreater(len(retrieved), 100)
        self.assertEqual(retrieved, known_chunks)

    # -------------------------------------------------------------------------
    # Test C — Preserve content
    # -------------------------------------------------------------------------
    def test_c_preserve_chunk_content(self):
        """
        Verify that returned chunk contents match the known input content verbatim,
        including whitespace, punctuation, newlines, and unicode characters.
        """
        known_chunks = [
            "Introduction: Welcome to StudyMate AI!\nSection 1: Setup & Usage.",
            "Formulas: e^(i*pi) + 1 = 0; alpha_1 <= beta_2 && gamma > 0.05.",
            "Special characters: « quotes », — em-dash, and foreign terms: café, résumé.",
            "Indented code snippet:\n    def calculate(x):\n        return x ** 2",
        ]
        self._create_and_save_vectorstore(known_chunks)

        retrieved = get_all_chunks()

        self.assertEqual(len(retrieved), len(known_chunks))
        for original, result in zip(known_chunks, retrieved):
            self.assertEqual(result, original)

    # -------------------------------------------------------------------------
    # Test D — Preserve order
    # -------------------------------------------------------------------------
    def test_d_preserve_chunk_order(self):
        """
        Verify that returned chunks match the original insertion order
        for both small and large chunk sets.
        """
        chunks_120 = [f"Step {idx}: Sequential pipeline execution" for idx in range(120)]
        self._create_and_save_vectorstore(chunks_120)

        retrieved = get_all_chunks()

        for idx, expected in enumerate(chunks_120):
            self.assertEqual(retrieved[idx], expected)

    # -------------------------------------------------------------------------
    # Test E — Empty or unavailable vector store
    # -------------------------------------------------------------------------
    def test_e_unavailable_vector_store_raises_file_not_found(self):
        """
        When vector store has not been created on disk, get_all_chunks()
        raises FileNotFoundError according to existing application conventions.
        """
        # Clean temp directory so no vectorstore exists
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        retriever._vectorstore = None

        self.assertFalse(vectorstore_exists())
        with self.assertRaises(FileNotFoundError) as ctx:
            get_all_chunks()

        self.assertIn("No vector store found", str(ctx.exception))

    def test_e_empty_vector_store_returns_empty_list(self):
        """
        When vector store exists but contains 0 chunks, get_all_chunks()
        returns an empty list without error.
        """
        empty_index = faiss.IndexFlatL2(10)
        empty_vs = FAISS(
            self.fake_embeddings,
            empty_index,
            InMemoryDocstore({}),
            {},
        )
        empty_vs.save_local(self.temp_dir)

        retrieved = get_all_chunks()

        self.assertEqual(retrieved, [])

    def test_e_fallback_to_docstore_dict(self):
        """
        When index_to_docstore_id is missing or empty, get_all_chunks()
        safely falls back to extracting Document page_content from docstore._dict.
        """
        doc1 = Document(page_content="Fallback chunk A", metadata={})
        doc2 = Document(page_content="Fallback chunk B", metadata={})
        mock_vs = MagicMock()
        mock_vs.index_to_docstore_id = {}
        mock_vs.docstore._dict = {"id1": doc1, "id2": doc2}
        retriever._vectorstore = mock_vs

        retrieved = get_all_chunks()

        self.assertEqual(retrieved, ["Fallback chunk A", "Fallback chunk B"])

    # -------------------------------------------------------------------------
    # Test F — Existing caller compatibility
    # -------------------------------------------------------------------------
    def test_f_existing_caller_compatibility(self):
        """
        Verify that existing callers expecting List[str] can process
        the returned collection with string joins, slicing, and length checks.
        """
        chunks = [f"Paragraph {i}" for i in range(10)]
        self._create_and_save_vectorstore(chunks)

        all_chunks = get_all_chunks()

        # Exactly simulates app.py line 388-389:
        # all_chunks = get_all_chunks()
        # context = "\n\n".join(all_chunks)[:12000]
        self.assertIsInstance(all_chunks, list)
        self.assertTrue(all(isinstance(c, str) for c in all_chunks))

        context = "\n\n".join(all_chunks)[:12000]
        self.assertIsInstance(context, str)
        self.assertTrue(context.startswith("Paragraph 0"))
        self.assertTrue(context.endswith("Paragraph 9"))

    # -------------------------------------------------------------------------
    # Test G — Integration with summarization
    # -------------------------------------------------------------------------
    def test_g_integration_with_summarization(self):
        """
        Verify that summarization receives all chunks from a 120-chunk document
        without being capped at 100 chunks.
        """
        from app import generate_document_summary

        chunks_120 = [f"Topic {i}: Important scientific finding number {i}." for i in range(120)]
        self._create_and_save_vectorstore(chunks_120)

        all_chunks = get_all_chunks()
        self.assertEqual(len(all_chunks), 120)

        context = "\n\n".join(all_chunks)[:12000]

        # Mock LLM client so no real API call is made
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Comprehensive summary of the 120-chunk document."
        mock_llm.invoke.return_value = mock_response

        summary = generate_document_summary(mock_llm, context)

        self.assertEqual(summary, "Comprehensive summary of the 120-chunk document.")
        mock_llm.invoke.assert_called_once()
        invoked_prompt = mock_llm.invoke.call_args[0][0]
        self.assertIn("Topic 0:", invoked_prompt)


if __name__ == "__main__":
    unittest.main()
