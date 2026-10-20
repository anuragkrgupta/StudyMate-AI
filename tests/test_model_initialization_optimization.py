import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from pypdf import PdfWriter

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
import ingest
import retriever
from app import get_llm, generate_document_summary, build_chat_prompt
from security import verify_vectorstore_index


class FakeEmbeddings:
    """Lightweight in-memory embeddings for unit testing without downloading model weights."""

    def __init__(self, size=10, model_name=None):
        self.size = size
        self.model_name = model_name

    def embed_documents(self, texts):
        return [[0.1] * self.size for _ in texts]

    def embed_query(self, text):
        return [0.1] * self.size


def make_sample_pdf(pages_text: list[str]) -> bytes:
    writer = PdfWriter()
    for text in pages_text:
        writer.add_blank_page(width=300, height=300)
    import io
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


class TestModelInitializationOptimization(unittest.TestCase):
    """
    Focused regression tests verifying that LLM clients and embedding objects
    are safely cached and reused without redundant initializations.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fake_embeddings = FakeEmbeddings(size=10, model_name=config.EMBEDDING_MODEL)

        # Patch VECTORSTORE_DIR across config, ingest, retriever
        self.config_dir_patch = patch.object(config, "VECTORSTORE_DIR", self.temp_dir)
        self.ingest_dir_patch = patch.object(ingest, "VECTORSTORE_DIR", self.temp_dir)
        self.retriever_dir_patch = patch.object(retriever, "VECTORSTORE_DIR", self.temp_dir)

        self.config_dir_patch.start()
        self.ingest_dir_patch.start()
        self.retriever_dir_patch.start()

        # Reset module state
        retriever.clear_embeddings_cache()
        ingest.clear_embeddings_cache()
        retriever._vectorstore = None
        retriever._vectorstores.clear()

    def tearDown(self):
        self.retriever_dir_patch.stop()
        self.ingest_dir_patch.stop()
        self.config_dir_patch.stop()

        retriever.clear_embeddings_cache()
        ingest.clear_embeddings_cache()
        retriever._vectorstore = None
        retriever._vectorstores.clear()

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test A — LLM client reuse
    # -------------------------------------------------------------------------
    def test_a_llm_client_reuse(self):
        """Repeated access with unchanged configuration reuses the same client instance."""
        client1 = get_llm(model="llama-3.3-70b-versatile", api_key="gsk_test_key_1", temperature=0.0)
        client2 = get_llm(model="llama-3.3-70b-versatile", api_key="gsk_test_key_1", temperature=0.0)

        self.assertIs(
            client1,
            client2,
            "get_llm must return the identical cached ChatGroq instance when configuration is unchanged",
        )

    # -------------------------------------------------------------------------
    # Test B — LLM configuration changes
    # -------------------------------------------------------------------------
    def test_b_llm_configuration_changes(self):
        """Changing model, api_key, or temperature returns distinct client instances."""
        base_client = get_llm(model="llama-3.3-70b-versatile", api_key="gsk_test_key_1", temperature=0.0)

        # Different model
        diff_model_client = get_llm(model="mixtral-8x7b-32768", api_key="gsk_test_key_1", temperature=0.0)
        self.assertIsNot(
            base_client,
            diff_model_client,
            "Changing model name must produce a new ChatGroq client",
        )

        # Different API key
        diff_key_client = get_llm(model="llama-3.3-70b-versatile", api_key="gsk_test_key_2", temperature=0.0)
        self.assertIsNot(
            base_client,
            diff_key_client,
            "Changing API key must produce a new ChatGroq client",
        )

        # Different temperature
        diff_temp_client = get_llm(model="llama-3.3-70b-versatile", api_key="gsk_test_key_1", temperature=0.7)
        self.assertIsNot(
            base_client,
            diff_temp_client,
            "Changing temperature must produce a new ChatGroq client",
        )

    # -------------------------------------------------------------------------
    # Test C — Embedding wrapper reuse
    # -------------------------------------------------------------------------
    def test_c_embedding_wrapper_reuse(self):
        """Repeated access for the same embedding configuration returns a reusable instance."""
        with patch("retriever.HuggingFaceEmbeddings", side_effect=lambda model_name: FakeEmbeddings(model_name=model_name)):
            emb1 = retriever.get_embeddings("all-MiniLM-L6-v2")
            emb2 = retriever.get_embeddings("all-MiniLM-L6-v2")
            self.assertIs(emb1, emb2, "Repeated get_embeddings in retriever must return the identical cached instance")

            # ingest.get_embeddings delegates to retriever in shared process
            emb3 = ingest.get_embeddings("all-MiniLM-L6-v2")
            self.assertIs(emb1, emb3, "ingest.get_embeddings must share the cached instance with retriever")

    # -------------------------------------------------------------------------
    # Test D — Model configuration isolation
    # -------------------------------------------------------------------------
    def test_d_model_configuration_isolation(self):
        """Embedding wrappers constructed for different configured model names are distinct."""
        with patch("retriever.HuggingFaceEmbeddings", side_effect=lambda model_name: FakeEmbeddings(model_name=model_name)):
            emb_a = retriever.get_embeddings("model-alpha")
            emb_b = retriever.get_embeddings("model-beta")

            self.assertIsNot(emb_a, emb_b, "Different model names must produce distinct embedding wrapper instances")
            self.assertEqual(emb_a.model_name, "model-alpha")
            self.assertEqual(emb_b.model_name, "model-beta")

    # -------------------------------------------------------------------------
    # Test E — Multiple document stores
    # -------------------------------------------------------------------------
    def test_e_multiple_document_stores(self):
        """Reusing an embedding wrapper does not cause cross-document leakage or store substitution."""
        from langchain_community.vectorstores import FAISS

        emb = FakeEmbeddings(size=10, model_name=config.EMBEDDING_MODEL)

        doc_a_id = "doc_alpha.pdf:11112222"
        doc_b_id = "doc_beta.pdf:33334444"

        dir_a = config.get_vectorstore_dir(doc_a_id, root_dir=self.temp_dir)
        dir_b = config.get_vectorstore_dir(doc_b_id, root_dir=self.temp_dir)

        vs_a = FAISS.from_texts(["Content Alpha 123"], embedding=emb)
        vs_b = FAISS.from_texts(["Content Beta 456"], embedding=emb)

        os.makedirs(dir_a, exist_ok=True)
        os.makedirs(dir_b, exist_ok=True)
        vs_a.save_local(dir_a)
        vs_b.save_local(dir_b)

        from security import sign_vectorstore_index
        sign_vectorstore_index(dir_a)
        sign_vectorstore_index(dir_b)

        with patch("retriever.get_embeddings", return_value=emb):
            loaded_a = retriever._load(doc_a_id)
            loaded_b = retriever._load(doc_b_id)

            self.assertIsNot(loaded_a, loaded_b, "Vector stores for distinct documents must be distinct instances")
            chunks_a = retriever.get_all_chunks(doc_a_id)
            chunks_b = retriever.get_all_chunks(doc_b_id)

            self.assertEqual(chunks_a, ["Content Alpha 123"])
            self.assertEqual(chunks_b, ["Content Beta 456"])

    # -------------------------------------------------------------------------
    # Test F — Ingestion behavior and signing
    # -------------------------------------------------------------------------
    def test_f_ingestion_behavior(self):
        """The optimized ingestion workflow produces embeddings and saves signed FAISS index."""
        from unittest.mock import MagicMock

        doc_id = "sample_lecture.pdf:aabbccddeeff"
        target_dir = config.get_vectorstore_dir(doc_id, root_dir=self.temp_dir)

        fake_reader = MagicMock()
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "Important lecture notes on operating systems and memory management."
        fake_reader.pages = [mock_page]

        with patch("ingest._read_pdf", return_value=fake_reader), \
             patch("ingest.get_embeddings", return_value=self.fake_embeddings):
            chunk_count = ingest.ingest_pdf(b"fake pdf bytes", doc_id=doc_id, source_name="lecture.pdf")

        self.assertGreater(chunk_count, 0)
        self.assertTrue(os.path.exists(os.path.join(target_dir, "index.faiss")))
        self.assertTrue(os.path.exists(os.path.join(target_dir, "index.pkl")))
        self.assertTrue(os.path.exists(os.path.join(target_dir, "index.sig")))

        # Verify HMAC signature validates successfully (verify_vectorstore_index raises on failure)
        verify_vectorstore_index(target_dir)

    # -------------------------------------------------------------------------
    # Test G — Initialization frequency instrumentation
    # -------------------------------------------------------------------------
    def test_g_initialization_frequency(self):
        """Confirm reduced constructor invocation counts through instrumentation."""
        # 1. ChatGroq constructor frequency test
        with patch("langchain_groq.ChatGroq.__init__", return_value=None) as mock_groq_init:
            for _ in range(5):
                get_llm(model="llama-3.3-70b-versatile", api_key="test_key_freq", temperature=0.0)
            self.assertEqual(
                mock_groq_init.call_count,
                1,
                "ChatGroq constructor must be invoked exactly once across repeated requests with identical config",
            )

        # 2. Embedding constructor frequency test across retriever and ingest
        retriever.clear_embeddings_cache()
        ingest.clear_embeddings_cache()

        with patch("langchain_huggingface.HuggingFaceEmbeddings.__init__", return_value=None) as mock_emb_init:
            for _ in range(3):
                retriever.get_embeddings("test-embedding-model")
            for _ in range(3):
                ingest.get_embeddings("test-embedding-model")

            self.assertEqual(
                mock_emb_init.call_count,
                1,
                "HuggingFaceEmbeddings constructor must be invoked exactly once across repeated retriever & ingest calls",
            )

    # -------------------------------------------------------------------------
    # Test H — Existing application behavior
    # -------------------------------------------------------------------------
    def test_h_existing_application_behavior(self):
        """Chat generation, summarization, and prompt building operate normally with cached client."""
        client = get_llm(model="llama-3.3-70b-versatile", api_key="gsk_test_key_app", temperature=0.0)

        with patch("langchain_groq.ChatGroq.invoke", return_value=MagicMock(content="Summary of operating systems")) as mock_inv:
            summary = generate_document_summary(client, "Page 1 notes")
            self.assertEqual(summary, "Summary of operating systems")
            self.assertEqual(mock_inv.call_count, 1)

        # Test build_chat_prompt formatting
        prompt = build_chat_prompt(
            question="What is paging?",
            context="Paging divides memory into frames.",
            history_messages=[{"role": "user", "content": "Hello"}],
        )
        self.assertIn("What is paging?", prompt)
        self.assertIn("Paging divides memory into frames.", prompt)
        self.assertIn("user: Hello", prompt)

    # -------------------------------------------------------------------------
    # Test I — Non-Streamlit standalone compatibility
    # -------------------------------------------------------------------------
    def test_i_non_streamlit_compatibility(self):
        """Embedding helper in ingest and retriever functions properly outside of Streamlit runtime."""
        retriever.clear_embeddings_cache()
        ingest.clear_embeddings_cache()

        with patch("retriever.HuggingFaceEmbeddings", side_effect=lambda model_name: FakeEmbeddings(model_name=model_name)):
            emb = ingest.get_embeddings(config.EMBEDDING_MODEL)
            self.assertIsNotNone(emb)
            self.assertEqual(emb.embed_query("test text"), [0.1] * 10)


if __name__ == "__main__":
    unittest.main()
