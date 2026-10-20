import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config
import retriever
import chatbot
from security import IndexSecurityError, sign_vectorstore_index


class FakeEmbeddings:
    """Lightweight in-memory embeddings for unit testing without downloading model weights."""

    def __init__(self, size=10, model_name=None):
        self.size = size
        self.model_name = model_name

    def embed_documents(self, texts):
        return [[0.1] * self.size for _ in texts]

    def embed_query(self, text):
        return [0.1] * self.size

    def __call__(self, text):
        return self.embed_query(text)


class TestLegacyCodeAudit(unittest.TestCase):
    """
    Focused regression tests verifying that the legacy chatbot entry point is
    consistent with project configuration, uses current retrieval and security
    conventions, and does not break existing interfaces.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fake_embeddings = FakeEmbeddings(size=10, model_name=config.EMBEDDING_MODEL)

        self.config_dir_patch = patch.object(config, "VECTORSTORE_DIR", self.temp_dir)
        self.retriever_dir_patch = patch.object(retriever, "VECTORSTORE_DIR", self.temp_dir)

        self.config_dir_patch.start()
        self.retriever_dir_patch.start()

        retriever.clear_embeddings_cache()
        retriever._vectorstore = None
        retriever._vectorstores.clear()

    def tearDown(self):
        self.retriever_dir_patch.stop()
        self.config_dir_patch.stop()

        retriever.clear_embeddings_cache()
        retriever._vectorstore = None
        retriever._vectorstores.clear()

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    # -------------------------------------------------------------------------
    # Test A — Legacy chatbot import
    # -------------------------------------------------------------------------
    def test_a_legacy_chatbot_import(self):
        """chatbot.py imports cleanly without missing dependencies or missing CHAT_MODEL."""
        self.assertTrue(hasattr(chatbot, "CHAT_MODEL"), "chatbot module must expose CHAT_MODEL")
        self.assertIsInstance(chatbot.CHAT_MODEL, str)
        self.assertGreater(len(chatbot.CHAT_MODEL), 0)

        # Verify function exports
        self.assertTrue(callable(chatbot.get_chatbot_llm))
        self.assertTrue(callable(chatbot.ask_question))
        self.assertTrue(callable(chatbot.run_cli))

    # -------------------------------------------------------------------------
    # Test B — Configuration resolution
    # -------------------------------------------------------------------------
    def test_b_configuration_resolution(self):
        """Chatbot reads model and API key from project conventions and fails closed if missing."""
        # 1. Successful initialization with explicit key and new default model
        client = chatbot.get_chatbot_llm(model="openai/gpt-oss-120b", api_key="gsk_valid_key")
        self.assertEqual(client.model_name, "openai/gpt-oss-120b")

        # 2. Fallback model defaults to openai/gpt-oss-120b when GROQ_MODEL is unset
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_env_key"}, clear=False):
            with patch.dict(os.environ, {"GROQ_MODEL": ""}):
                client_default = chatbot.get_chatbot_llm()
                self.assertEqual(client_default.model_name, "openai/gpt-oss-120b")

        # 3. Initialization reading from environment variable override
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_env_key", "GROQ_MODEL": "custom-groq-model"}):
            client_env = chatbot.get_chatbot_llm()
            self.assertEqual(client_env.model_name, "custom-groq-model")

        # 4. Fails closed when key is missing or empty
        with patch.dict(os.environ, {"GROQ_API_KEY": ""}, clear=False):
            with patch("security.load_env_file"):
                with self.assertRaises(ValueError) as ctx:
                    chatbot.get_chatbot_llm(api_key="")
                self.assertIn("GROQ_API_KEY is not configured", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Test C — Retrieval compatibility
    # -------------------------------------------------------------------------
    def test_c_retrieval_compatibility(self):
        """ask_question uses retriever.retrieve() and formats context properly."""
        sample_chunks = [
            "Paging is a memory management scheme that eliminates the need for contiguous allocation.",
            "Frames are fixed-size blocks of physical memory.",
        ]

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Paging divides memory into fixed-size frames.")

        with patch("chatbot.retrieve", return_value=sample_chunks) as mock_retrieve:
            answer = chatbot.ask_question("What is paging?", doc_id="os_notes.pdf:1234", llm=mock_llm)

            mock_retrieve.assert_called_once_with("What is paging?", doc_id="os_notes.pdf:1234")
            self.assertEqual(answer, "Paging divides memory into fixed-size frames.")

            # Check prompt content sent to LLM
            sent_prompt = mock_llm.invoke.call_args[0][0]
            self.assertIn("What is paging?", sent_prompt)
            self.assertIn("Paging is a memory management scheme", sent_prompt)
            self.assertIn("Frames are fixed-size blocks", sent_prompt)

    # -------------------------------------------------------------------------
    # Test D — FAISS security enforcement
    # -------------------------------------------------------------------------
    def test_d_faiss_security_enforcement(self):
        """Chatbot queries route through FAISS index verification and reject unsigned/tampered stores."""
        from langchain_community.vectorstores import FAISS

        doc_id = "test_doc.pdf:99998888"
        target_dir = config.get_vectorstore_dir(doc_id, root_dir=self.temp_dir)
        os.makedirs(target_dir, exist_ok=True)

        vs = FAISS.from_texts(["Secure chunk data"], embedding=self.fake_embeddings)
        vs.save_local(target_dir)
        # Note: Deliberately do NOT sign the index so it remains unsigned

        mock_llm = MagicMock()
        with patch("retriever.get_embeddings", return_value=self.fake_embeddings):
            with self.assertRaises(IndexSecurityError) as ctx:
                chatbot.ask_question("query text", doc_id=doc_id, llm=mock_llm)
            self.assertIn("Missing integrity signature", str(ctx.exception))

        # Now sign the index and verify that ask_question succeeds
        sign_vectorstore_index(target_dir)
        with patch("retriever.get_embeddings", return_value=self.fake_embeddings):
            mock_llm.invoke.return_value = MagicMock(content="Answer from secure index")
            res = chatbot.ask_question("query text", doc_id=doc_id, llm=mock_llm)
            self.assertEqual(res, "Answer from secure index")

    # -------------------------------------------------------------------------
    # Test E — Redundant code and backward compatibility
    # -------------------------------------------------------------------------
    def test_e_redundant_code_and_backward_compatibility(self):
        """Verify exports and backward-compatible paths remain functional."""
        # 1. config constants and fallback alignment
        self.assertTrue(hasattr(config, "CHAT_MODEL"))
        self.assertTrue(hasattr(config, "DEFAULT_GROQ_MODEL"))
        self.assertEqual(config.DEFAULT_GROQ_MODEL, "openai/gpt-oss-120b")
        self.assertEqual(config.CHAT_MODEL, "openai/gpt-oss-120b")
        self.assertTrue(hasattr(config, "CHUNKS_PATH"))
        self.assertTrue(hasattr(config, "FAISS_INDEX_PATH"))

        # Dynamic model getter
        self.assertTrue(hasattr(config, "get_groq_model"))
        with patch.dict(os.environ, {"GROQ_MODEL": "test-override-model"}):
            self.assertEqual(config.get_groq_model(), "test-override-model")

        # Alignment with app.py fallback
        from app import DEFAULT_GROQ_MODEL as APP_DEFAULT_MODEL
        self.assertEqual(APP_DEFAULT_MODEL, config.DEFAULT_GROQ_MODEL)

        # 2. retriever backward-compatible _get_embeddings
        self.assertTrue(hasattr(retriever, "_get_embeddings"))
        self.assertTrue(callable(retriever._get_embeddings))

        # 3. empty query handling in ask_question
        self.assertEqual(chatbot.ask_question("   "), "")


if __name__ == "__main__":
    unittest.main()
