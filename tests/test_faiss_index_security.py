import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import FakeEmbeddings

import config
import retriever
from config import get_vectorstore_dir
from retriever import _load, reload, get_all_chunks
from security import (
    IndexSecurityError,
    SIGNATURE_FILENAME,
    compute_file_hmac,
    get_signing_key,
    load_env_file,
    sign_vectorstore_index,
    verify_vectorstore_index,
)


class TestFaissIndexSecurity(unittest.TestCase):
    """
    Security regression test suite for FAISS index loading boundary in StudyMate AI.
    Verifies HMAC-SHA256 signature verification over index.pkl and index.faiss
    before FAISS.load_local(..., allow_dangerous_deserialization=True) can be executed.
    
    Tests A through G satisfy all required security acceptance criteria.
    """

    TEST_KEY = "test-secret-signing-key-for-faiss-index-integrity-32bytes!"
    ALT_KEY = "different-secret-signing-key-attacker-or-rotated-key-xyz!"

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.fake_embeddings = FakeEmbeddings(size=10)

        # Ensure environment has the standard test signing key
        os.environ["STUDYMATE_INDEX_SIGNING_KEY"] = self.TEST_KEY

        # Patch retriever's VECTORSTORE_DIR to point to isolated temp directory
        self.retriever_dir_patcher = patch.object(
            retriever, "VECTORSTORE_DIR", self.temp_dir
        )
        self.config_dir_patcher = patch.object(
            config, "VECTORSTORE_DIR", self.temp_dir
        )
        self.retriever_dir_patcher.start()
        self.config_dir_patcher.start()

        # Reset global singleton state in retriever
        retriever._vectorstore = None
        retriever._vectorstores.clear()
        retriever._embeddings = self.fake_embeddings

    def tearDown(self):
        self.retriever_dir_patcher.stop()
        self.config_dir_patcher.stop()

        retriever._vectorstore = None
        retriever._vectorstores.clear()
        retriever._embeddings = None

        if "STUDYMATE_INDEX_SIGNING_KEY" in os.environ:
            os.environ["STUDYMATE_INDEX_SIGNING_KEY"] = self.TEST_KEY

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_raw_index(self, directory: str, texts: list[str]) -> FAISS:
        """Create and save a FAISS index to the specified directory without signing."""
        os.makedirs(directory, exist_ok=True)
        vs = FAISS.from_texts(texts, self.fake_embeddings)
        vs.save_local(directory)
        return vs

    def _create_signed_index(
        self,
        directory: str,
        texts: list[str],
        key: str | None = None,
    ) -> FAISS:
        """Create, save, and cryptographically sign a FAISS index."""
        vs = self._create_raw_index(directory, texts)
        sign_vectorstore_index(directory, key=key)
        return vs

    # -------------------------------------------------------------------------
    # Test A — Valid signed index
    # -------------------------------------------------------------------------
    def test_a_valid_signed_index_loads_successfully(self):
        """
        Save an index, produce a valid HMAC signature, and verify that
        loading proceeds successfully and returns expected data.
        """
        chunks = ["Section 1: Operating Systems", "Section 2: Memory Management"]
        self._create_signed_index(self.temp_dir, chunks, key=self.TEST_KEY)

        # Signature sidecar must exist
        sig_file = os.path.join(self.temp_dir, SIGNATURE_FILENAME)
        self.assertTrue(os.path.exists(sig_file))

        # Direct verification must pass without exception
        verify_vectorstore_index(self.temp_dir, key=self.TEST_KEY)

        # Loading via retriever must succeed and retrieve all chunks
        retrieved = get_all_chunks()
        self.assertEqual(retrieved, chunks)

    def test_a2_restart_simulation_with_same_key_loads_successfully(self):
        """
        Verify that an index persisted and signed in one application session
        remains loadable after clearing in-memory caches and re-initializing.
        """
        chunks = ["Chapter 1: Compilers and Parsers", "Chapter 2: Code Generation"]
        self._create_signed_index(self.temp_dir, chunks, key=self.TEST_KEY)

        # Simulate complete application restart: clear all in-memory caches
        retriever._vectorstore = None
        retriever._vectorstores.clear()

        # Reload with the same signing key configured
        reloaded_chunks = get_all_chunks()
        self.assertEqual(reloaded_chunks, chunks)

    # -------------------------------------------------------------------------
    # Test B — Tampered pickle
    # -------------------------------------------------------------------------
    def test_b_tampered_pickle_rejected_before_deserialization(self):
        """
        Modify the bytes of index.pkl after signing.
        Verify that loading is rejected with IndexSecurityError before
        FAISS.load_local() is called.
        """
        chunks = ["Data Mining: Apriori Algorithm"]
        self._create_signed_index(self.temp_dir, chunks, key=self.TEST_KEY)

        # Tamper with index.pkl by appending malicious/corrupt bytes
        pkl_path = Path(self.temp_dir) / "index.pkl"
        original_bytes = pkl_path.read_bytes()
        pkl_path.write_bytes(original_bytes + b"\x00\xffMALICIOUS_PAYLOAD")

        # Spy on FAISS.load_local to guarantee it is NEVER called
        with patch.object(FAISS, "load_local") as mock_load_local:
            with self.assertRaises(IndexSecurityError) as ctx:
                _load()

            mock_load_local.assert_not_called()

        self.assertIn("index.pkl", str(ctx.exception))
        self.assertIn("modified", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # Test C — Missing signature
    # -------------------------------------------------------------------------
    def test_c_missing_signature_rejected_before_deserialization(self):
        """
        Create a raw FAISS index without an index.sig signature sidecar.
        Verify that loading fails with an actionable IndexSecurityError and
        does not invoke FAISS.load_local().
        """
        chunks = ["Legacy lecture notes created before security fix"]
        self._create_raw_index(self.temp_dir, chunks)

        sig_file = os.path.join(self.temp_dir, SIGNATURE_FILENAME)
        self.assertFalse(os.path.exists(sig_file))

        with patch.object(FAISS, "load_local") as mock_load_local:
            with self.assertRaises(IndexSecurityError) as ctx:
                _load()

            mock_load_local.assert_not_called()

        err_msg = str(ctx.exception)
        self.assertIn("Missing integrity signature", err_msg)
        self.assertIn("STUDYMATE_INDEX_SIGNING_KEY", err_msg)
        self.assertIn("rebuild the index", err_msg)

    # -------------------------------------------------------------------------
    # Test D — Incorrect signing key
    # -------------------------------------------------------------------------
    def test_d_mismatched_signing_key_rejected(self):
        """
        Sign index with Key A, then attempt to load with Key B configured in environment.
        Verify that index loading is rejected and FAISS.load_local() is not called.
        """
        chunks = ["Secret Research Data: Confidential findings"]
        self._create_signed_index(self.temp_dir, chunks, key=self.TEST_KEY)

        # Switch environment to a different key
        os.environ["STUDYMATE_INDEX_SIGNING_KEY"] = self.ALT_KEY

        with patch.object(FAISS, "load_local") as mock_load_local:
            with self.assertRaises(IndexSecurityError) as ctx:
                _load()

            mock_load_local.assert_not_called()

        self.assertIn("different key", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # Test E — Missing configuration (fail closed)
    # -------------------------------------------------------------------------
    def test_e_missing_signing_key_fails_closed(self):
        """
        Remove the signing-key configuration from the environment and secrets.
        Verify that loading fails closed with IndexSecurityError without executing
        pickle deserialization.
        """
        chunks = ["Critical Course Material"]
        self._create_signed_index(self.temp_dir, chunks, key=self.TEST_KEY)

        # Remove key configuration completely
        os.environ.pop("STUDYMATE_INDEX_SIGNING_KEY", None)

        mock_st = MagicMock()
        mock_st.secrets.get.return_value = None

        with patch.object(FAISS, "load_local") as mock_load_local, \
             patch.dict("sys.modules", {"streamlit": mock_st}):
            with self.assertRaises(IndexSecurityError) as ctx:
                _load()

            mock_load_local.assert_not_called()

        self.assertIn("signing key is not configured", str(ctx.exception))

    def test_e2_empty_signing_key_fails_closed(self):
        """Verify that a whitespace-only or empty signing key fails closed."""
        os.environ["STUDYMATE_INDEX_SIGNING_KEY"] = "   "
        with self.assertRaises(IndexSecurityError):
            get_signing_key()

    # -------------------------------------------------------------------------
    # Test F — Valid signature with modified FAISS index
    # -------------------------------------------------------------------------
    def test_f_tampered_faiss_binary_rejected_before_loading(self):
        """
        Test the dual-file integrity policy. Modify index.faiss after signing.
        Verify that loading is rejected before FAISS.load_local() is invoked.
        """
        chunks = ["Signal Processing: Fast Fourier Transform"]
        self._create_signed_index(self.temp_dir, chunks, key=self.TEST_KEY)

        # Tamper with index.faiss binary file
        faiss_path = Path(self.temp_dir) / "index.faiss"
        original_bytes = faiss_path.read_bytes()
        faiss_path.write_bytes(original_bytes + b"\xba\xad\xf0\x0d")

        with patch.object(FAISS, "load_local") as mock_load_local:
            with self.assertRaises(IndexSecurityError) as ctx:
                _load()

            mock_load_local.assert_not_called()

        self.assertIn("index.faiss", str(ctx.exception))
        self.assertIn("modified", str(ctx.exception).lower())

    # -------------------------------------------------------------------------
    # Test G — Separate document stores & isolation under security checks
    # -------------------------------------------------------------------------
    def test_g_separate_document_stores_isolated_and_independent(self):
        """
        Verify that each document's integrity check uses the correct index files
        and that failure for one document does not cause another document's index
        to load in its place or compromise other documents.
        """
        doc_a_id = "doc_a.pdf:aaaa1111"
        doc_b_id = "doc_b.pdf:bbbb2222"

        dir_a = get_vectorstore_dir(doc_a_id, root_dir=self.temp_dir)
        dir_b = get_vectorstore_dir(doc_b_id, root_dir=self.temp_dir)

        # Document A is properly created and signed
        self._create_signed_index(dir_a, ["Doc A Chunk 1", "Doc A Chunk 2"], key=self.TEST_KEY)

        # Document B is created but left UNSIGNED (or tampered)
        self._create_raw_index(dir_b, ["Doc B Chunk 1", "Doc B Chunk 2"])

        # 1. Attempting to load Document B must fail closed
        with patch.object(FAISS, "load_local") as mock_load_local:
            with self.assertRaises(IndexSecurityError):
                _load(doc_id=doc_b_id)
            mock_load_local.assert_not_called()

        # 2. Document A must load cleanly with its own data
        chunks_a = get_all_chunks(doc_id=doc_a_id)
        self.assertEqual(chunks_a, ["Doc A Chunk 1", "Doc A Chunk 2"])

        # 3. Verify Document B never contaminated Document A's state or cache
        self.assertNotIn(dir_b, retriever._vectorstores)
        self.assertIn(dir_a, retriever._vectorstores)

    # -------------------------------------------------------------------------
    # Test H — Corrupted signature payload handling
    # -------------------------------------------------------------------------
    def test_h_corrupted_signature_json_rejected(self):
        """Verify that a corrupted or non-JSON index.sig file is rejected safely."""
        chunks = ["Distributed Systems: Paxos and Raft"]
        self._create_signed_index(self.temp_dir, chunks, key=self.TEST_KEY)

        sig_path = Path(self.temp_dir) / SIGNATURE_FILENAME
        sig_path.write_text("{this is not valid json!}", encoding="utf-8")

        with patch.object(FAISS, "load_local") as mock_load_local:
            with self.assertRaises(IndexSecurityError) as ctx:
                _load()
            mock_load_local.assert_not_called()

        self.assertIn("Corrupted or unreadable signature", str(ctx.exception))

    def test_h2_incomplete_signature_keys_rejected(self):
        """Verify that a signature file missing pkl_hmac or faiss_hmac is rejected."""
        chunks = ["Database Systems: Concurrency Control"]
        self._create_signed_index(self.temp_dir, chunks, key=self.TEST_KEY)

        sig_path = Path(self.temp_dir) / SIGNATURE_FILENAME
        sig_path.write_text(json.dumps({"version": 1, "algorithm": "HMAC-SHA256"}), encoding="utf-8")

        with patch.object(FAISS, "load_local") as mock_load_local:
            with self.assertRaises(IndexSecurityError) as ctx:
                _load()
            mock_load_local.assert_not_called()

        self.assertIn("Incomplete signature payload", str(ctx.exception))

    # -------------------------------------------------------------------------
    # Test I — Rebuilding and re-signing index restores validity
    # -------------------------------------------------------------------------
    def test_i_rebuilding_index_restores_validity(self):
        """
        Verify the documented migration workflow:
        An unsigned or tampered index can be re-signed (or re-indexed from PDF)
        with the valid key, after which it loads successfully.
        """
        chunks = ["Rebuilt Section 1", "Rebuilt Section 2"]
        # Start with an unsigned index
        self._create_raw_index(self.temp_dir, chunks)

        # Unsigned fails
        with self.assertRaises(IndexSecurityError):
            _load()

        # Re-sign the index (simulating re-ingestion with signing key configured)
        sign_vectorstore_index(self.temp_dir, key=self.TEST_KEY)

        # Now loading must succeed
        reloaded = get_all_chunks()
        self.assertEqual(reloaded, chunks)


class TestSigningKeyConfiguration(unittest.TestCase):
    """
    Focused tests for signing-key configuration mechanisms:
    1. .env file discovery and loading via python-dotenv.
    2. Fallback line parser when python-dotenv is unavailable.
    3. Environment variable precedence over .env file.
    4. Streamlit secrets fallback.
    5. Fail-closed behavior when key is missing everywhere.
    """

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_env_key = os.environ.get("STUDYMATE_INDEX_SIGNING_KEY")

    def tearDown(self):
        if self.original_env_key is not None:
            os.environ["STUDYMATE_INDEX_SIGNING_KEY"] = self.original_env_key
        else:
            os.environ.pop("STUDYMATE_INDEX_SIGNING_KEY", None)

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_env_file_loads_key_into_environ_when_unset(self):
        """
        When STUDYMATE_INDEX_SIGNING_KEY is not in os.environ,
        load_env_file() reads it from a .env file and makes it accessible via get_signing_key().
        """
        os.environ.pop("STUDYMATE_INDEX_SIGNING_KEY", None)
        env_file = Path(self.temp_dir) / ".env"
        test_key_val = "secret-key-loaded-from-dotenv-file-32bytes!"
        env_file.write_text(f"STUDYMATE_INDEX_SIGNING_KEY={test_key_val}\n", encoding="utf-8")

        load_env_file(dotenv_path=env_file)

        self.assertEqual(os.environ.get("STUDYMATE_INDEX_SIGNING_KEY"), test_key_val)
        self.assertEqual(get_signing_key(), test_key_val)

    def test_env_file_fallback_parser_without_python_dotenv(self):
        """
        When python-dotenv is not installed or import fails, the built-in fallback parser
        in load_env_file() still correctly parses the .env file.
        """
        os.environ.pop("STUDYMATE_INDEX_SIGNING_KEY", None)
        env_file = Path(self.temp_dir) / ".env"
        test_key_val = "fallback-parsed-secret-key-value-12345"
        env_file.write_text(f'STUDYMATE_INDEX_SIGNING_KEY="{test_key_val}"\n# comment\n', encoding="utf-8")

        with patch.dict("sys.modules", {"dotenv": None}):
            load_env_file(dotenv_path=env_file)

        self.assertEqual(os.environ.get("STUDYMATE_INDEX_SIGNING_KEY"), test_key_val)
        self.assertEqual(get_signing_key(), test_key_val)

    def test_os_environ_precedence_over_env_file(self):
        """
        An explicitly set environment variable in os.environ takes precedence
        and is never overwritten by values inside a .env file.
        """
        os.environ["STUDYMATE_INDEX_SIGNING_KEY"] = "explicit-priority-key"
        env_file = Path(self.temp_dir) / ".env"
        env_file.write_text("STUDYMATE_INDEX_SIGNING_KEY=inferior-env-file-key\n", encoding="utf-8")

        load_env_file(dotenv_path=env_file)

        self.assertEqual(get_signing_key(), "explicit-priority-key")

    def test_missing_key_everywhere_fails_closed(self):
        """
        When no key is configured in os.environ, .env does not exist,
        and no Streamlit secrets exist, get_signing_key() raises IndexSecurityError.
        """
        os.environ.pop("STUDYMATE_INDEX_SIGNING_KEY", None)
        non_existent_env = Path(self.temp_dir) / "does_not_exist.env"

        mock_st = MagicMock()
        mock_st.secrets.get.return_value = None

        with patch("security.load_env_file"), \
             patch.dict("sys.modules", {"streamlit": mock_st}):
            with self.assertRaises(IndexSecurityError) as ctx:
                get_signing_key()

        self.assertIn("FAISS index signing key is not configured", str(ctx.exception))
        self.assertIn(".env file", str(ctx.exception))

    def test_streamlit_secrets_fallback(self):
        """
        When key is not in os.environ and no .env file provides it,
        get_signing_key() retrieves it from streamlit.secrets.
        """
        os.environ.pop("STUDYMATE_INDEX_SIGNING_KEY", None)
        mock_st = MagicMock()
        mock_st.secrets.get.return_value = "secret-from-streamlit-secrets-store"

        with patch.dict("sys.modules", {"streamlit": mock_st}):
            with patch("security.load_env_file"):
                key = get_signing_key()

        self.assertEqual(key, "secret-from-streamlit-secrets-store")


if __name__ == "__main__":
    unittest.main()
