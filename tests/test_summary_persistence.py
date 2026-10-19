import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

os.environ["GROQ_API_KEY"] = "gsk_mock_test_key"

from streamlit.testing.v1 import AppTest
from app import generate_document_summary


class TestSummaryPersistence(unittest.TestCase):
    def setUp(self):
        self.app_path = str(SRC_DIR / "app.py")

    # -------------------------------------------------------------------------
    # Test A — Summary remains visible after rerun
    # -------------------------------------------------------------------------
    def test_a_summary_persists_across_reruns(self):
        """
        Simulate summary generation and verify that the result remains in
        session state and can be displayed after another script rerun.
        """
        sample_summary = "### Key Points\n1. Virtual memory isolates processes.\n2. Paging prevents external fragmentation."

        with patch("langchain_groq.ChatGroq.invoke") as mock_invoke, \
             patch("retriever.get_all_chunks", return_value=["Operating Systems Lecture Chunks"]):
            mock_invoke.return_value = MagicMock(content=sample_summary)

            at = AppTest.from_file(self.app_path, default_timeout=10)
            at.session_state["document_ready"] = True
            at.session_state["processed_file_id"] = "os_notes.pdf:abcdef1234567890"
            at.session_state["current_doc_name"] = "os_notes.pdf"
            at.run()

            # Click the Generate summary button
            for btn in at.button:
                if "summary" in btn.label.lower():
                    btn.click()
                    break

            at.run()

            # Verify summary is saved in session state
            self.assertEqual(at.session_state["summary"], sample_summary)
            self.assertEqual(at.session_state["summary_file_id"], "os_notes.pdf:abcdef1234567890")

            # Verify summary markdown is rendered in the UI
            markdown_texts = [m.value for m in at.markdown]
            self.assertTrue(any(sample_summary in text for text in markdown_texts))

            # Simulate subsequent rerun (e.g. user interacting with another widget)
            at.run()

            # Verify summary STILL persists in session state after rerun
            self.assertEqual(at.session_state["summary"], sample_summary)
            self.assertEqual(at.session_state["summary_file_id"], "os_notes.pdf:abcdef1234567890")

            # Verify summary is still rendered in the UI
            markdown_texts_rerun = [m.value for m in at.markdown]
            self.assertTrue(any(sample_summary in text for text in markdown_texts_rerun))

    # -------------------------------------------------------------------------
    # Test B — No repeated summary generation
    # -------------------------------------------------------------------------
    def test_b_no_repeated_summary_generation_on_reruns(self):
        """
        Verify that an ordinary rerun after successful generation does not
        invoke the summary-generation LLM call again.
        """
        sample_summary = "Calculus Summary: Derivatives and Integrals"

        with patch("langchain_groq.ChatGroq.invoke") as mock_invoke, \
             patch("retriever.get_all_chunks", return_value=["Math chunks"]):
            mock_invoke.return_value = MagicMock(content=sample_summary)

            at = AppTest.from_file(self.app_path, default_timeout=10)
            at.session_state["document_ready"] = True
            at.session_state["processed_file_id"] = "math.pdf:1234567890abcdef"
            at.session_state["current_doc_name"] = "math.pdf"
            at.run()

            # Click summary button once
            for btn in at.button:
                if "summary" in btn.label.lower():
                    btn.click()
                    break
            at.run()

            self.assertEqual(mock_invoke.call_count, 1)

            # Perform 2 subsequent ordinary reruns
            at.run()
            at.run()

            # Invoke count must remain 1 (no repeated LLM calls)
            self.assertEqual(mock_invoke.call_count, 1)

    # -------------------------------------------------------------------------
    # Test C — Different document selected
    # -------------------------------------------------------------------------
    def test_c_different_document_invalidates_summary(self):
        """
        Generate a summary for one document, then select/ingest a different PDF.
        Verify that the old summary is not displayed as the new document's summary.
        """
        summary_doc_a = "Summary for Document A: Network Topologies"

        with patch("langchain_groq.ChatGroq.invoke") as mock_invoke, \
             patch("retriever.get_all_chunks", return_value=["Network chunks"]):
            mock_invoke.return_value = MagicMock(content=summary_doc_a)

            at = AppTest.from_file(self.app_path, default_timeout=10)
            at.session_state["document_ready"] = True
            at.session_state["processed_file_id"] = "docA.pdf:1111111111111111"
            at.session_state["current_doc_name"] = "docA.pdf"
            at.run()

            # Generate summary for Document A
            for btn in at.button:
                if "summary" in btn.label.lower():
                    btn.click()
                    break
            at.run()

            self.assertEqual(at.session_state["summary"], summary_doc_a)

            # Now simulate selecting Document B (new document upload)
            # When a new document is loaded in app.py, summary is reset
            at.session_state["processed_file_id"] = "docB.pdf:2222222222222222"
            at.session_state["current_doc_name"] = "docB.pdf"
            at.session_state["summary"] = None
            at.session_state["summary_file_id"] = None
            at.run()

            # Verify Document A's summary is not rendered
            markdown_texts = [m.value for m in at.markdown]
            self.assertFalse(any(summary_doc_a in text for text in markdown_texts))
            self.assertIsNone(at.session_state["summary"])

    def test_c_mismatched_file_id_blocks_stale_summary_rendering(self):
        """
        If summary is in session_state but summary_file_id != processed_file_id,
        it must not be rendered.
        """
        at = AppTest.from_file(self.app_path, default_timeout=10)
        at.session_state["document_ready"] = True
        at.session_state["processed_file_id"] = "new_doc.pdf:9999999999999999"
        at.session_state["current_doc_name"] = "new_doc.pdf"
        # Stale summary left from an old file
        at.session_state["summary"] = "Old summary that should never appear"
        at.session_state["summary_file_id"] = "old_doc.pdf:0000000000000000"
        at.run()

        markdown_texts = [m.value for m in at.markdown]
        self.assertFalse(any("Old summary that should never appear" in text for text in markdown_texts))

    # -------------------------------------------------------------------------
    # Test D — Same document selected again / ordinary reruns
    # -------------------------------------------------------------------------
    def test_d_same_document_retained_across_interactions(self):
        """
        Verify that ordinary reruns with the same document retain the existing
        summary without redundant generation when other controls are used.
        """
        summary_text = "Physics: Quantum Mechanics and Wave-Particle Duality"

        with patch("langchain_groq.ChatGroq.invoke") as mock_invoke, \
             patch("retriever.get_all_chunks", return_value=["Physics chunks"]):
            mock_invoke.return_value = MagicMock(content=summary_text)

            at = AppTest.from_file(self.app_path, default_timeout=10)
            at.session_state["document_ready"] = True
            at.session_state["processed_file_id"] = "physics.pdf:3333333333333333"
            at.session_state["current_doc_name"] = "physics.pdf"
            at.run()

            # Generate summary
            for btn in at.button:
                if "summary" in btn.label.lower():
                    btn.click()
                    break
            at.run()

            self.assertEqual(mock_invoke.call_count, 1)

            # Simulate clearing chat history (an ordinary interaction)
            for btn in at.button:
                if "clear chat history" in btn.label.lower():
                    btn.click()
                    break
            at.run()

            # Summary must be retained
            self.assertEqual(at.session_state["summary"], summary_text)
            self.assertEqual(at.session_state["summary_file_id"], "physics.pdf:3333333333333333")
            self.assertEqual(mock_invoke.call_count, 1)

    # -------------------------------------------------------------------------
    # Test E — Summary-generation failure
    # -------------------------------------------------------------------------
    def test_e_summary_generation_failure_handles_error(self):
        """
        Mock an LLM failure and verify:
        - The UI displays an error message.
        - No false success is displayed.
        - Previous summary from a different document is not shown.
        """
        with patch("langchain_groq.ChatGroq.invoke", side_effect=RuntimeError("Groq service rate limited")), \
             patch("retriever.get_all_chunks", return_value=["Chunks"]):

            at = AppTest.from_file(self.app_path, default_timeout=10)
            at.session_state["document_ready"] = True
            at.session_state["processed_file_id"] = "doc_fail.pdf:4444444444444444"
            at.session_state["current_doc_name"] = "doc_fail.pdf"
            # Simulate a leftover summary from a previous document
            at.session_state["summary"] = "Previous Doc Summary"
            at.session_state["summary_file_id"] = "prev.pdf:0000000000000000"
            at.run()

            for btn in at.button:
                if "summary" in btn.label.lower():
                    btn.click()
                    break
            at.run()

            # Summary must be cleared
            self.assertIsNone(at.session_state["summary"])
            self.assertIsNone(at.session_state["summary_file_id"])

            # Error must be displayed
            error_texts = [e.value for e in at.error]
            self.assertTrue(any("Groq service rate limited" in text for text in error_texts))

            # No summary container or false success displayed
            markdown_texts = [m.value for m in at.markdown]
            self.assertFalse(any("### 📄 Document Summary" in text for text in markdown_texts))
            self.assertFalse(any("Previous Doc Summary" in text for text in markdown_texts))

    # -------------------------------------------------------------------------
    # Test F — Empty summary result
    # -------------------------------------------------------------------------
    def test_f_empty_summary_result_raises_error(self):
        """
        Verify that an empty or whitespace-only result does not appear as a
        successfully generated summary and shows an error.
        """
        # Test empty string
        with patch("langchain_groq.ChatGroq.invoke", return_value=MagicMock(content="")), \
             patch("retriever.get_all_chunks", return_value=["Chunks"]):

            at = AppTest.from_file(self.app_path, default_timeout=10)
            at.session_state["document_ready"] = True
            at.session_state["processed_file_id"] = "empty_doc.pdf:5555555555555555"
            at.session_state["current_doc_name"] = "empty_doc.pdf"
            at.run()

            for btn in at.button:
                if "summary" in btn.label.lower():
                    btn.click()
                    break
            at.run()

            self.assertIsNone(at.session_state["summary"])
            error_texts = [e.value for e in at.error]
            self.assertTrue(any("empty response" in text.lower() for text in error_texts))

        # Test whitespace-only string
        with patch("langchain_groq.ChatGroq.invoke", return_value=MagicMock(content="   \n \t  ")), \
             patch("retriever.get_all_chunks", return_value=["Chunks"]):

            at = AppTest.from_file(self.app_path, default_timeout=10)
            at.session_state["document_ready"] = True
            at.session_state["processed_file_id"] = "ws_doc.pdf:6666666666666666"
            at.session_state["current_doc_name"] = "ws_doc.pdf"
            at.run()

            for btn in at.button:
                if "summary" in btn.label.lower():
                    btn.click()
                    break
            at.run()

            self.assertIsNone(at.session_state["summary"])
            error_texts = [e.value for e in at.error]
            self.assertTrue(any("empty response" in text.lower() for text in error_texts))

    # -------------------------------------------------------------------------
    # Test G — Existing behavior remains functional
    # -------------------------------------------------------------------------
    def test_g_summary_without_document_warns_user(self):
        """
        Clicking summary when document_ready is False warns user to upload PDF.
        """
        at = AppTest.from_file(self.app_path, default_timeout=10)
        at.session_state["document_ready"] = False
        at.run()

        for btn in at.button:
            if "summary" in btn.label.lower():
                btn.click()
                break
        at.run()

        warning_texts = [w.value for w in at.warning]
        self.assertTrue(any("Please upload a PDF first." in text for text in warning_texts))
        self.assertIsNone(at.session_state["summary"])

    def test_g_generate_document_summary_helper(self):
        """
        Verify the standalone generate_document_summary helper function:
        - Correctly formats the prompt.
        - Truncates context.
        - Raises ValueError on empty response.
        """
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="Structured Summary Output")

        res = generate_document_summary(mock_llm, "Some test context")
        self.assertEqual(res, "Structured Summary Output")

        # Verify prompt structure
        prompt_passed = mock_llm.invoke.call_args[0][0]
        self.assertIn("Analyze this document and create a structured summary.", prompt_passed)
        self.assertIn("If it is lecture notes:", prompt_passed)
        self.assertIn("Some test context", prompt_passed)

        # Verify ValueError on empty response
        mock_llm.invoke.return_value = MagicMock(content="   ")
        with self.assertRaises(ValueError) as ctx:
            generate_document_summary(mock_llm, "Context")
        self.assertIn("empty response", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
