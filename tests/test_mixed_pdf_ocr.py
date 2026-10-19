import io
import os
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
import ingest
import handwriting_ocr
from ingest import ingest_pdf, MAX_OCR_PAGES
from handwriting_ocr import recognize_handwritten_pdf


def make_pdf(pages_content):
    """
    Create an in-memory PDF from a list of page specifications.
    If content is non-empty string, insert text on that page.
    If content is None or empty string, create a blank page without selectable text.
    """
    doc = pymupdf.open()
    for content in pages_content:
        page = doc.new_page(width=595, height=842)
        if content and content.strip():
            page.insert_text((50, 100), content.strip())
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestMixedPdfOcrIngestion(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory for vectorstore output during tests
        self.temp_dir = tempfile.TemporaryDirectory()
        self.vectorstore_patcher = patch.object(
            ingest, "VECTORSTORE_DIR", self.temp_dir.name
        )
        self.vectorstore_patcher.start()

        # Mock embeddings and FAISS to make unit tests fast and deterministic
        self.embeddings_patcher = patch("ingest.HuggingFaceEmbeddings")
        self.mock_embeddings_cls = self.embeddings_patcher.start()
        self.mock_embeddings = MagicMock()
        self.mock_embeddings_cls.return_value = self.mock_embeddings

        self.faiss_patcher = patch("ingest.FAISS")
        self.mock_faiss_cls = self.faiss_patcher.start()
        self.mock_vectorstore = MagicMock()
        self.mock_faiss_cls.from_texts.return_value = self.mock_vectorstore

    def tearDown(self):
        self.faiss_patcher.stop()
        self.embeddings_patcher.stop()
        self.vectorstore_patcher.stop()
        self.temp_dir.cleanup()

    # -------------------------------------------------------------------------
    # Test A — Text-only PDF
    # -------------------------------------------------------------------------
    def test_a_text_only_pdf(self):
        """
        Input: A PDF where all pages contain selectable text.
        Expected:
        - Every page's text is retained.
        - OCR is not called.
        - Chunking and indexing complete successfully.
        - The function returns the chunk count.
        """
        pdf_bytes = make_pdf([
            "Chapter 1: Principles of Operating Systems and Kernel Architecture.",
            "Chapter 2: Memory Management, Virtual Memory, and Paging Algorithms.",
            "Chapter 3: File Systems, Inodes, and Directory Structures.",
        ])

        with patch("handwriting_ocr.recognize_handwritten_pdf") as mock_ocr:
            chunk_count = ingest_pdf(pdf_bytes)

            # OCR should not be called at all
            mock_ocr.assert_not_called()

            # FAISS from_texts was called with chunks containing all 3 pages
            self.mock_faiss_cls.from_texts.assert_called_once()
            passed_texts = self.mock_faiss_cls.from_texts.call_args[1]["texts"]
            combined = " ".join(passed_texts)
            self.assertIn("Operating Systems", combined)
            self.assertIn("Memory Management", combined)
            self.assertIn("File Systems", combined)

            # Vectorstore save_local called
            self.mock_vectorstore.save_local.assert_called_once_with(self.temp_dir.name)

            # Return value is an integer chunk count
            self.assertIsInstance(chunk_count, int)
            self.assertEqual(chunk_count, len(passed_texts))
            self.assertGreater(chunk_count, 0)

    # -------------------------------------------------------------------------
    # Test B — Scanned-only PDF
    # -------------------------------------------------------------------------
    def test_b_scanned_only_pdf(self):
        """
        Input: A PDF with no selectable text on its content pages.
        Expected:
        - OCR is invoked for each page requiring OCR.
        - Recognized text is included in the indexed content.
        - The ingestion succeeds when OCR returns usable text.
        """
        pdf_bytes = make_pdf(["", ""])  # 2 scanned pages

        def ocr_side_effect(page_bytes, max_pages=10, page_number_offset=0):
            return f"[Page {page_number_offset + 1}]\nRecognized handwriting on page {page_number_offset + 1}"

        with patch("handwriting_ocr.recognize_handwritten_pdf", side_effect=ocr_side_effect) as mock_ocr:
            chunk_count = ingest_pdf(pdf_bytes)

            # OCR must be invoked once per scanned page (2 times total)
            self.assertEqual(mock_ocr.call_count, 2)

            # Verify offsets passed
            call_0 = mock_ocr.call_args_list[0]
            call_1 = mock_ocr.call_args_list[1]
            self.assertEqual(call_0[1]["page_number_offset"], 0)
            self.assertEqual(call_1[1]["page_number_offset"], 1)

            # Recognized text is included in indexed content
            passed_texts = self.mock_faiss_cls.from_texts.call_args[1]["texts"]
            combined = " ".join(passed_texts)
            self.assertIn("Recognized handwriting on page 1", combined)
            self.assertIn("Recognized handwriting on page 2", combined)

            self.assertIsInstance(chunk_count, int)
            self.assertGreater(chunk_count, 0)

    # -------------------------------------------------------------------------
    # Test C — Mixed PDF
    # -------------------------------------------------------------------------
    def test_c_mixed_pdf(self):
        """
        Input: A PDF containing both selectable-text pages and scanned pages.
        Expected:
        - Selectable text is extracted normally.
        - OCR is invoked only for pages without selectable text.
        - No page's content is omitted solely because another page contains selectable text.
        - The final text preserves the original page order.
        """
        pdf_bytes = make_pdf([
            "Page 1: Digital selectable lecture notes on Sorting Algorithms.",
            "",  # Page 2 scanned diagram/handwritten notes
            "Page 3: Digital selectable lecture notes on Graph Traversals.",
        ])

        def ocr_side_effect(page_bytes, max_pages=10, page_number_offset=0):
            return f"[Page {page_number_offset + 1}]\nHandwritten notes for Quicksort partitioning."

        with patch("handwriting_ocr.recognize_handwritten_pdf", side_effect=ocr_side_effect) as mock_ocr:
            chunk_count = ingest_pdf(pdf_bytes)

            # OCR invoked ONLY for page 2 (exactly 1 call)
            self.assertEqual(mock_ocr.call_count, 1)
            self.assertEqual(mock_ocr.call_args[1]["page_number_offset"], 1)

            # Check that content from all 3 pages is present
            passed_texts = self.mock_faiss_cls.from_texts.call_args[1]["texts"]
            combined = " ".join(passed_texts)
            self.assertIn("Sorting Algorithms", combined)
            self.assertIn("Quicksort partitioning", combined)
            self.assertIn("Graph Traversals", combined)

            # Check page order: Page 1 appears before Page 2 OCR, which appears before Page 3
            idx_p1 = combined.find("Sorting Algorithms")
            idx_p2 = combined.find("Quicksort partitioning")
            idx_p3 = combined.find("Graph Traversals")
            self.assertTrue(idx_p1 < idx_p2 < idx_p3)

            self.assertIsInstance(chunk_count, int)
            self.assertGreater(chunk_count, 0)

    # -------------------------------------------------------------------------
    # Test D — Alternating page types
    # -------------------------------------------------------------------------
    def test_d_alternating_page_types(self):
        """
        Input: A PDF with sequence text, scan, text, scan.
        Expected:
        - Both sources of text are retained in the correct order.
        - Each scanned page is processed individually.
        - Original page numbering is maintained.
        """
        pdf_bytes = make_pdf([
            "Section A: Introduction to Database Systems (Page 1)",
            "",  # Page 2 scanned
            "Section B: Relational Algebra and SQL Queries (Page 3)",
            "",  # Page 4 scanned
        ])

        def ocr_side_effect(page_bytes, max_pages=10, page_number_offset=0):
            return f"[Page {page_number_offset + 1}]\nHandwritten query tree diagram page {page_number_offset + 1}"

        with patch("handwriting_ocr.recognize_handwritten_pdf", side_effect=ocr_side_effect) as mock_ocr:
            chunk_count = ingest_pdf(pdf_bytes)

            self.assertEqual(mock_ocr.call_count, 2)
            self.assertEqual(mock_ocr.call_args_list[0][1]["page_number_offset"], 1)  # Page 2
            self.assertEqual(mock_ocr.call_args_list[1][1]["page_number_offset"], 3)  # Page 4

            passed_texts = self.mock_faiss_cls.from_texts.call_args[1]["texts"]
            combined = " ".join(passed_texts)

            # Confirm page ordering and numbering
            idx_1 = combined.find("Section A: Introduction to Database Systems (Page 1)")
            idx_2 = combined.find("[Page 2]")
            idx_3 = combined.find("Section B: Relational Algebra and SQL Queries (Page 3)")
            idx_4 = combined.find("[Page 4]")

            self.assertTrue(idx_1 < idx_2 < idx_3 < idx_4)
            self.assertGreater(chunk_count, 0)

    # -------------------------------------------------------------------------
    # Test E — OCR page limit
    # -------------------------------------------------------------------------
    def test_e_ocr_page_limit_exceeded(self):
        """
        Input: A document with more than 10 pages requiring OCR (11 scanned pages).
        Expected:
        - Ingestion stops safely with clear ValueError.
        - Explains that the document should be split into smaller parts.
        - OCR is not called on any pages.
        """
        pdf_bytes = make_pdf([""] * 11)  # 11 pages requiring OCR

        with patch("handwriting_ocr.recognize_handwritten_pdf") as mock_ocr:
            with self.assertRaises(ValueError) as ctx:
                ingest_pdf(pdf_bytes)

            self.assertIn("11 pages requiring OCR", str(ctx.exception))
            self.assertIn("smaller parts", str(ctx.exception))
            mock_ocr.assert_not_called()

    def test_e_ocr_page_limit_allowed_with_many_text_pages(self):
        """
        Input: A document with 14 total pages: 12 selectable-text pages and 2 scanned pages.
        Expected:
        - Not rejected merely because total page count (14) > 10.
        - Ingestion completes successfully.
        - OCR invoked exactly 2 times for the 2 scanned pages.
        """
        pages = ["Selectable text content on digital page " + str(i) for i in range(1, 13)]
        pages.append("")  # Page 13 scanned
        pages.append("")  # Page 14 scanned
        pdf_bytes = make_pdf(pages)

        with patch("handwriting_ocr.recognize_handwritten_pdf", return_value="[Page X]\nNotes") as mock_ocr:
            chunk_count = ingest_pdf(pdf_bytes)
            self.assertEqual(mock_ocr.call_count, 2)
            self.assertGreater(chunk_count, 0)

    # -------------------------------------------------------------------------
    # Test F — Empty OCR result
    # -------------------------------------------------------------------------
    def test_f_empty_ocr_result_mixed_pdf_raises_page_specific_error(self):
        """
        Input: A mixed PDF where a scanned page (page 2) returns empty OCR text.
        Expected:
        - Raises ValueError identifying the original page number (page 2).
        - FAISS vector store is NOT created or saved, even though page 1 had text.
        """
        pdf_bytes = make_pdf([
            "Selectable text on page 1.",
            "",  # Scanned page 2
        ])

        with patch("handwriting_ocr.recognize_handwritten_pdf", return_value="") as mock_ocr:
            with self.assertRaises(ValueError) as ctx:
                ingest_pdf(pdf_bytes)

            error_msg = str(ctx.exception)
            self.assertIn("page 2", error_msg.lower())
            self.assertIn("returned no text on page 2", error_msg)

            # Vector store MUST NOT be saved or created
            self.mock_vectorstore.save_local.assert_not_called()
            self.mock_faiss_cls.from_texts.assert_not_called()

    def test_f_empty_ocr_result_whitespace_only_raises_page_specific_error(self):
        """
        Input: A mixed PDF where a scanned page (page 2) returns whitespace-only OCR text.
        Expected:
        - Raises ValueError identifying the original page number (page 2).
        - FAISS vector store is NOT created or saved.
        """
        pdf_bytes = make_pdf([
            "Selectable text on page 1.",
            "",  # Scanned page 2
        ])

        with patch("handwriting_ocr.recognize_handwritten_pdf", return_value="   \n \t  ") as mock_ocr:
            with self.assertRaises(ValueError) as ctx:
                ingest_pdf(pdf_bytes)

            error_msg = str(ctx.exception)
            self.assertIn("page 2", error_msg.lower())
            self.mock_vectorstore.save_local.assert_not_called()
            self.mock_faiss_cls.from_texts.assert_not_called()

    def test_f_empty_ocr_result_all_scanned_pdf_raises_error(self):
        """
        Input: An all-scanned PDF where page 1 returns empty OCR text.
        Expected:
        - Clear error raised identifying page 1.
        - Vector store is NOT saved.
        """
        pdf_bytes = make_pdf(["", ""])

        with patch("handwriting_ocr.recognize_handwritten_pdf", return_value="") as mock_ocr:
            with self.assertRaises(ValueError) as ctx:
                ingest_pdf(pdf_bytes)

            error_msg = str(ctx.exception)
            self.assertIn("page 1", error_msg.lower())
            self.mock_vectorstore.save_local.assert_not_called()
            self.mock_faiss_cls.from_texts.assert_not_called()

    # -------------------------------------------------------------------------
    # Test G — OCR exception
    # -------------------------------------------------------------------------
    def test_g_ocr_exception_identifies_page(self):
        """
        Input: OCR raises an exception for a particular page.
        Expected:
        - Handled according to existing conventions (raises ValueError).
        - Error identifies original page number.
        - Document is not reported as fully successful.
        """
        pdf_bytes = make_pdf([
            "Page 1 selectable text.",
            "",  # Page 2 scanned
            "Page 3 selectable text.",
        ])

        with patch("handwriting_ocr.recognize_handwritten_pdf", side_effect=RuntimeError("CUDA out of memory")):
            with self.assertRaises(ValueError) as ctx:
                ingest_pdf(pdf_bytes)

            error_msg = str(ctx.exception)
            self.assertIn("page 2", error_msg.lower())
            self.assertIn("Local handwriting recognition failed on page 2", error_msg)
            self.assertIn("CUDA out of memory", error_msg)
            self.mock_vectorstore.save_local.assert_not_called()

    # -------------------------------------------------------------------------
    # Test H — Existing compatibility
    # -------------------------------------------------------------------------
    def test_h1_recognize_handwritten_pdf_default_offset(self):
        """
        Verify recognize_handwritten_pdf() defaults to page_number_offset=0
        and labels pages starting at Page 1.
        """
        pdf_bytes = make_pdf([""])

        mock_processor = MagicMock()
        mock_processor.batch_decode.return_value = ["Sample transcribed line"]
        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]

        with patch("handwriting_ocr.load_handwriting_model", return_value=(mock_processor, mock_model, "cpu")), \
             patch("handwriting_ocr.detect_text_lines", return_value=[MagicMock()]):
            result = recognize_handwritten_pdf(pdf_bytes)
            self.assertIn("[Page 1]", result)
            self.assertIn("Sample transcribed line", result)

    def test_h2_recognize_handwritten_pdf_with_offset(self):
        """
        Verify recognize_handwritten_pdf() with page_number_offset=3
        labels the first page as [Page 4].
        """
        pdf_bytes = make_pdf([""])

        mock_processor = MagicMock()
        mock_processor.batch_decode.return_value = ["Sample transcribed line on page 4"]
        mock_model = MagicMock()
        mock_model.generate.return_value = [[1, 2, 3]]

        with patch("handwriting_ocr.load_handwriting_model", return_value=(mock_processor, mock_model, "cpu")), \
             patch("handwriting_ocr.detect_text_lines", return_value=[MagicMock()]):
            result = recognize_handwritten_pdf(pdf_bytes, page_number_offset=3)
            self.assertIn("[Page 4]", result)
            self.assertIn("Sample transcribed line on page 4", result)

    def test_h3_recognize_handwritten_pdf_max_pages_limit(self):
        """
        Verify recognize_handwritten_pdf() still enforces max_pages check
        when called directly on a multi-page PDF.
        """
        pdf_bytes = make_pdf(["", ""])  # 2 pages

        mock_processor = MagicMock()
        mock_model = MagicMock()

        with patch("handwriting_ocr.load_handwriting_model", return_value=(mock_processor, mock_model, "cpu")):
            with self.assertRaises(ValueError) as ctx:
                recognize_handwritten_pdf(pdf_bytes, max_pages=1)
            self.assertIn("PDF has 2 pages", str(ctx.exception))
            self.assertIn("at most 1 pages", str(ctx.exception))

    def test_h4_ingest_pdf_input_types(self):
        """
        Verify ingest_pdf() supports file path (str/Path), bytes, and BytesIO.
        """
        pdf_bytes = make_pdf(["Selectable text for input types test."])

        with patch("handwriting_ocr.recognize_handwritten_pdf") as mock_ocr:
            # Test with bytes
            count_bytes = ingest_pdf(pdf_bytes)
            self.assertGreater(count_bytes, 0)

            # Test with BytesIO
            count_stream = ingest_pdf(io.BytesIO(pdf_bytes))
            self.assertGreater(count_stream, 0)

            # Test with temporary file path (str and Path)
            temp_pdf = Path(self.temp_dir.name) / "test.pdf"
            temp_pdf.write_bytes(pdf_bytes)

            count_str = ingest_pdf(str(temp_pdf))
            self.assertGreater(count_str, 0)

            count_path = ingest_pdf(temp_pdf)
            self.assertGreater(count_path, 0)

            mock_ocr.assert_not_called()


if __name__ == "__main__":
    unittest.main()
