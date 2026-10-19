import io
import os
import sys
import tempfile
import unittest
from pathlib import Path
from PIL import Image, ImageDraw

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pymupdf
import ingest


class TestRealOcrEndToEnd(unittest.TestCase):
    """
    Real end-to-end integration test executing real TrOCR inference
    and real HuggingFace embeddings / FAISS on a mixed PDF.
    """

    def setUp(self):
        os.environ.setdefault(
            "STUDYMATE_INDEX_SIGNING_KEY",
            "test-signing-key-for-automated-regression-tests-only-32chars",
        )

    def test_e2e_mixed_pdf_with_real_ocr_model(self):
        # 1. Create a 2-page mixed PDF:
        # Page 1: Selectable text
        # Page 2: Rendered handwriting image with no selectable text
        doc = pymupdf.open()
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text((50, 100), "Section 1: Digital lecture on Modern Artificial Intelligence Architectures.")

        img = Image.new("RGB", (600, 150), color="white")
        draw = ImageDraw.Draw(img)
        draw.text((30, 60), "Artificial Intelligence", fill="black")
        img_buf = io.BytesIO()
        img.save(img_buf, format="PNG")

        p2 = doc.new_page(width=595, height=842)
        p2.insert_image(pymupdf.Rect(50, 100, 550, 250), stream=img_buf.getvalue())
        pdf_bytes = doc.tobytes()
        doc.close()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Point to a temporary vectorstore directory to avoid touching production data
            orig_dir = ingest.VECTORSTORE_DIR
            try:
                ingest.VECTORSTORE_DIR = tmpdir
                chunk_count = ingest.ingest_pdf(pdf_bytes)

                self.assertIsInstance(chunk_count, int)
                self.assertGreater(chunk_count, 0)
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "index.faiss")))
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "index.pkl")))
                self.assertTrue(os.path.exists(os.path.join(tmpdir, "index.sig")))
            finally:
                ingest.VECTORSTORE_DIR = orig_dir


if __name__ == "__main__":
    unittest.main()
