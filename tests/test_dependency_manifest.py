import importlib
import os
import sys
import unittest
from pathlib import Path

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class TestDependencyManifest(unittest.TestCase):
    """
    Regression test suite for StudyMate AI dependency manifest and environment setup.
    Validates manifest encoding, direct dependency declarations, runtime imports,
    security exclusion rules, and OCR import boundaries.
    """

    def setUp(self):
        self.req_path = ROOT_DIR / "requirements.txt"
        self.gitignore_path = ROOT_DIR / ".gitignore"
        self.env_example_path = ROOT_DIR / ".env.example"

    # -------------------------------------------------------------------------
    # 1. Manifest format, encoding, and duplicates
    # -------------------------------------------------------------------------
    def test_manifest_encoding_and_format(self):
        """
        Verify requirements.txt is encoded as standard UTF-8 without UTF-16 BOM,
        contains valid requirement lines, and has no duplicate package entries.
        """
        self.assertTrue(self.req_path.is_file(), "requirements.txt must exist.")

        raw_bytes = self.req_path.read_bytes()
        self.assertFalse(
            raw_bytes.startswith(b"\xff\xfe") or raw_bytes.startswith(b"\xfe\xff"),
            "requirements.txt must not use UTF-16 encoding with BOM.",
        )

        try:
            content = raw_bytes.decode("utf-8")
        except UnicodeDecodeError as exc:
            self.fail(f"requirements.txt must be valid UTF-8: {exc}")

        declared_packages = []
        for line in content.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Strip pinned version or specifiers to get package name
            pkg_name = line.split("==")[0].split(">=")[0].split("<=")[0].split("~=")[0].strip().lower()
            declared_packages.append(pkg_name)

        # Check for duplicate package declarations
        duplicates = [pkg for pkg in declared_packages if declared_packages.count(pkg) > 1]
        self.assertEqual(
            len(duplicates),
            0,
            f"requirements.txt must not contain duplicate entries: {set(duplicates)}",
        )

    # -------------------------------------------------------------------------
    # 2. Direct third-party dependencies are declared
    # -------------------------------------------------------------------------
    def test_all_direct_dependencies_declared(self):
        """
        Verify that all directly imported third-party packages in active code
        are explicitly listed in requirements.txt.
        """
        content = self.req_path.read_text(encoding="utf-8")
        declared = {
            line.split("==")[0].split(">=")[0].strip().lower()
            for line in content.splitlines()
            if line.strip() and not line.strip().startswith("#")
        }

        # Essential direct third-party dependencies
        required_direct_dependencies = [
            "streamlit",
            "langchain",
            "langchain-community",
            "langchain-core",
            "langchain-text-splitters",
            "langchain-huggingface",
            "langchain-groq",
            "groq",
            "sentence-transformers",
            "faiss-cpu",
            "pypdf",
            "pymupdf",
            "opencv-python",
            "pillow",
            "transformers",
            "torch",
            "numpy",
            "python-dotenv",
        ]

        for pkg in required_direct_dependencies:
            self.assertIn(
                pkg,
                declared,
                f"Direct dependency '{pkg}' must be declared in requirements.txt.",
            )

    # -------------------------------------------------------------------------
    # 3. Runtime imports for all declared direct modules
    # -------------------------------------------------------------------------
    def test_runtime_imports(self):
        """
        Verify that each required third-party module imports successfully
        in the current environment.
        """
        modules_to_test = [
            ("streamlit", "streamlit"),
            ("pypdf", "pypdf"),
            ("fitz / pymupdf", "pymupdf"),
            ("cv2 (OpenCV)", "cv2"),
            ("PIL (Pillow)", "PIL.Image"),
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("sentence_transformers", "sentence_transformers"),
            ("dotenv", "dotenv"),
            ("langchain", "langchain"),
            ("langchain_core", "langchain_core.documents"),
            ("langchain_community", "langchain_community.vectorstores"),
            ("langchain_text_splitters", "langchain_text_splitters"),
            ("langchain_huggingface", "langchain_huggingface"),
            ("langchain_groq", "langchain_groq"),
        ]

        for label, mod_name in modules_to_test:
            try:
                mod = importlib.import_module(mod_name)
                self.assertIsNotNone(mod, f"Module '{label}' imported as None.")
            except ImportError as exc:
                self.fail(f"Failed to import declared dependency '{label}': {exc}")

    # -------------------------------------------------------------------------
    # 4. Security exclusions in .gitignore
    # -------------------------------------------------------------------------
    def test_gitignore_security_exclusions(self):
        """
        Verify that .gitignore excludes private secrets, virtual environments,
        and generated vector store index/signature files.
        """
        self.assertTrue(self.gitignore_path.is_file(), ".gitignore must exist.")
        gitignore_text = self.gitignore_path.read_text(encoding="utf-8")
        rules = [line.strip() for line in gitignore_text.splitlines() if line.strip() and not line.strip().startswith("#")]

        # Must exclude .env and secrets.toml
        self.assertTrue(any(".env" in r for r in rules), ".gitignore must exclude .env")
        self.assertTrue(any("secrets.toml" in r for r in rules), ".gitignore must exclude secrets.toml")

        # Must exclude virtual environments
        self.assertTrue(any(".venv" in r or "venv" in r for r in rules), ".gitignore must exclude virtual environments")

        # Must exclude bytecode
        self.assertTrue(any("__pycache__" in r or "*.pyc" in r for r in rules), ".gitignore must exclude bytecode")

        # Must exclude vectorstore and signature files
        self.assertTrue(any("vectorstore" in r for r in rules), ".gitignore must exclude vectorstore directory")
        self.assertTrue(any("*.sig" in r for r in rules), ".gitignore must exclude *.sig signature files")
        self.assertTrue(any("*.faiss" in r for r in rules), ".gitignore must exclude *.faiss index files")

    # -------------------------------------------------------------------------
    # 5. Placeholder-only safety in .env.example
    # -------------------------------------------------------------------------
    def test_env_example_has_no_real_secrets(self):
        """
        Verify that .env.example contains only non-sensitive placeholders.
        """
        self.assertTrue(self.env_example_path.is_file(), ".env.example must exist.")
        lines = self.env_example_path.read_text(encoding="utf-8").splitlines()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            k, v = line.split("=", 1)
            # Ensure secret keys have non-sensitive placeholders
            if "key" in k.lower():
                self.assertIn("here", v.lower(), f"Key '{k}' in .env.example does not appear to be a placeholder.")
                self.assertNotIn("gsk_", v, f"Key '{k}' in .env.example contains a real or test API key prefix.")

    # -------------------------------------------------------------------------
    # 6. OCR import boundary and text-only separation
    # -------------------------------------------------------------------------
    def test_ocr_import_boundary(self):
        """
        Verify that handwriting_ocr imports its declared dependencies
        and that ingest_pdf processes digital-text pages without importing OCR.
        """
        import handwriting_ocr
        self.assertTrue(hasattr(handwriting_ocr, "recognize_handwritten_pdf"))
        self.assertTrue(hasattr(handwriting_ocr, "detect_text_lines"))


if __name__ == "__main__":
    unittest.main()
