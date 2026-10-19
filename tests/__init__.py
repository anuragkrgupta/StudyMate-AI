# Test suite package marker
import os

# Set a dedicated test-suite signing key for running automated regression tests
if "STUDYMATE_INDEX_SIGNING_KEY" not in os.environ:
    os.environ["STUDYMATE_INDEX_SIGNING_KEY"] = "test-signing-key-for-automated-regression-tests-only-32chars"
