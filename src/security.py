import hashlib
import hmac
import json
import os
from pathlib import Path


SIGNATURE_FILENAME = "index.sig"


class IndexSecurityError(ValueError):
    """Raised when FAISS index signature verification fails or signing key is missing."""
    pass


def load_env_file(dotenv_path: str | Path | None = None) -> None:
    """
    Load environment variables from a .env file into os.environ if not already set.
    Uses python-dotenv if installed, with a lightweight line-parsing fallback.
    Never overwrites existing environment variables.
    """
    if dotenv_path is None:
        base_dir = Path(__file__).resolve().parent.parent
        target_path = base_dir / ".env"
    else:
        target_path = Path(dotenv_path)

    if not target_path.is_file():
        return

    try:
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=target_path, override=False)
        return
    except Exception:
        pass

    try:
        for line in target_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip("'\"")
            if k and k not in os.environ:
                os.environ[k] = v
    except Exception:
        pass


def get_signing_key() -> str:
    """
    Retrieve the index HMAC signing key from the environment, a local .env file,
    or Streamlit secrets.
    Fails closed if the key is not configured or is empty.
    """
    key = os.environ.get("STUDYMATE_INDEX_SIGNING_KEY")

    if not key:
        load_env_file()
        key = os.environ.get("STUDYMATE_INDEX_SIGNING_KEY")

    if not key:
        try:
            import streamlit as st
            key = st.secrets.get("STUDYMATE_INDEX_SIGNING_KEY")
        except Exception:
            key = None

    if not key or not str(key).strip():
        raise IndexSecurityError(
            "FAISS index signing key is not configured. Set the "
            "STUDYMATE_INDEX_SIGNING_KEY environment variable, define it in a .env file, "
            "or define it in .streamlit/secrets.toml to securely sign and verify vector indexes."
        )

    return str(key).strip()


def compute_file_hmac(filepath: str | Path, key: str | bytes) -> str:
    """Compute HMAC-SHA256 hex digest for the exact bytes of a file."""
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot compute HMAC: file not found: {filepath}")

    key_bytes = key if isinstance(key, bytes) else key.encode("utf-8")
    data = path.read_bytes()
    return hmac.new(key_bytes, data, hashlib.sha256).hexdigest()


def sign_vectorstore_index(
    directory: str | Path,
    key: str | bytes | None = None,
) -> Path:
    """
    Compute and write an HMAC-SHA256 signature sidecar for index.pkl and index.faiss.
    
    The signature is written to index.sig in the index directory before the index
    is deemed ready for retrieval.
    """
    dir_path = Path(directory)
    pkl_path = dir_path / "index.pkl"
    faiss_path = dir_path / "index.faiss"

    if not pkl_path.exists():
        raise FileNotFoundError(f"Missing index.pkl in '{directory}' for signing.")
    if not faiss_path.exists():
        raise FileNotFoundError(f"Missing index.faiss in '{directory}' for signing.")

    signing_key = key if key is not None else get_signing_key()

    pkl_hmac = compute_file_hmac(pkl_path, signing_key)
    faiss_hmac = compute_file_hmac(faiss_path, signing_key)

    sig_data = {
        "version": 1,
        "algorithm": "HMAC-SHA256",
        "pkl_hmac": pkl_hmac,
        "faiss_hmac": faiss_hmac,
    }

    sig_path = dir_path / SIGNATURE_FILENAME
    sig_path.write_text(json.dumps(sig_data, indent=2), encoding="utf-8")
    return sig_path


def verify_vectorstore_index(
    directory: str | Path,
    key: str | bytes | None = None,
) -> None:
    """
    Verify the integrity and authenticity of index.pkl and index.faiss
    before FAISS.load_local() is called.
    
    Raises:
        FileNotFoundError: If the index directory or index files are missing.
        IndexSecurityError: If the signature is missing, invalid, or does not match.
    """
    dir_path = Path(directory)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"Index directory does not exist: '{directory}'.")

    pkl_path = dir_path / "index.pkl"
    faiss_path = dir_path / "index.faiss"
    sig_path = dir_path / SIGNATURE_FILENAME

    if not pkl_path.exists():
        raise FileNotFoundError(f"Index pickle file missing: '{pkl_path}'.")
    if not faiss_path.exists():
        raise FileNotFoundError(f"Index binary file missing: '{faiss_path}'.")

    if not sig_path.exists():
        raise IndexSecurityError(
            f"Missing integrity signature ({SIGNATURE_FILENAME}) in '{directory}'. "
            f"Refusing to deserialize unsigned or untrusted FAISS pickle file. "
            f"Configure STUDYMATE_INDEX_SIGNING_KEY and rebuild the index from the original PDF."
        )

    signing_key = key if key is not None else get_signing_key()

    try:
        sig_data = json.loads(sig_path.read_text(encoding="utf-8"))
        stored_pkl_hmac = sig_data.get("pkl_hmac")
        stored_faiss_hmac = sig_data.get("faiss_hmac")
    except Exception as exc:
        raise IndexSecurityError(
            f"Corrupted or unreadable signature file in '{directory}': {exc}. "
            f"Refusing to load untrusted index."
        ) from exc

    if not stored_pkl_hmac or not stored_faiss_hmac:
        raise IndexSecurityError(
            f"Incomplete signature payload in '{sig_path}'. Refusing to load untrusted index."
        )

    # 1. Verify index.pkl bytes
    actual_pkl_hmac = compute_file_hmac(pkl_path, signing_key)
    if not hmac.compare_digest(stored_pkl_hmac, actual_pkl_hmac):
        raise IndexSecurityError(
            f"Integrity check failed: index.pkl in '{directory}' has been modified, tampered with, "
            f"or was signed with a different key. Refusing to deserialize."
        )

    # 2. Verify index.faiss bytes
    actual_faiss_hmac = compute_file_hmac(faiss_path, signing_key)
    if not hmac.compare_digest(stored_faiss_hmac, actual_faiss_hmac):
        raise IndexSecurityError(
            f"Integrity check failed: index.faiss in '{directory}' has been modified, tampered with, "
            f"or was signed with a different key. Refusing to load."
        )
