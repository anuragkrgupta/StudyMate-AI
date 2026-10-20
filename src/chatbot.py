"""
StudyMate AI CLI Chatbot.

Provides a standalone command-line interface for asking questions against
the indexed StudyMate AI vector store using the configured Groq LLM model.
"""

import os
import sys

from langchain_groq import ChatGroq

from config import CHAT_MODEL
from retriever import retrieve
from security import load_env_file

# Load local .env configuration if available
load_env_file()


def get_chatbot_llm(model: str | None = None, api_key: str | None = None) -> ChatGroq:
    """
    Initialize a ChatGroq client using project configuration.
    
    Fails closed with a clear error message if GROQ_API_KEY is not configured.
    """
    resolved_model = model or os.getenv("GROQ_MODEL") or CHAT_MODEL
    resolved_key = api_key or os.getenv("GROQ_API_KEY")

    if not resolved_key or not resolved_key.strip():
        raise ValueError(
            "GROQ_API_KEY is not configured. Set the GROQ_API_KEY environment variable "
            "or define it in your .env file."
        )

    return ChatGroq(
        api_key=resolved_key.strip(),
        model=resolved_model,
        temperature=0,
    )


def ask_question(query: str, doc_id: str | None = None, llm: ChatGroq | None = None) -> str:
    """
    Retrieve context and answer a query using the configured LLM.
    
    Routes through retriever.retrieve(), which verifies index HMAC integrity
    prior to deserialization.
    """
    if not query or not query.strip():
        return ""

    chunks = retrieve(query.strip(), doc_id=doc_id)
    context = "\n\n".join(chunks)

    prompt = f"""You are a helpful study assistant.

Use the document context below to answer the user's question.

Document Context:
{context}

Current Question:
{query.strip()}

If the answer is not present in the document context, say:
I could not find that information in the document.
"""

    client = llm or get_chatbot_llm()
    response = client.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    return content.strip()


def run_cli(doc_id: str | None = None) -> None:
    """Run interactive command-line chatbot session."""
    try:
        llm = get_chatbot_llm()
    except Exception as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return

    target_msg = f" for document '{doc_id}'" if doc_id else ""
    print(f"StudyMate AI CLI Chatbot ({CHAT_MODEL})")
    print(f"Querying vector store{target_msg}. Type 'exit' to quit.\n")

    while True:
        try:
            query = input("Ask a question (or type exit): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() == "exit":
            break

        try:
            answer = ask_question(query, doc_id=doc_id, llm=llm)
            print("\nAnswer:\n")
            print(answer)
            print()
        except FileNotFoundError as fnf_err:
            print(f"\nVector store not found: {fnf_err}", file=sys.stderr)
            print("Tip: Ingest a document first using: python src/ingest.py <path-to-pdf>\n", file=sys.stderr)
        except Exception as exc:
            print(f"\nError processing question: {exc}\n", file=sys.stderr)


if __name__ == "__main__":
    cli_doc_id = sys.argv[1] if len(sys.argv) > 1 else None
    run_cli(doc_id=cli_doc_id)