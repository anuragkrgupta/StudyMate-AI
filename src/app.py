
import os
import hashlib
import streamlit as st
from langchain_groq import ChatGroq

from security import load_env_file

# Load local .env file if present
load_env_file()

# ---------------------------------
# Page Configuration
# ---------------------------------

st.set_page_config(
    page_title="StudyMate AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------
# Existing LLM Configuration
# ---------------------------------

api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

model = st.secrets.get(
    "GROQ_MODEL",
    os.getenv("GROQ_MODEL", "openai/gpt-oss-120b"),
)

llm = ChatGroq(
    api_key=api_key,
    model=model,
    temperature=0,
)

# ---------------------------------
# UI Styling Only
# ---------------------------------

st.markdown(
    """
    <style>
    #MainMenu, footer {
        visibility: hidden;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    [data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.2);
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    .brand {
        font-size: 1.65rem;
        font-weight: 750;
        letter-spacing: -0.8px;
        margin-bottom: 0;
    }

    .brand-accent {
        color: #7C83FD;
    }

    .muted {
        opacity: 0.72;
        font-size: 0.9rem;
    }

    .hero-title {
        font-size: clamp(2rem, 4vw, 2.8rem);
        font-weight: 750;
        letter-spacing: -1.3px;
        line-height: 1.15;
        margin-bottom: 0.5rem;
    }

    .hero-subtitle {
        font-size: 1rem;
        opacity: 0.75;
        margin-bottom: 1.2rem;
    }

    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 1.1px;
        text-transform: uppercase;
        opacity: 0.7;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] {
        border-radius: 16px;
    }

    div[data-testid="stChatMessage"] {
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.7rem;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        min-height: 2.7rem;
        transition: border-color 0.15s ease;
    }

    .stButton > button:hover {
        border-color: #7C83FD;
    }

    div[data-testid="stChatInput"] {
        border-radius: 14px;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero-title {
            letter-spacing: -0.7px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------
# Session State — Existing Keys
# ---------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processed_file_id" not in st.session_state:
    st.session_state.processed_file_id = None

if "document_ready" not in st.session_state:
    st.session_state.document_ready = False

if "current_doc_name" not in st.session_state:
    st.session_state.current_doc_name = None

if "summary" not in st.session_state:
    st.session_state.summary = None

if "summary_file_id" not in st.session_state:
    st.session_state.summary_file_id = None


def _file_id(name, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()[:16]
    return f"{name}:{digest}"


def generate_document_summary(llm_client, context: str) -> str:
    """Generate a structured document summary using the configured LLM."""
    prompt = f"""
Analyze this document and create a structured summary.

If it is lecture notes:
- Main topic
- Key concepts
- Important formulas

If it is a notice:
- Purpose
- Important dates
- Key instructions

If it is a report:
- Objective
- Findings
- Conclusions

Document:
{context}
"""
    response = llm_client.invoke(prompt)
    content = response.content if hasattr(response, "content") else str(response)
    if not content or not content.strip():
        raise ValueError("The model returned an empty response.")
    return content.strip()


# ---------------------------------
# Sidebar
# ---------------------------------

with st.sidebar:
    st.markdown(
        """
        <div class="brand">📚 StudyMate <span class="brand-accent">AI</span></div>
        <p class="muted">Your personal study assistant</p>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    st.markdown("#### Workspace")
    st.caption("Upload your notes, ask questions, and review key concepts.")

    if st.session_state.document_ready and st.session_state.current_doc_name:
        st.success("Document ready")
        st.markdown(f"**📄 {st.session_state.current_doc_name}**")
    else:
        st.info("No document loaded")
        st.caption("Upload a PDF to get started.")

    st.divider()

    st.markdown("#### What you can do")

    st.markdown(
        """
        - 💬 Ask questions about your PDF
        - 📝 Generate structured summaries
        - 🔎 Inspect answer sources
        - ✍️ Process scanned handwritten notes
        """
    )

    st.divider()

    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by Groq · LangChain · FAISS")
    st.caption("Handwriting OCR runs locally when needed.")


# ---------------------------------
# Main Header
# ---------------------------------

st.markdown(
    '<div class="hero-title">Learn smarter. <span class="brand-accent">Study better.</span></div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="hero-subtitle">Your documents, simplified. Upload your study material and let AI help you understand it.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------
# Document Upload
# ---------------------------------

st.markdown('<div class="section-label">YOUR DOCUMENT</div>', unsafe_allow_html=True)
st.write("")

with st.container(border=True):
    upload_col, status_col = st.columns([1.7, 1], gap="large")

    with upload_col:
        st.markdown("### 📄 Upload study material")
        st.caption("Choose a PDF containing lecture notes, textbooks, reports, or scanned handwritten notes.")

        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="The PDF is processed by your existing local ingestion pipeline.",
            label_visibility="collapsed",
        )

    with status_col:
        st.markdown("### ✨ Ready to study?")

        if st.session_state.document_ready and st.session_state.current_doc_name:
            st.success("Your document is indexed and ready.")
            st.caption(f"Active file: {st.session_state.current_doc_name}")
        else:
            st.markdown(
                """
                **Start with a PDF**

                Upload a document to unlock document chat, summaries, and source references.
                """
            )

        st.caption("Scanned handwriting is handled by your existing local OCR fallback when applicable.")


if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    file_id = _file_id(uploaded_file.name, pdf_bytes)

    if file_id != st.session_state.processed_file_id:
        with st.spinner("🔄 Reading and indexing your PDF..."):
            try:
                from ingest import ingest_pdf
                from retriever import reload

                # Document-isolated ingestion and vector-store workflow.
                chunk_count = ingest_pdf(
                    pdf_bytes,
                    doc_id=file_id,
                    source_name=uploaded_file.name,
                )
                reload(doc_id=file_id)

                st.session_state.processed_file_id = file_id
                st.session_state.document_ready = True
                st.session_state.current_doc_name = uploaded_file.name
                st.session_state.messages = []
                st.session_state.summary = None
                st.session_state.summary_file_id = None

            except Exception as exc:
                st.session_state.document_ready = False
                st.session_state.summary = None
                st.session_state.summary_file_id = None
                st.error(f"❌ Failed to process PDF: {exc}")
                st.stop()

        st.success(
            f"**{uploaded_file.name}** is ready — "
            f"{chunk_count} sections indexed. You can ask questions now."
        )

    elif st.session_state.document_ready:
        st.success(f"📄 **{uploaded_file.name}** is loaded and ready.")


# ---------------------------------
# Study Tools
# ---------------------------------

st.write("")
st.markdown('<div class="section-label">STUDY TOOLS</div>', unsafe_allow_html=True)
st.write("")

summary_col, chat_col = st.columns([1, 1], gap="medium")

with summary_col:
    with st.container(border=True):
        st.markdown("### 📝 Document summary")
        st.caption("Turn lengthy notes into structured, easy-to-review points.")

        generate_summary = st.button(
            "Generate summary",
            use_container_width=True,
            type="primary",
        )

with chat_col:
    with st.container(border=True):
        st.markdown("### 💬 Ask your document")
        st.caption("Ask questions and get answers grounded in your uploaded material.")

        if st.session_state.document_ready:
            st.success("Question answering is available.")
        else:
            st.caption("Upload a PDF to begin asking questions.")


# ---------------------------------
# PDF Summary — Existing Logic
# ---------------------------------

if generate_summary:
    if not st.session_state.document_ready:
        st.warning("Please upload a PDF first.")
    else:
        from retriever import get_all_chunks

        try:
            all_chunks = get_all_chunks(doc_id=st.session_state.processed_file_id)
            context = "\n\n".join(all_chunks)[:12000]

            with st.spinner("Generating summary..."):
                summary_text = generate_document_summary(llm, context)

            st.session_state.summary = summary_text
            st.session_state.summary_file_id = st.session_state.processed_file_id

        except Exception as exc:
            st.session_state.summary = None
            st.session_state.summary_file_id = None
            st.error(f"Failed to generate summary: {exc}")

if (
    st.session_state.document_ready
    and st.session_state.summary
    and st.session_state.summary_file_id == st.session_state.processed_file_id
):
    st.markdown("### 📄 Document Summary")

    with st.container(border=True):
        st.markdown(st.session_state.summary)


# ---------------------------------
# Chat Workspace
# ---------------------------------

st.write("")
st.markdown('<div class="section-label">DOCUMENT CONVERSATION</div>', unsafe_allow_html=True)
st.write("")

if not st.session_state.document_ready:
    with st.container(border=True):
        st.markdown("### 👋 Welcome to StudyMate AI")
        st.write("Your study session starts with a document.")

        st.markdown(
            """
            **Try this workflow:**

            1. Upload a PDF above.
            2. Generate a summary of the important concepts.
            3. Ask questions and inspect the sources behind each answer.
            """
        )
else:
    st.caption(f"Chatting with: {st.session_state.current_doc_name}")

def _format_source_label(source_name: str | None, page_num: int | None) -> str:
    """Format human-readable source attribution string for a retrieved chunk."""
    if source_name and page_num is not None:
        return f"{source_name} (Page {page_num})"
    if source_name:
        return f"{source_name}"
    if page_num is not None:
        return f"Page {page_num}"
    return "Unknown source"


def _render_source_attribution(sources: list[dict]):
    """Render deduplicated source labels and detailed expandable chunk sources."""
    if not sources:
        return

    # Deduplicate source/page labels in original retrieval order
    seen = set()
    unique_labels = []
    for s in sources:
        src = s.get("source")
        pg = s.get("page")
        key = (src, pg)
        if key not in seen:
            seen.add(key)
            label = _format_source_label(src, pg)
            if label != "Unknown source" or not unique_labels:
                unique_labels.append(label)

    if unique_labels:
        st.caption("📍 **Sources:** " + ", ".join(unique_labels))

    with st.expander("📄 View Sources"):
        for i, s in enumerate(sources, start=1):
            src = s.get("source")
            pg = s.get("page")
            header = f"### Source {i}"
            if src and pg is not None:
                header += f" — {src}, Page {pg}"
            elif src:
                header += f" — {src}"
            elif pg is not None:
                header += f" — Page {pg}"
            st.markdown(header)
            st.code(s.get("text", ""))


def build_chat_prompt(
    question: str,
    context: str,
    history_messages: list[dict] | None = None,
) -> str:
    """
    Construct the chat prompt for the LLM.

    Includes prior conversation history (excluding the current question),
    retrieved document context, and the current question exactly once.
    """
    if history_messages:
        chat_history = "\n".join(
            [
                f"{msg['role']}: {msg['content']}"
                for msg in history_messages[-6:]
                if msg.get("content")
            ]
        )
    else:
        chat_history = ""

    return f"""
You are a helpful study assistant.

Use the conversation history and document context
to answer the user's question.

Conversation History:
{chat_history}

Document Context:
{context}

Current Question:
{question}

If the answer is not present in the document context,
say:

I could not find that information in the document.
"""


# ---------------------------------
# Display Chat History — Existing Logic
# ---------------------------------

for message in st.session_state.messages:
    with st.chat_message(
        message["role"],
        avatar="👤" if message["role"] == "user" else "🤖",
    ):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources"):
            _render_source_attribution(message["sources"])


# ---------------------------------
# Chat Input — Existing Logic
# ---------------------------------

question = st.chat_input("Ask a question about your PDF...")

if question:
    if not st.session_state.document_ready:
        st.warning("Please upload a PDF before asking questions.")
    else:
        from retriever import retrieve_with_metadata

        st.session_state.messages.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user", avatar="👤"):
            st.markdown(question)

        with st.spinner("Searching document..."):
            retrieved_docs = retrieve_with_metadata(
                question,
                doc_id=st.session_state.processed_file_id,
            )

        chunks = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc in retrieved_docs
        ]
        sources_data = [
            {
                "text": doc.page_content if hasattr(doc, "page_content") else str(doc),
                "source": (getattr(doc, "metadata", {}) or {}).get("source"),
                "page": (getattr(doc, "metadata", {}) or {}).get("page"),
            }
            for doc in retrieved_docs
        ]

        context = "\n\n".join(chunks)

        # Prior conversation turns excluding the current question that was just appended
        prior_messages = st.session_state.messages[:-1]

        prompt = build_chat_prompt(
            question=question,
            context=context,
            history_messages=prior_messages,
        )

        with st.spinner("Generating answer..."):
            response = llm.invoke(prompt)
            answer = response.content

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": sources_data,
            }
        )

        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(answer)
            if sources_data:
                _render_source_attribution(sources_data)
