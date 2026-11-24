"""DocQuery – Conversational PDF Research Agent (Streamlit UI)."""

import os

import streamlit as st
from dotenv import load_dotenv

from utils.agent import run_docquery_agent
from utils.memory import (
    add_message,
    clear_chat_history,
    get_display_messages,
    get_recent_history_messages,
    init_memory,
)
from utils.pdf_processor import process_pdf
from utils.vector_store import (
    compute_document_id,
    create_vector_store,
    get_embedding_model,
    load_vector_store,
)


load_dotenv(override=True)

st.set_page_config(
    page_title="DocQuery – PDF Research Agent",
    page_icon="📄",
    layout="centered",
)

# Minimal light-theme spacing & status box styling
st.markdown(
    """
    <style>
        .doc-status-box {
            padding: 0.75rem 0.9rem;
            border-radius: 0.5rem;
            background-color: #f1f5f9;
            border: 1px solid #cbd5e1;
            color: #0f172a;
            font-size: 0.9rem;
            margin-top: 0.5rem;
            margin-bottom: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def init_cached_embeddings() -> None:
    """Load the local sentence-transformers/all-MiniLM-L6-v2 model once."""
    get_embedding_model()


def init_session_state() -> None:
    """Initialize required Streamlit session state keys."""
    init_memory()
    defaults = {
        "active_doc_id": None,
        "active_filename": None,
        "page_count": 0,
        "chunk_count": 0,
        "doc_status": "Waiting for PDF upload",
        "doc_error": None,
        "vector_store": None,
        "uploader_key": 0,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def reset_active_document() -> None:
    """Clear the active PDF state and chat history to upload a different PDF."""
    st.session_state["active_doc_id"] = None
    st.session_state["active_filename"] = None
    st.session_state["page_count"] = 0
    st.session_state["chunk_count"] = 0
    st.session_state["doc_status"] = "Waiting for PDF upload"
    st.session_state["doc_error"] = None
    st.session_state["vector_store"] = None
    st.session_state["uploader_key"] += 1
    clear_chat_history()


def handle_pdf_upload(uploaded_file) -> None:
    """Process and index a newly uploaded PDF once (never per chat turn)."""
    file_bytes = uploaded_file.getvalue()
    new_doc_id = compute_document_id(file_bytes, uploaded_file.name)

    # Reuse existing vector store if the same PDF is still active
    if new_doc_id == st.session_state["active_doc_id"]:
        if st.session_state["vector_store"] is None:
            st.session_state["vector_store"] = load_vector_store()
        return

    # Replacing the active document clears the previous conversation context
    clear_chat_history()
    st.session_state["doc_error"] = None

    with st.spinner(f"Processing '{uploaded_file.name}' and generating embeddings..."):
        try:
            chunks, num_pages = process_pdf(file_bytes, uploaded_file.name)
            vector_store = create_vector_store(chunks, doc_id=new_doc_id)

            st.session_state["active_doc_id"] = new_doc_id
            st.session_state["active_filename"] = uploaded_file.name
            st.session_state["page_count"] = num_pages
            st.session_state["chunk_count"] = len(chunks)
            st.session_state["vector_store"] = vector_store
            st.session_state["doc_status"] = (
                f"Ready ({num_pages} pages, {len(chunks)} chunks)"
            )
        except Exception as exc:
            st.session_state["active_doc_id"] = None
            st.session_state["active_filename"] = uploaded_file.name
            st.session_state["page_count"] = 0
            st.session_state["chunk_count"] = 0
            st.session_state["vector_store"] = None
            st.session_state["doc_status"] = "Processing failed"
            st.session_state["doc_error"] = str(exc)


def render_assistant_extras(message: dict) -> None:
    """Render expandable source page references, agent activity, and verification."""
    sources = message.get("sources") or []
    activity = message.get("activity") or []
    verification = message.get("verification") or {}

    if verification:
        status = verification.get("status", "")
        reason = verification.get("reason", "")
        if status:
            st.caption(f"Answer Verification: {status}" + (f" — {reason}" if reason else ""))

    if sources:
        unique_pages = sorted({src["page"] for src in sources})
        page_label = ", ".join(f"Page {p}" for p in unique_pages)
        with st.expander(f"Source Page References ({page_label})"):
            for idx, src in enumerate(sources, start=1):
                st.markdown(f"**Reference {idx} — Page {src['page']}**")
                st.text(src["snippet"])

    if activity:
        with st.expander("Agent Activity"):
            for step in activity:
                tool_name = step.get("tool", "unknown")
                action_input = step.get("action_input", {})
                observation = step.get("observation", "")
                st.markdown(f"**Step {step.get('step', 1)} — Tool Selected:** `{tool_name}`")
                st.markdown(f"**Action Input:** `{action_input}`")
                st.text_area(
                    label="Retrieved Observation",
                    value=observation,
                    height=120,
                    disabled=True,
                    key=f"obs_{id(message)}_{step.get('step', 1)}_{tool_name}",
                )


def main() -> None:
    """Main Streamlit application entry point."""
    init_session_state()
    init_cached_embeddings()

    # Sidebar
    with st.sidebar:
        st.title("DocQuery")
        st.caption("Conversational PDF Research Agent")
        st.divider()

        uploaded_file = st.file_uploader(
            "Upload a PDF",
            type=["pdf"],
            accept_multiple_files=False,
            key=f"pdf_uploader_{st.session_state['uploader_key']}",
        )

        if uploaded_file is not None:
            handle_pdf_upload(uploaded_file)

        active_name = st.session_state.get("active_filename") or "None"
        doc_status = st.session_state.get("doc_status", "Waiting for PDF upload")

        st.markdown(
            f"""
            <div class="doc-status-box">
                <strong>Uploaded PDF:</strong> {active_name}<br>
                <strong>Status:</strong> {doc_status}
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.session_state.get("doc_error"):
            st.error(st.session_state["doc_error"])

        if st.button("Clear Chat", use_container_width=True):
            clear_chat_history()
            st.rerun()
        if st.button("Upload Another PDF", use_container_width=True):
            reset_active_document()
            st.rerun()

    # Main Interface
    st.title("DocQuery")
    st.markdown(
        "Welcome to **DocQuery**! Upload a PDF in the sidebar to ask questions, "
        "summarize topics, compare concepts, and ask follow-up questions."
    )

    if not os.getenv("GROQ_API_KEY"):
        st.warning(
            "GROQ_API_KEY is not set. Create a `.env` file from `.env.example` "
            "and add your Groq API key to start chatting."
        )

    # Display chat history
    for message in get_display_messages():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_assistant_extras(message)

    # Chat input
    pdf_ready = st.session_state["vector_store"] is not None
    prompt_placeholder = (
        "Ask a question, request a summary, or compare topics from the PDF..."
        if pdf_ready
        else "Upload a valid PDF in the sidebar to begin..."
    )

    user_question = st.chat_input(prompt_placeholder, disabled=not pdf_ready)

    if user_question and pdf_ready:
        # Capture last 6 exchanges BEFORE appending the new question
        recent_history = get_recent_history_messages()

        add_message("user", user_question)
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking and searching the PDF..."):
                try:
                    response = run_docquery_agent(
                        question=user_question,
                        vector_store=st.session_state["vector_store"],
                        doc_id=st.session_state["active_doc_id"],
                        chat_history=recent_history,
                    )
                    answer = response["answer"]
                    sources = response["sources"]
                    activity = response["activity"]
                    verification = response["verification"]
                except Exception as exc:
                    answer = f"An error occurred while generating the answer: {exc}"
                    sources = []
                    activity = []
                    verification = {}

            st.markdown(answer)
            assistant_entry = {
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "activity": activity,
                "verification": verification,
            }
            render_assistant_extras(assistant_entry)

        add_message(
            role="assistant",
            content=answer,
            sources=sources,
            activity=activity,
            verification=verification,
        )


if __name__ == "__main__":
    main()
