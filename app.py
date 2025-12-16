import os
import streamlit as st
from dotenv import load_dotenv

from model import (
    process_uploaded_files,
    create_vectorstore,
    build_rag_chain
)

# -----------------------------
# App Configuration
# -----------------------------

st.set_page_config(
    page_title="Document Q&A (RAG with Gemini)",
    page_icon="📄",
    layout="wide"
)

load_dotenv()

# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("📂 Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF, DOCX, CSV, or TXT files",
    type=["pdf", "docx", "csv", "txt"],
    accept_multiple_files=True
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses a Retrieval-Augmented Generation (RAG) pipeline "
    "with Gemini and FAISS to answer questions from your documents."
)

# -----------------------------
# Cached Resources
# -----------------------------

@st.cache_resource(show_spinner="Processing documents and building index...")
def get_vectorstore(docs):
    return create_vectorstore(docs)


@st.cache_resource(show_spinner="Initializing RAG chain...")
def get_rag_chain(vectorstore):
    return build_rag_chain(vectorstore)


# -----------------------------
# Main UI
# -----------------------------

st.title("📄 Document Question Answering (RAG)")
st.write(
    "Upload documents on the left, then ask questions based strictly on their content."
)

# -----------------------------
# Document Processing
# -----------------------------

if uploaded_files:
    try:
        docs = process_uploaded_files(uploaded_files)

        if not docs:
            st.error("No readable text found in the uploaded documents.")
            st.stop()

        vectorstore = get_vectorstore(docs)
        qa_chain = get_rag_chain(vectorstore)

        st.success("Documents processed successfully! Ask your question below.")

        # Optional: Show extracted snippets
        with st.expander("🔍 View extracted document snippets"):
            for i, doc in enumerate(docs[:3], start=1):
                st.text_area(
                    label=f"Document Snippet {i}",
                    value=doc.page_content[:500] + "...",
                    height=150,
                    label_visibility="collapsed"
                )

    except Exception as e:
        st.error("An error occurred while processing documents.")
        st.exception(e)
        st.stop()

else:
    st.info("Please upload at least one document to begin.")
    st.stop()

# -----------------------------
# Question Answering
# -----------------------------

st.markdown("### ❓ Ask a Question")

question = st.text_input(
    "Enter your question:",
    placeholder="e.g. What skills are mentioned in the resume?",
    label_visibility="collapsed"
)

if question:
    with st.spinner("Generating answer..."):
        try:
            result = qa_chain.invoke({"input": question})

            answer = result.get("answer", "")
            context_docs = result.get("context", [])

            st.markdown("### ✅ Answer")
            st.write(answer)

            if context_docs:
                with st.expander("📌 Source Context"):
                    for i, doc in enumerate(context_docs, start=1):
                        st.markdown(f"**Source {i}:**")
                        st.text_area(
                            label=f"Context {i}",
                            value=doc.page_content[:400] + "...",
                            height=120,
                            label_visibility="collapsed"
                        )

        except Exception as e:
            st.error("An error occurred while generating the answer.")
            st.exception(e)
