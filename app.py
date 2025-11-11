# app.py
import os
from dotenv import load_dotenv
import streamlit as st
import google.generativeai as genai
from model import process_uploaded_files, create_vectorstore, build_rag_chain



# Environment Setup

load_dotenv()  # Load .env file (contains GOOGLE_API_KEY)
st.set_page_config(page_title="GPT Assistant (RAG-Style System", layout="wide")
st.title("GPT Assistant (RAG-Style System")

# Configure Gemini API Key
if "GOOGLE_API_KEY" not in os.environ:
    st.warning("GOOGLE_API_KEY not found in environment. Please set it in your .env file.")
else:
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



# Sidebar: Document Upload
st.sidebar.header("Upload Documents")
uploaded_files = st.sidebar.file_uploader(
    "Upload TXT, PDF, DOCX, or CSV files",
    type=["txt", "pdf", "docx", "csv"],
    accept_multiple_files=True,
)

# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None



# Cached Vectorstore Creation

@st.cache_resource
def get_vectorstore(docs):
    """Cache FAISS vectorstore to avoid recomputation."""
    return create_vectorstore(docs)


if uploaded_files:
    with st.spinner("Processing uploaded files..."):
        docs = process_uploaded_files(uploaded_files)

        if len(docs) == 0:
            st.sidebar.error("No text could be extracted. Try uploading another file.")
        else:
            st.sidebar.success(f"Processed {len(docs)} text chunks.")
            vectorstore = get_vectorstore(docs)
            st.session_state.vectorstore = vectorstore
            st.sidebar.success("Vector store created and cached successfully.")



# Main Section – Query Interface

st.header("Ask a question")
question = st.text_area("Enter your question related to the uploaded documents:", height=100)
top_k = st.slider("Top K results for retrieval", 1, 10, 5)

if st.button("Get Answer"):
    if not question.strip():
        st.warning("Please enter a question first.")
    elif st.session_state.vectorstore is None:
        st.error("Please upload and process documents before asking questions.")
    else:
        with st.spinner("Querying..."):
            qa_chain = build_rag_chain(st.session_state.vectorstore, k=top_k)
            result = qa_chain.invoke({"query": question})

        # Display answer
        st.subheader("Answer")
        st.write(result["result"])

        # Display retrieved sources
        st.subheader("Source Documents")
        for i, doc in enumerate(result["source_documents"], start=1):
            st.text_area(
                label=f"Source {i}: Extracted Document Snippet",
                value=doc.page_content[:500] + "...",
                height=150,
                label_visibility="collapsed"
            )


# Footer

st.markdown("---")
st.caption("Built withLangChain, Streamlit, and Gemini 2.5 Flash")
