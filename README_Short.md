# GPT Assistant 

## Overview of Approach
This project implements a **Retrieval-Augmented Generation (RAG)** system using **Gemini 2.5 Flash** for intelligent document-based question answering.  
Users upload documents (PDF, DOCX, TXT, CSV), which are:
1. **Parsed and preprocessed** to extract text using specialized loaders.  
2. **Chunked and embedded** using `all-mpnet-base-v2` sentence transformer embeddings.  
3. **Stored in a FAISS vector index** for semantic retrieval.  

When a user asks a question, the top-k most relevant document chunks are retrieved and passed to **Gemini 2.5 Flash**, which generates an answer **grounded only in the provided documents**.  
If relevant context is missing, the system explicitly responds with:  
> “I don’t have enough information in the uploaded documents.”

---

## Libraries Used
| Purpose | Library |
|----------|----------|
| Front-end UI | Streamlit |
| LLM Integration | Google Generative AI (Gemini 2.5 Flash) |
| Framework | LangChain |
| Embeddings | HuggingFace / sentence-transformers (all-mpnet-base-v2) |
| Vector Database | FAISS |
| Document Parsing | PyPDF2, pdfminer.six, python-docx, pandas |
| Configuration | python-dotenv |

---

## Example Input & Output
**Input:**  
- Uploaded file: `Anurag_Singh_Resume.pdf`  
- User query: “What programming languages does Anurag know?”

**Output:**  
> “Anurag Singh is skilled in Python, SQL, and has experience with machine learning frameworks such as TensorFlow and PyTorch.”

**Input (out-of-scope):**  
- Query: “Who is the Prime Minister of India?”

**Output:**  
> “I don’t have enough information in the uploaded documents.”

---

##  Handling Out-of-Scope Queries
The system uses a **custom prompt** within the LangChain RetrievalQA pipeline.  
If Gemini detects that the retrieved context lacks relevant information, it is instructed to respond with:  
> “I don’t have enough information in the uploaded documents.”  

This ensures factual accuracy, prevents hallucination, and keeps all responses strictly based on user-provided documents.

## This is the demo link to the deployed model [GPT Assistant](https://aiassignment-moqxrgmivs6osgizhstcpm.streamlit.app/)

