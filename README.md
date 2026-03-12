# DSCI 560 Lab 9  
## Local PDF Question Answering Chatbot (RAG System)

This project implements a **local Retrieval-Augmented Generation (RAG) chatbot** that answers questions based on the content of a PDF document. The system extracts text from a PDF file, splits the text into chunks, stores embeddings in a FAISS vector database, and retrieves relevant information to generate answers using a local language model.

The chatbot runs **entirely locally without external APIs**.

---

# System Architecture

The system follows a typical **Retrieval-Augmented Generation pipeline**:



---

# Project Structure


---

# Components

## 1. PDF Text Extraction

The PDF document is loaded and converted into raw text using **PyPDFLoader**.

File:pdf_extraction.py

This script reads each page of the PDF and concatenates the content into a single text string.

Example:

```python
loader = PyPDFLoader(pdf_path)
pages = loader.load()

for page in pages:
    text += page.page_content


## 2. Text Chunking

The extracted text is split into smaller chunks using:
