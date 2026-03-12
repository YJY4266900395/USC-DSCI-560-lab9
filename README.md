# DSCI 560 Lab 9  
## Local PDF Question Answering Chatbot (RAG System)

This project implements a **local Retrieval-Augmented Generation (RAG) chatbot** that answers questions based on the content of a PDF document. The system extracts text from a PDF file, splits the text into chunks, stores embeddings in a FAISS vector database, and retrieves relevant information to generate answers using a local language model.

The chatbot runs **entirely locally without external APIs**.

---

# System Architecture

The system follows a typical **Retrieval-Augmented Generation pipeline**:

PDF Document
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Text Embedding
      ↓
Vector Database (FAISS)
      ↓
User Question
      ↓
Similarity Search
      ↓
Context Retrieval
      ↓
Local LLM (FLAN-T5)
      ↓
Generated Answer
---

# Project Structure

