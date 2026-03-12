### DSCI 560 Lab 9 – Local PDF Question Answering Chatbot

This project implements a local PDF-based question answering system using open-source tools.
The chatbot allows users to ask questions about the content of a PDF document, and it retrieves relevant sections of the document to generate answers.

The system uses:

LangChain for PDF loading and text processing

SentenceTransformers for text embeddings

FAISS for vector similarity search

FLAN-T5 as the local language model

Python CLI interface for interaction

The chatbot works completely locally without external APIs.
