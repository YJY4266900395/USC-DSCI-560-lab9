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

# Running Instruction

First install dependencies and download correct model:

```bash
pip install -r requirements.txt
# adjust version of llama according to computer settings
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu129

mkdir models
curl -L -o models/llama-2-7b-chat.Q4_K_M.gguf "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf"
```

Or download the model to /models from [this link](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf) manually.

Then running the corresponding version:
```bash
streamlit run app_open_source.py
```
Same command for other .py files. Browser window will be activated automatically.

For OpenAI version: create a `.env` file in the same path of .py file and set `OPENAI_API_KEY=xxxxxxxxxxxxxx` (put the real api key here).

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

```
## 2. Text Chunking

The extracted text is split into smaller chunks using:

RecursiveCharacterTextSplitter

File:

text_chunks.py
Parameters used:

chunk size: 500 characters

overlap: 50 characters

Chunking allows the retrieval system to locate relevant sections more accurately.

Example:
``` python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
```

## 3. Vector Database

Each chunk is converted into an embedding using the model:

all-MiniLM-L6-v2

File:

vector_store.py

The embeddings are stored in a FAISS vector index for fast similarity search.

Generated files:

faiss_index/docs.index
faiss_index/chunks.pkl

Example:
``` python
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(chunks)

index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
```
## 4. Retrieval

When the user asks a question:

The question is converted into an embedding

FAISS finds the most relevant chunk

The retrieved chunk is used as context for the LLM

Example retrieval:
``` python
retrieve_top_k(question, index, chunks, embed_model, k=1)
```
## 5. Local Language Model

The system uses a local HuggingFace model:

google/flan-t5-small

File:

chatbot.py

The model receives:

retrieved context

user question

It then generates a summarized answer.

Example:
``` python
pipeline(
    "text2text-generation",
    model="google/flan-t5-small",
    max_new_tokens=128
)
```
## 6. Chat Interface

The chatbot runs in a command line interface (CLI).

File:

main.py

Workflow:

Load PDF

Create or load vector index

Load local LLM

Start interactive question loop

Example interaction:
``` python
Chatbot is ready. Type 'exit' to quit.
```
Ask a question: What is Fast Cosim?

Answer:
Fast Cosim is a feature in the Envelope controller that speeds up RF-DSP co-simulation by characterizing the RF subsystem once and reusing it during simulation.

Users can exit with:
```
exit
```
