import os
from pdf_extraction import extract_pdf
from text_chunks import create_chunks
from vector_store import build_vector_store, load_vector_store, retrieve_top_k
from chatbot import create_local_llm, generate_answer

PDF_PATH = "ads_cookbook.pdf"
INDEX_DIR = "faiss_index"

def main():
    print("Loading PDF...")
    text = extract_pdf(PDF_PATH)
    print("Text length:", len(text))

    if not os.path.exists(os.path.join(INDEX_DIR, "docs.index")):
        print("Creating chunks...")
        chunks = create_chunks(text)
        print("Total chunks:", len(chunks))

        print("Building vector store...")
        index, chunks, embed_model = build_vector_store(chunks, INDEX_DIR)
    else:
        print("Loading existing vector store...")
        index, chunks, embed_model = load_vector_store(INDEX_DIR)
        print("Loaded chunks:", len(chunks))

    print("Loading local LLM...")
    llm = create_local_llm()

    chat_history = []

    print("\nChatbot is ready. Type 'exit' to quit.\n")

    while True:
        question = input("Ask a question: ").strip()

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question:
            continue

        retrieved = retrieve_top_k(question, index, chunks, embed_model, k=1)

        answer = generate_answer(
            question=question,
            retrieved_chunks=retrieved,
            llm=llm,
            chat_history=chat_history
        )

        chat_history.append((question, answer))

        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    main()