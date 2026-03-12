from transformers import pipeline


def create_local_llm():
    generator = pipeline(
        "text2text-generation",
        model="google/flan-t5-small",
        max_new_tokens=128
    )
    return generator


def generate_answer(question, retrieved_chunks, llm, chat_history=None):
    if chat_history is None:
        chat_history = []

    context = "\n\n".join(retrieved_chunks)[:1000]
    history_text = ""
    for q, a in chat_history[-3:]:
        history_text += f"User: {q}\nAssistant: {a}\n"

    prompt = f"""
    You are a helpful Q&A assistant for a PDF document.

    Rules:
    1. Answer only using the provided context.
    2. If the answer is not in the context, say:
       "I could not find the answer in the PDF."
    3. Answer in 3-5 short sentences.
    4. Do not repeat text.
    5. Summarize the answer clearly.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    result = llm(prompt)
    return result[0]["generated_text"].strip()