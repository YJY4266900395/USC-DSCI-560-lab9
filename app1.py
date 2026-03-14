import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.messages import HumanMessage, AIMessage

from html_modified import css, bot_template, user_template

TOP_K = 10


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.warning(f"Failed to read one PDF: {e}")
    return text


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    vectorstore = FAISS.from_texts(
        texts=text_chunks,
        embedding=embeddings
    )

    return vectorstore


def get_conversation_chain(vectorstore):

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    def ask_llm(question, docs):

        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are answering questions about a PDF document.

Use ONLY the provided context.
Do NOT use outside knowledge.
Do NOT guess.

If the answer is not in the context say:
"I cannot find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

        response = llm.invoke(prompt)

        return response.content

    def conversation_chain(question):

        docs = vectorstore.similarity_search(question, k=TOP_K)

        answer = ask_llm(question, docs)

        return answer

    return conversation_chain


def handle_userinput(user_question):

    if st.session_state.conversation is None:
        st.warning("Please upload and process PDFs first.")
        return

    answer = st.session_state.conversation(user_question)

    st.session_state.chat_history.append(user_question)
    st.session_state.chat_history.append(answer)

    for i, msg in enumerate(st.session_state.chat_history):

        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", msg), unsafe_allow_html=True)

        else:
            st.write(bot_template.replace(
                "{{MSG}}", msg), unsafe_allow_html=True)


def main():

    load_dotenv()

    st.set_page_config(
        page_title="Chat with PDFs",
        page_icon=":robot_face:"
    )

    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with PDFs :robot_face:")

    user_question = st.text_input("Ask questions about your documents:")

    if user_question:
        handle_userinput(user_question)

    with st.sidebar:

        st.subheader("Your documents")

        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on Process",
            accept_multiple_files=True
        )

        if st.button("Process"):

            with st.spinner("Processing"):

                raw_text = get_pdf_text(pdf_docs)

                text_chunks = get_text_chunks(raw_text)

                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(
                    vectorstore
                )

                st.success("Processing complete.")


if __name__ == "__main__":
    main()