
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


import re
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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
    try:
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        vectorstore = FAISS.from_texts(
            texts=text_chunks,
            embedding=embeddings
        )
        return vectorstore
    except Exception as e:
        st.error(f"Error while creating embeddings/vector store: {e}")
        return None


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are answering questions about a PDF document.\n"
            "Use ONLY the provided context.\n"
            "Do NOT use outside knowledge.\n"
            "Do NOT guess.\n"
            "If the answer is not in the context, say exactly: "
            "'I cannot find the answer in the provided document.'\n"
            "Keep answers short, clear, and faithful to the document.\n\n"
            "Context:\n{context}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def _is_intro_question(q):
    patterns = [
        r"introduce yourself",
        r"introduce the document",
        r"what (is|are) (this|the) (document|pdf|book)",
        r"what document",
        r"summarize (this|the) (document|pdf|book)",
        r"what('s| is) (this|the) (document|pdf|book)",
        r"tell me about (this|the) (document|pdf|book)",
        r"what did i upload",
        r"what is (uploaded|the uploaded)",
    ]
    return any(re.search(p, q.lower()) for p in patterns)


def build_history_messages(chat_history):
    history = []
    for i, msg in enumerate(chat_history):
        if i % 2 == 0:
            history.append(HumanMessage(content=msg))
        else:
            history.append(AIMessage(content=msg))
    return history


def clean_answer(answer):
    answer = answer.strip()
    answer = re.sub(r"\s+", " ", answer).strip()
    answer = re.sub(r"\s+([.,;:!?])", r"\1", answer)
    return answer


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDF documents first.")
        return

    user_question = user_question.strip()
    if not user_question:
        return

    if _is_intro_question(user_question) and st.session_state.get("doc_intro"):
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        intro_prompt = (
            "You are answering questions about a PDF document.\n"
            "Use ONLY the provided context.\n"
            "Do NOT use outside knowledge.\n"
            "Do NOT guess.\n"
            "If the document title or name is not explicitly stated in the context, do NOT invent one.\n"
            "Keep the answer concise and faithful to the document.\n\n"
            f"Context (first section of the document):\n{st.session_state.doc_intro}\n\n"
            f"Question:\n{user_question}\n\nAnswer:"
        )
        try:
            response = llm.invoke([HumanMessage(content=intro_prompt)]).content
            response = clean_answer(response)
        except Exception as e:
            st.error(f"Error while generating answer: {e}")
            return

        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response)
        return

    history = build_history_messages(st.session_state.chat_history)

    try:
        response = st.session_state.conversation.invoke({
            "question": user_question,
            "chat_history": history,
        })
        response = clean_answer(response)
    except Exception as e:
        st.error(f"Error while generating answer: {e}")
        return

    st.session_state.chat_history.append(user_question)
    st.session_state.chat_history.append(response)


def render_chat_history():
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg), unsafe_allow_html=True)


def main():
    load_dotenv()

    st.set_page_config(page_title="Chat with PDFs", page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "doc_intro" not in st.session_state:
        st.session_state.doc_intro = ""

    st.header("Chat with PDFs :robot_face:")
    st.caption("OpenAI version: PDF extraction + chunking + embeddings + FAISS + chat history.")

    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_input("Ask questions about your documents:")
        submitted = st.form_submit_button("Send")

    if submitted:
        cleaned_question = user_question.strip()
        if cleaned_question:
            handle_userinput(cleaned_question)
        else:
            st.warning("Please enter a question.")

    if st.session_state.chat_history:
        render_chat_history()

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )

        if st.button("Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF first.")
            else:
                with st.spinner("Processing"):
                    raw_text = get_pdf_text(pdf_docs)

                    if not raw_text.strip():
                        st.warning("No readable text was found in the uploaded PDF(s).")
                        return

                    st.session_state.doc_intro = raw_text[:1500]

                    text_chunks = get_text_chunks(raw_text)

                    if not text_chunks:
                        st.warning("No text chunks could be created from the uploaded PDF(s).")
                        return

                    st.write(f"Total chunks: {len(text_chunks)}")

                    vectorstore = get_vectorstore(text_chunks)
                    if vectorstore is None:
                        return

                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.chat_history = []

                    st.success("Processing complete.")


if __name__ == "__main__":
    main()