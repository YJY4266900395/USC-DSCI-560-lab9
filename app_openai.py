import re
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from html_modified import css, bot_template, user_template


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the question based on the context below. "
                   "If the answer is not in the context, say you don't know.\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(retriever.invoke(x["question"])))
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def _is_intro_question(q):
    patterns = [
        r"introduce yourself", r"introduce the document", r"what (is|are) (this|the) (document|pdf|book)",
        r"what document", r"summarize (this|the) (document|pdf|book)", r"what('s| is) (this|the) (document|pdf|book)",
        r"tell me about (this|the) (document|pdf|book)", r"what did i upload", r"what is (uploaded|the uploaded)",
    ]
    return any(re.search(p, q.lower()) for p in patterns)


def handle_userinput(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process PDF documents first!")
        return

    # For intro-style questions, bypass FAISS and use the actual first section
    # of the document to prevent hallucinating the document title/content
    if _is_intro_question(user_question) and st.session_state.get("doc_intro"):
        llm = ChatOpenAI()
        intro_prompt = (
            "You are answering questions about a PDF document.\n"
            "Use ONLY the provided context. Do NOT use outside knowledge. Do NOT guess.\n"
            "If the document title or name is not explicitly stated in the context, do NOT invent one.\n"
            "Keep the answer concise and faithful to the document.\n\n"
            f"Context (first section of the document):\n{st.session_state.doc_intro}\n\n"
            f"Question:\n{user_question}\n\nAnswer:"
        )
        response = llm.invoke([HumanMessage(content=intro_prompt)]).content
        st.session_state.chat_history.append(user_question)
        st.session_state.chat_history.append(response)
        for i, msg in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", msg), unsafe_allow_html=True)
        return

    # Build message objects from flat chat_history list
    history = []
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            history.append(HumanMessage(content=msg))
        else:
            history.append(AIMessage(content=msg))

    response = st.session_state.conversation.invoke({
        "question": user_question,
        "chat_history": history,
    })

    st.session_state.chat_history.append(user_question)
    st.session_state.chat_history.append(response)

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
    user_question = st.text_input("Ask questions about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                st.session_state.doc_intro = raw_text[:1500]
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorstore(text_chunks)
                st.session_state.conversation = get_conversation_chain(vectorstore)
                st.session_state.chat_history = []


if __name__ == '__main__':
    main()
