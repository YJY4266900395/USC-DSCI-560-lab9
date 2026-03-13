import os
import re
import hashlib
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

from htmlTemplates import css, bot_template, user_template


CACHE_DIR = "vector_cache"
MODEL_DIR = "models"
MODEL_FILE = "llama-2-7b-chat.Q4_K_M.gguf"   # 或 llama-2-7b-chat.Q5_K_M.gguf
TOP_K = 6

os.makedirs(CACHE_DIR, exist_ok=True)


def setup_windows_dll_paths():
    if os.name != "nt":
        return

    base_dir = os.path.dirname(os.path.abspath(__file__))
    dll_dirs = [
        os.path.join(base_dir, ".venv", "Lib", "site-packages", "llama_cpp", "lib"),
        os.path.join(base_dir, ".venv", "Lib", "site-packages", "torch", "lib"),
    ]

    for dll_dir in dll_dirs:
        if os.path.isdir(dll_dir):
            os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
            try:
                os.add_dll_directory(dll_dir)
            except (AttributeError, FileNotFoundError, OSError):
                pass


def get_pdf_text(pdf_docs):
    print("STEP A: reading PDF text", flush=True)
    text = ""

    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    print(f"WARNING: failed to read page {page_num}: {e}", flush=True)
        except Exception as e:
            print(f"WARNING: failed to open PDF: {e}", flush=True)

    print(f"STEP A DONE: extracted {len(text)} characters", flush=True)
    return text


def get_text_chunks(text):
    print("STEP B: splitting text into chunks", flush=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_text(text)
    print(f"STEP B DONE: created {len(chunks)} chunks", flush=True)
    return chunks


def get_cache_key(raw_text):
    cache_basis = (
        f"{raw_text}|chunk_size=500|chunk_overlap=100|"
        f"embedding=all-MiniLM-L6-v2|top_k={TOP_K}|conversation_chain=yes"
    )
    return hashlib.md5(cache_basis.encode("utf-8")).hexdigest()


def get_vectorstore(text_chunks, cache_key):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    index_path = os.path.join(CACHE_DIR, f"{cache_key}.faiss")
    pkl_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")

    if os.path.exists(index_path) and os.path.exists(pkl_path):
        print("STEP C: loading cached FAISS vectorstore", flush=True)
        try:
            vectorstore = FAISS.load_local(
                folder_path=CACHE_DIR,
                embeddings=embeddings,
                index_name=cache_key,
                allow_dangerous_deserialization=True,
            )
        except TypeError:
            vectorstore = FAISS.load_local(
                folder_path=CACHE_DIR,
                embeddings=embeddings,
                index_name=cache_key,
            )

        print("STEP C DONE: cached vectorstore loaded", flush=True)
        return vectorstore

    print("STEP C: building new FAISS vectorstore", flush=True)
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)

    print("STEP C2: saving vectorstore to local cache", flush=True)
    vectorstore.save_local(folder_path=CACHE_DIR, index_name=cache_key)

    print("STEP C DONE: vectorstore created and cached", flush=True)
    return vectorstore


def build_llm():
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Please put {MODEL_FILE} into the models folder."
        )

    print(f"STEP D1: loading local llama model from {model_path}", flush=True)

    llm = LlamaCpp(
        model_path=model_path,
        n_ctx=4096,
        n_batch=512,
        temperature=0.0,
        max_tokens=220,
        verbose=False,
        f16_kv=True,
        n_gpu_layers=-1,
        n_threads=max(1, os.cpu_count() // 2),
    )

    print("STEP D1 DONE: local llama model loaded", flush=True)
    return llm


def get_conversation_chain(llm, vectorstore):
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are answering questions about a PDF document.\n"
            "Use ONLY the provided context.\n"
            "Do NOT use outside knowledge.\n"
            "Do NOT guess.\n"
            "If the answer is explicitly stated in the context, answer it directly.\n"
            "For fact questions, give a very short direct answer first.\n"
            "For list questions, use bullet points.\n"
            "If the answer is not in the context, say exactly: "
            "'I cannot find the answer in the provided document.'\n"
            "Do not mention information that is not clearly supported by the context.\n"
            "Keep the answer concise and faithful to the document.\n\n"
            "Context:\n{context}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
    )

    print("STEP D2: creating conversation memory", flush=True)
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    print("STEP D3: creating ConversationalRetrievalChain", flush=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": TOP_K}
        ),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=False,
    )

    print("STEP D3 DONE: conversation chain ready", flush=True)
    return conversation_chain


def is_general_chat_question(user_question):
    q = user_question.strip().lower()

    general_patterns = [
        r"can you speak chinese",
        r"do you speak chinese",
        r"can you speak english",
        r"who are you",
        r"what can you do",
        r"hello",
        r"hi\b",
        r"thanks",
        r"thank you",
        r"你会说中文吗",
        r"你能说中文吗",
        r"你会中文吗",
        r"你是谁",
        r"你能做什么",
        r"你好",
        r"谢谢",
    ]

    return any(re.search(pattern, q) for pattern in general_patterns)


def detect_question_language(user_question):
    if re.search(r"[\u4e00-\u9fff]", user_question):
        return "Chinese"
    return "English"


def get_recent_general_history_text(history, num_turns=2):
    if not history:
        return ""

    recent_turns = history[-num_turns:]
    lines = []
    for turn in recent_turns:
        lines.append(f"User: {turn['user']}")
        lines.append(f"Assistant: {turn['bot']}")
    return "\n".join(lines).strip()


def answer_general_question(llm, user_question, recent_history_text, answer_language):
    prompt = PromptTemplate(
        input_variables=["recent_history", "question", "answer_language"],
        template=(
            "You are a helpful bilingual assistant.\n"
            "You can understand and respond in English and Simplified Chinese.\n"
            "Answer in {answer_language}.\n"
            "Keep the answer concise and natural.\n"
            "Do not invent document-specific details when the question is general.\n\n"
            "Recent conversation:\n{recent_history}\n\n"
            "Question:\n{question}\n\n"
            "Answer:"
        ),
    )

    final_prompt = prompt.format(
        recent_history=recent_history_text,
        question=user_question,
        answer_language=answer_language
    )
    return llm(final_prompt).strip()


def get_answer_language(user_question):
    q_lower = user_question.lower()

    if (
        "answer in chinese" in q_lower
        or "please answer in chinese" in q_lower
        or "请用中文" in user_question
        or "用中文回答" in user_question
    ):
        return "Simplified Chinese"

    if (
        "answer in english" in q_lower
        or "please answer in english" in q_lower
        or "请用英文" in user_question
        or "用英文回答" in user_question
    ):
        return "English"

    return "Simplified Chinese" if detect_question_language(user_question) == "Chinese" else "English"


def normalize_text(text):
    return " ".join(text.split())


def clean_answer(answer):
    answer = answer.strip()

    answer = re.sub(r"\(Chunk\s*\d+\)", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"^Note:\s*", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\s+", " ", answer).strip()

    tail_patterns = [
        r"I cannot find the answer in the provided document\.\s*$",
        r"I cannot find the answer to this question in the provided document\.\s*$",
        r"我无法在提供的文档中找到答案。\s*$",
    ]
    for pattern in tail_patterns:
        answer = re.sub(pattern, "", answer, flags=re.IGNORECASE).strip()

    split_patterns = [
        r"\s+I cannot find the answer in the provided document\.",
        r"\s+I cannot find the answer to this question in the provided document\.",
        r"\s+我无法在提供的文档中找到答案。",
    ]
    for pattern in split_patterns:
        answer = re.split(pattern, answer, flags=re.IGNORECASE)[0].strip()

    answer = re.sub(r"\s+([.,;:!?])", r"\1", answer)

    return answer


def fallback_numeric_fact_answer(user_question, docs, answer_language):
    q = user_question.lower()
    numeric_cues = [
        "how many", "what size", "what is the size", "what percentage",
        "多少", "多大", "百分之几", "size"
    ]

    if not any(cue in q for cue in numeric_cues):
        return None

    combined = " ".join(doc.page_content for doc in docs)
    combined = normalize_text(combined)

    m = re.search(r"chunks?\s+of\s+size\s+(\d+)", combined, re.IGNORECASE)
    if m:
        num = m.group(1)
        if answer_language == "Simplified Chinese":
            return f"文档中提到的数值是 {num}。"
        return f"The document states the value is {num}."

    nums = re.findall(r"\b\d+(?:\.\d+)?%?\b", combined)
    if nums:
        if answer_language == "Simplified Chinese":
            return f"文档中提到的数字是 {nums[0]}。"
        return f"The document mentions {nums[0]}."

    return None


def answer_document_question(conversation_chain, vectorstore, user_question):
    answer_language = get_answer_language(user_question)
    final_question = (
        f"{user_question}\n\nPlease answer in {answer_language}."
    )

    print("STEP R1: querying conversation chain", flush=True)
    response = conversation_chain({"question": final_question})

    answer = response.get("answer", "").strip()
    answer = clean_answer(answer)

    docs = response.get("source_documents", [])
    print("STEP R2: retrieved documents from chain", flush=True)
    for idx, doc in enumerate(docs, start=1):
        preview = doc.page_content[:500].replace("\n", " ")
        print(f"[DOC {idx}] {preview}", flush=True)

    lowered = answer.lower()

    if (
        not answer
        or "i cannot find" in lowered
        or "cannot find the answer" in lowered
        or "does not explicitly state" in lowered
        or "not explicitly stated" in lowered
        or "我无法在提供的文档中找到答案" in answer
    ):
        fallback = fallback_numeric_fact_answer(user_question, docs, answer_language)
        if fallback is not None:
            print("STEP R3: minimal numeric fallback hit", flush=True)
            return fallback, docs

        if answer_language == "Simplified Chinese":
            answer = "我无法在提供的文档中找到答案。"
        else:
            answer = "I cannot find the answer in the provided document."

    return answer, docs


def render_chat_history(history):
    for turn in history:
        st.write(
            user_template.replace("{{MSG}}", turn["user"]),
            unsafe_allow_html=True
        )
        st.write(
            bot_template.replace("{{MSG}}", turn["bot"]),
            unsafe_allow_html=True
        )


def handle_userinput(user_question):
    if not user_question or not user_question.strip():
        return

    user_question = user_question.strip()
    print(f"STEP E: received question -> {user_question}", flush=True)

    if is_general_chat_question(user_question):
        print("STEP E1: detected general chat question", flush=True)

        if st.session_state.llm is None:
            st.warning("General chat model is not ready yet.")
            return

        recent_history = get_recent_general_history_text(
            st.session_state.display_history,
            num_turns=2
        )
        answer_language = "Simplified Chinese" if detect_question_language(user_question) == "Chinese" else "English"

        try:
            answer = answer_general_question(
                st.session_state.llm,
                user_question,
                recent_history,
                answer_language
            )
        except Exception as e:
            st.error(f"General chat failed: {e}")
            print(f"STEP E1 ERROR: {e}", flush=True)
            return

        st.session_state.display_history.append({
            "user": user_question,
            "bot": answer
        })
        print("STEP E1 DONE: general history updated", flush=True)
        return

    print("STEP E2: detected document-related question", flush=True)

    if st.session_state.conversation is None or st.session_state.vectorstore is None:
        st.warning("Please upload and process your PDFs first.")
        print("STEP E2 STOP: conversation or vectorstore is None", flush=True)
        return

    try:
        answer, docs = answer_document_question(
            st.session_state.conversation,
            st.session_state.vectorstore,
            user_question
        )
    except Exception as e:
        st.error(f"Document question failed: {e}")
        print(f"STEP E2 ERROR: {e}", flush=True)
        return

    st.session_state.last_source_docs = docs
    st.session_state.display_history.append({
        "user": user_question,
        "bot": answer
    })
    print("STEP E2 DONE: document history updated", flush=True)


def main():
    print("STEP 0: app starting", flush=True)
    setup_windows_dll_paths()
    load_dotenv()

    st.set_page_config(page_title="Chat with PDFs", page_icon=":robot_face:")
    st.write(css, unsafe_allow_html=True)

    if "llm" not in st.session_state:
        st.session_state.llm = None
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "display_history" not in st.session_state:
        st.session_state.display_history = []
    if "last_source_docs" not in st.session_state:
        st.session_state.last_source_docs = []

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True
        )

        if st.button("Process"):
            print("STEP P1: Process button clicked", flush=True)

            if not pdf_docs:
                st.warning("Please upload at least one PDF first.")
                print("STEP P1 STOP: no PDFs uploaded", flush=True)
            else:
                with st.spinner("Processing"):
                    try:
                        raw_text = get_pdf_text(pdf_docs)

                        if not raw_text.strip():
                            st.warning("No readable text was found in the uploaded PDF(s).")
                            print("STEP P2 STOP: raw_text empty", flush=True)
                            return

                        text_chunks = get_text_chunks(raw_text)

                        if not text_chunks:
                            st.warning("No text chunks could be created from the uploaded PDF(s).")
                            print("STEP P3 STOP: no chunks created", flush=True)
                            return

                        cache_key = get_cache_key(raw_text)

                        print("STEP P4: creating/loading vectorstore", flush=True)
                        st.session_state.vectorstore = get_vectorstore(text_chunks, cache_key)

                        if st.session_state.llm is None:
                            print("STEP P5: loading shared LLM", flush=True)
                            st.session_state.llm = build_llm()

                        print("STEP P6: creating conversation chain", flush=True)
                        st.session_state.conversation = get_conversation_chain(
                            st.session_state.llm,
                            st.session_state.vectorstore
                        )

                        st.session_state.display_history = []
                        st.session_state.last_source_docs = []

                        print("STEP P7 DONE: processing complete", flush=True)
                        st.success("Processing complete.")

                    except Exception as e:
                        st.error(f"Processing failed: {e}")
                        print(f"STEP P ERROR: {e}", flush=True)

    st.header("Chat with PDFs :robot_face:")
    st.caption("Conversation chain + local open-source models + light answer cleaning.")

    with st.form("question_form", clear_on_submit=True):
        user_question = st.text_input("Ask questions about your documents:")
        submitted = st.form_submit_button("Send")

    if submitted:
        cleaned_question = user_question.strip()
        if cleaned_question:
            handle_userinput(cleaned_question)
        else:
            st.warning("Please enter a question.")

    if st.session_state.display_history:
        render_chat_history(st.session_state.display_history)


if __name__ == "__main__":
    main()