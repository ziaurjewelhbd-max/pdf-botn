import streamlit as st

# --- ফেসবুক ভেরিফিকেশন কোড শুরু ---
params = st.query_params

if "hub.mode" in params and "hub.verify_token" in params:
    # এখানে 'my_pdf_bot_token' হলো আপনার ভেরিফাই টোকেন
    if params["hub.verify_token"] == "my_pdf_bot_token":
        st.write(params["hub.challenge"])
        st.stop()
# --- ফেসবুক ভেরিফিকেশন কোড শেষ ---

import os
import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

st.set_page_config(page_title="PDF Bot", page_icon="📄", layout="wide")

st.title("PDF Bot")
st.write("Upload one or more PDF files, then ask a question about their content.")

uploaded_files = st.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True,
    help="Select one or more PDF files to process.",
)

query = st.text_input("Ask a question", value="এই পিডিএফ এর মূল বিষয় কী?")
run_button = st.button("Run")

if run_button:
    if not uploaded_files:
        st.warning("দয়া করে প্রথমে একটি বা একাধিক PDF আপলোড করুন।")
    elif not query.strip():
        st.warning("দয়া করে একটি প্রশ্ন লিখুন।")
    else:
        with st.spinner("Processing PDFs..."):
            try:
                docs = []
                for uploaded_file in uploaded_files:
                    file_path = Path("uploaded_pdfs") / uploaded_file.name
                    file_path.parent.mkdir(exist_ok=True, parents=True)
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.read())
                    loader = PyPDFLoader(str(file_path))
                    pages = loader.load()
                    docs.extend(pages)

                if not docs:
                    st.error("PDF থেকে কোন টেক্সট পাওয়া যায়নি।")
                else:
                    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                    split_docs = splitter.split_documents(docs)

                    embeddings = GoogleGenerativeAIEmbeddings(
                        model="models/gemini-embedding-2",
                        task_type="RETRIEVAL_DOCUMENT",
                    )

                    texts = [doc.page_content for doc in split_docs]
                    metadatas = [doc.metadata for doc in split_docs]
                    text_embeddings = [
                        (text, embeddings.embed_query(text))
                        for text in texts
                    ]
                    vectorstore = FAISS.from_embeddings(
                        text_embeddings,
                        embeddings,
                        metadatas=metadatas,
                    )

                    retriever = vectorstore.as_retriever()
                    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
                    prompt = ChatPromptTemplate.from_messages([
                        ("system", "Answer using context only.\n\n{context}"),
                        ("human", "{input}"),
                    ])

                    doc_chain = create_stuff_documents_chain(llm, prompt)
                    rag_chain = create_retrieval_chain(retriever, doc_chain)

                    result = rag_chain.invoke({"input": query})

                    st.success("Done")
                    st.markdown("### Answer")
                    st.write(result["answer"])
            except Exception as exc:
                st.error(f"Error: {exc}")
                raise
