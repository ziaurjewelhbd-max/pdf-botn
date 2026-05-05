import os
import sys
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


def start_bot():

    pdf_path = "document.pdf"

    if not os.path.exists(pdf_path):
        print("PDF file পাওয়া যায়নি")
        return

    print("Processing PDF...")

    # PDF load
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # split
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = splitter.split_documents(pages)

    # embeddings
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        task_type="RETRIEVAL_DOCUMENT",
    )

    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
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

    # LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

    # prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Answer using context only.\n\n{context}"),
        ("human", "{input}")
    ])

    doc_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    query = "এই পিডিএফ এর মূল বিষয় কী?"

    result = rag_chain.invoke({"input": query})

    print(result["answer"])

if __name__ == "__main__":
    start_bot()