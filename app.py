import streamlit as st
import os
import streamlit as st
import pickle
import time

from dotenv import load_dotenv
load_dotenv()

# LLM
from langchain_groq import ChatGroq

# LangChain Core (modern LCEL)
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Community tools
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------- UI -------------------
st.title("NewsBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")
st.image("assets/NewsBot logo with futuristic tech design.png",width = 250)

urls = []
n = st.sidebar.number_input("Number of URLs", min_value=1, max_value=10, value=3)
for i in range(n):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store.pkl"

main_placeholder = st.empty()

# ------------------- LLM -------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

# ------------------- Embeddings -------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------- Process URLs -------------------
if process_url_clicked and urls:
    loader = WebBaseLoader(urls)

    main_placeholder.text("Loading data... ✅")
    data = loader.load()
    st.write(f"Loaded {len(data)} documents")  # debug

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    main_placeholder.text("Splitting text... ✅")
    docs = text_splitter.split_documents(data)

    main_placeholder.text("Creating embeddings... ✅")
    vectorstore = FAISS.from_documents(docs, embeddings)

    time.sleep(1)

    with open(file_path, "wb") as f:
        pickle.dump(vectorstore, f)

    main_placeholder.text("Processing complete ✅")

# ------------------- Query -------------------
query = st.text_input("Ask a question:")

if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

        # Debug retrieved docs
        docs = retriever.invoke(query)
        st.subheader("🔍 Retrieved Chunks (Debug)")
        # for i, doc in enumerate(docs):
        #     st.write(f"Chunk {i+1}:")
        #     st.write(doc.page_content[:300])

        # Prompt
        prompt = PromptTemplate.from_template(
            """Answer the question based on the context below.
If the answer is partially available, try your best.

Context:
{context}

Question: {question}

Answer:"""
        )

        # LCEL Chain
        chain = (
            {
                "context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        result = chain.invoke(query)

        st.header("Answer")
        st.write(result)

    else:
        st.warning("⚠️ Please process URLs first.")
