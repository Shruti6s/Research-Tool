import os
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
import langchain

load_dotenv()

st.title("RESEARCH TOOLS")
st.sidebar.title("Enter URLs to Research")
urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vectorstore_path = "faiss_index_store"

retriever = None
if process_url_clicked:
    with st.spinner("Loading and processing data from URLs..."):
        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(data)

        vectorstore = FAISS.from_documents(docs, embedding)
        vectorstore.save_local(vectorstore_path)
        retriever = vectorstore.as_retriever()
        st.success("URLs processed successfully!")
query = st.text_input("Ask a question based on the processed URLs:")

if query and retriever:
    langchain.debug = True
    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
    result = chain({"question": query}, return_only_outputs=True)

    st.header("Answer")
    st.subheader(result['answer'])

    if 'sources' in result:
        st.markdown(f"**Sources:** {result['sources']}")
