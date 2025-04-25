pip install streamlit langchain openai faiss-cpu unstructured pypdf tiktoken

import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import tempfile
import os

st.set_page_config(page_title="Tariff Chatbot", layout="wide")
st.title("📄 Tariff Chatbot")

# Set your API key (you can also use st.secrets or environment variables securely)
openai_api_key = st.text_input("Enter your OpenAI API key", type="password")

uploaded_file = st.file_uploader("Upload your Tariff PDF", type="pdf")

if uploaded_file and openai_api_key:
    with st.spinner("Processing document..."):
        # Save PDF to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load & chunk document
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = splitter.split_documents(pages)

        # Embed & store in FAISS
        os.environ["OPENAI_API_KEY"] = openai_api_key
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(docs, embeddings)

        retriever = vectorstore.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(model_name="gpt-3.5-turbo"),
            retriever=retriever,
            return_source_documents=True
        )

        st.success("Document ready! Ask your questions below 👇")

        query = st.text_input("Ask a question about the tariff document")
        if query:
            result = qa_chain({"query": query})
            st.write("### Answer:")
            st.markdown(result["result"])

            with st.expander("Show source document chunks"):
                for doc in result["source_documents"]:
                    st.markdown(f"**Page snippet:**\n{doc.page_content}")
