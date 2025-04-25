
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os

st.set_page_config(page_title="Tariff Chatbot", layout="wide")
st.title("📄 Tariff Chatbot")

# Set your API key (you can also use st.secrets or environment variables securely)
openai_api_key = "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"

# Set the path to your static PDF file stored in the 'data' folder of your repo
pdf_path = "finalCopy.pdf"  # Update this to your file's location

if os.path.exists(pdf_path) and openai_api_key:
    with st.spinner("Processing document..."):
        # Load & chunk document
        loader = PyPDFLoader(pdf_path)  # Directly load from static PDF
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
else:
    st.warning(f"PDF not found at {pdf_path}. Make sure it's in the repo and the path is correct.")
