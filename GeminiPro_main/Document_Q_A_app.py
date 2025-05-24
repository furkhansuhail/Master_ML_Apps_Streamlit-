import streamlit as st
import os
import time
import shutil
import tempfile
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'keys.env'))
os.getenv("GROK_API_KEY")

class DocumentQAApp:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.getenv("GROK_API_KEY")
#         self.groq_api_key = os.getenv("GROK_API_KEY")
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
#
        self.llm = ChatGroq(groq_api_key=self.groq_api_key, model_name="Llama3-8b-8192")

        self.prompt_template = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            <context>
            Question: {input}
            """
        )

    def load_and_embed_documents(self, directory_path):
        st.info("🔍 Reading and embedding documents...")

        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        loader = PyPDFDirectoryLoader(directory_path)
        docs = loader.load()

        if not docs:
            st.error("No valid PDF documents found.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        vectors = FAISS.from_documents(final_documents, embeddings)
        st.session_state.vectors = vectors
        st.success("✅ Document embeddings completed.")

    def ask_question(self, question):
        if "vectors" not in st.session_state:
            st.warning("Please embed documents first.")
            return

        document_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': question})
        elapsed = time.process_time() - start_time
        st.write(f"Response time: {elapsed:.2f} seconds")
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")

    def run(self):
        st.title("📄 Gemma Document Q&A")

        # ---- Upload Section ----
        uploaded_files = st.file_uploader("📤 Upload PDF Documents", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            temp_dir = tempfile.mkdtemp()
            for file in uploaded_files:
                with open(os.path.join(temp_dir, file.name), "wb") as f:
                    f.write(file.read())
            if st.button("🔎 Embed Uploaded Documents"):
                self.load_and_embed_documents(temp_dir)
                # Optionally clean up: shutil.rmtree(temp_dir)

        # ---- Question Input ----
        st.markdown("---")
        user_input = st.text_input("❓ Ask a question from the uploaded documents")
        if user_input:
            self.ask_question(user_input)


# Run the app
if __name__ == "__main__":
    DocumentQAApp().run()


import streamlit as st
import os
import time
import google.generativeai as genai
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'keys.env'))



class DocumentQAApp:
    def __init__(self):
        load_dotenv()
        self.groq_api_key = os.getenv("GROK_API_KEY")
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        self.llm = ChatGroq(groq_api_key=self.groq_api_key, model_name="Llama3-8b-8192")

        self.prompt_template = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            <context>
            Question: {input}
            """
        )

        self.embeddings = None
        self.loader = None
        self.docs = None
        self.text_splitter = None
        self.final_documents = None
        self.vectors = None

    def load_and_embed_documents(self, directory_path="./us_census"):
        self.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        self.loader = PyPDFDirectoryLoader(directory_path)
        self.docs = self.loader.load()
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.final_documents = self.text_splitter.split_documents(self.docs[:20])
        self.vectors = FAISS.from_documents(self.final_documents, self.embeddings)
        st.session_state.vectors = self.vectors  # Save to session for reuse
        st.write("Vector Store DB is ready.")

    def ask_question(self, question):
        if "vectors" not in st.session_state:
            st.warning("Please embed documents first.")
            return

        document_chain = create_stuff_documents_chain(self.llm, self.prompt_template)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start_time = time.process_time()
        response = retrieval_chain.invoke({'input': question})
        elapsed = time.process_time() - start_time
        st.write(f"Response time: {elapsed:.2f} seconds")
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("--------------------------------")

    def run(self):
        st.title("Gemma Model Document Q&A")

        if st.button("Documents Embedding"):
            self.load_and_embed_documents()

        user_input = st.text_input("Enter Your Question From Documents")
        if user_input:
            self.ask_question(user_input)


# Run the app
if __name__ == "__main__":
    DocumentQAAppObj = DocumentQAApp()
    DocumentQAAppObj.run()
