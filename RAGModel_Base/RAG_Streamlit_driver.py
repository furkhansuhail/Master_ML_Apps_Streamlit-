import os
import sys
import time
import base64
import torch
from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer
from sentence_transformers import util

import streamlit as st

# Load environment variables
dotenv_path = Path(__file__).resolve().parent.parent / "keys.env"
load_dotenv(dotenv_path=dotenv_path)

hf_token = os.getenv("HF_TOKEN")  # Updated casing for consistency
google_token = os.getenv("GOOGLE_API_KEY")

# Add submodule path
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import local modules
from RAGModel_Base.Rag_Streamlit import RagModel
from RAGModel_Base.RagSearch import Search_Semantic, LLmSearch
from RAGModel_Base.ImportsForRag import *

class RAGUploaderApp:
    def __init__(self):
        self.user_data_dir = Path("userData")
        self.user_data_dir.mkdir(exist_ok=True)
        self.file_path = None
        self.rag_model = None

    def return_to_dashboard(self):
        if st.button("🔙 Return to Main Page"):
            st.session_state.page = "main"

    def configure_streamlit(self):
        st.title("📄 Upload PDF for RAG Model")

    def upload_file(self):
        return st.file_uploader("Upload a PDF Document", type=["pdf"])

    def save_uploaded_file(self, uploaded_file):
        file_path = self.user_data_dir / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        max_wait_time = 5
        wait_time = 0
        while (not file_path.exists() or file_path.stat().st_size == 0) and wait_time < max_wait_time:
            time.sleep(0.5)
            wait_time += 0.5

        if not file_path.exists() or file_path.stat().st_size == 0:
            st.error("❌ File saving failed or incomplete.")
            return None

        return file_path

    def display_pdf(self, file_path):
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 5:
            st.warning(f"📄 PDF is too large to display inline ({file_size_mb:.2f} MB).")
            with open(file_path, "rb") as f:
                st.download_button(label="📥 Download PDF", data=f, file_name=file_path.name)
        else:
            with open(file_path, "rb") as pdf_file:
                base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
                pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="1000px" type="application/pdf"></iframe>'
                st.markdown(pdf_display, unsafe_allow_html=True)

    # def display_pdf(self, file_path):
    #     with open(file_path, "rb") as pdf_file:
    #         base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
    #         pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    #         st.markdown(pdf_display, unsafe_allow_html=True)

    def process_with_rag_model(self, file_path):
        with st.spinner("Processing the document and initializing the model..."):
            try:
                self.rag_model = RagModel(file_path)
                st.success("✅ RAG Model initialized successfully!")
            except Exception as e:
                st.error(f"❌ Error initializing RAG model: {e}")
                return

        st.markdown("---")
        st.subheader("🧠 Choose Mode")
        selected_mode = st.radio("Select RAG Mode:", ["Semantic Search", "LLM Answer Generator"])

        if selected_mode == "Semantic Search":
            st.subheader("🔍 Semantic Search")
            semantic_query = st.text_input("Enter your semantic search query here:", key="semantic_input")
            if st.button("Run Semantic Search"):
                self.run_semantic_search(semantic_query)

        elif selected_mode == "LLM Answer Generator":
            st.subheader("🤖 LLM Answer Generator")
            llm_query = st.text_input("Enter your query for the local LLM:", key="llm_input")
            if st.button("Run LLM Response"):
                self.run_llm_prompt(llm_query)

    def run_semantic_search(self, query):
        if not query:
            st.warning("Please enter a query before searching.")
            return
        with st.spinner("Running Semantic Search..."):
            try:
                query_embedding = self.rag_model.embedding_model.encode(query, convert_to_tensor=True).to(
                    self.rag_model.device)
                dot_scores = util.dot_score(a=query_embedding, b=self.rag_model.embeddings)[0]
                top_results = torch.topk(dot_scores, k=5)
                st.markdown("### 📄 Top Matching Chunks:")
                for score, idx in zip(top_results[0], top_results[1]):
                    chunk = self.rag_model.pages_and_chunks[idx]
                    st.markdown(
                        f"**Score:** {score:.4f}  \n**Page:** {chunk['page_number']}  \n**Text:** {chunk['sentence_chunk']}")
            except Exception as e:
                st.error(f"Semantic search failed: {e}")

    def run_llm_prompt(self, query):
        if not query:
            st.warning("Please enter a query before generating.")
            return
        with st.spinner("Generating LLM response..."):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=self.rag_model.model_id,
                    use_auth_token=hf_token
                )
                dialogue_template = [{"role": "user", "content": query}]
                prompt = tokenizer.apply_chat_template(dialogue_template, tokenize=False, add_generation_prompt=True)
                input_ids = tokenizer(prompt, return_tensors="pt").to(self.rag_model.device)
                outputs = self.rag_model.llm_model.generate(**input_ids, max_new_tokens=256)
                decoded = tokenizer.decode(outputs[0])
                response = decoded.replace(prompt, '').replace('<bos>', '').replace('<eos>', '')
                st.markdown("### 🤖 LLM Response:")
                st.write(response)
            except Exception as e:
                st.error(f"LLM generation failed: {e}")

    def run(self):
        self.configure_streamlit()
        uploaded_file = self.upload_file()

        if uploaded_file is not None:
            self.file_path = self.save_uploaded_file(uploaded_file)
            if self.file_path:
                st.success(f"✅ Uploaded and saved: {self.file_path}")
                self.display_pdf(self.file_path)

        if self.file_path:
            if st.button("🚀 Start RAG Model Training"):
                self.process_with_rag_model(self.file_path)

            st.markdown("### 🔍 Semantic Search")
            semantic_query = st.text_input("Enter your semantic search query:")
            if semantic_query:
                if semantic_query.strip().lower() == "exit":
                    st.warning("Shutting down the Streamlit app...")
                    os._exit(0)
                result = Search_Semantic(semantic_query)
                st.markdown("**🔎 Semantic Search Result:**")
                st.text_area("Result", result.result_text, height=400)

            st.markdown("### 💬 LLM Search")
            llm_query = st.text_input("Enter your LLM query:")
            if llm_query:
                if llm_query.strip().lower() == "exit":
                    st.warning("Shutting down the Streamlit app...")
                    os._exit(0)
                result = LLmSearch(llm_query)
                st.markdown("**🔎 LLM Search Result:**")
                st.text_area("Result", result.result_text, height=400)

        st.markdown("---")

        if st.button("🔙 Return to Main Page"):
            st.session_state.go_home = True
            st.rerun()

        if st.button("🛑 Stop App"):
            st.success("App stopped. You may now close the tab or refresh to restart.")
            st.stop()

        if st.button("🛑 Force Close App"):
            st.warning("Shutting down the Streamlit app...")
            os._exit(0)

        # if st.button("🔄 Rerun App"):
        #     st.rerun()


# from RAGModel_Base.ImportsForRag import *
# from dotenv import load_dotenv
# # from MainApp import main_dashboard
#
# # Ensure keys.env is loaded
# dotenv_path = Path(__file__).resolve().parent.parent / "keys.env"
# load_dotenv(dotenv_path=dotenv_path)
#
# hf_token = os.getenv("hf_Token")
# google_token = os.getenv("GOOGLE_API_KEY")
#
# # Add submodule path
# sys.path.append(str(Path(__file__).resolve().parent))
#
# # Local module imports
# # from Base_RagModel.Rag_Streamlit import RagModel
# # from Base_RagModel.RagSearch import Search_Semantic, LLmSearch
# # from Base_RagModel.ImportsForRag import *
#
# from RAGModel_Base.Rag_Streamlit import RagModel
# from RAGModel_Base.RagSearch import Search_Semantic, LLmSearch
# from RAGModel_Base.ImportsForRag import *
#
# from streamlit.runtime.scriptrunner import rerun
# class RAGUploaderApp:
#     def __init__(self):
#         self.user_data_dir = Path("userData")
#         self.user_data_dir.mkdir(exist_ok=True)
#         self.file_path = None
#         self.rag_model = None
#
#     def return_to_dashboard(self):
#         if st.button("🔙 Return to Main Page"):
#             st.session_state.page = "main"
#
#     def configure_streamlit(self):
#         # DO NOT CALL st.set_page_config() here
#         st.title("📄 Upload PDF for RAG Model")
#
#     def upload_file(self):
#         return st.file_uploader("Upload a PDF Document", type=["pdf"])
#
#     def save_uploaded_file(self, uploaded_file):
#         file_path = self.user_data_dir / uploaded_file.name
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.read())
#
#         max_wait_time = 5
#         wait_time = 0
#         while (not file_path.exists() or file_path.stat().st_size == 0) and wait_time < max_wait_time:
#             time.sleep(0.5)
#             wait_time += 0.5
#
#         if not file_path.exists() or file_path.stat().st_size == 0:
#             st.error("❌ File saving failed or incomplete.")
#             return None
#
#         return file_path
#
#     def display_pdf(self, file_path):
#         with open(file_path, "rb") as pdf_file:
#             base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
#             pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
#             st.markdown(pdf_display, unsafe_allow_html=True)
#
#     def process_with_rag_model(self, file_path):
#         with st.spinner("Processing the document and initializing the model..."):
#             try:
#                 self.rag_model = RagModel(file_path)
#                 st.success("✅ RAG Model initialized successfully!")
#             except Exception as e:
#                 st.error(f"❌ Error initializing RAG model: {e}")
#                 return
#
#         st.markdown("---")
#         st.subheader("🧠 Choose Mode")
#         selected_mode = st.radio("Select RAG Mode:", ["Semantic Search", "LLM Answer Generator"])
#
#         if selected_mode == "Semantic Search":
#             st.subheader("🔍 Semantic Search")
#             semantic_query = st.text_input("Enter your semantic search query here:", key="semantic_input")
#             if st.button("Run Semantic Search"):
#                 self.run_semantic_search(semantic_query)
#
#         elif selected_mode == "LLM Answer Generator":
#             st.subheader("🤖 LLM Answer Generator")
#             llm_query = st.text_input("Enter your query for the local LLM:", key="llm_input")
#             if st.button("Run LLM Response"):
#                 self.run_llm_prompt(llm_query)
#
#     def run_semantic_search(self, query):
#         if not query:
#             st.warning("Please enter a query before searching.")
#             return
#         with st.spinner("Running Semantic Search..."):
#             try:
#                 query_embedding = self.rag_model.embedding_model.encode(query, convert_to_tensor=True).to(
#                     self.rag_model.device)
#                 dot_scores = util.dot_score(a=query_embedding, b=self.rag_model.embeddings)[0]
#                 top_results = torch.topk(dot_scores, k=5)
#                 st.markdown("### 📄 Top Matching Chunks:")
#                 for score, idx in zip(top_results[0], top_results[1]):
#                     chunk = self.rag_model.pages_and_chunks[idx]
#                     st.markdown(
#                         f"**Score:** {score:.4f}  \n**Page:** {chunk['page_number']}  \n**Text:** {chunk['sentence_chunk']}")
#             except Exception as e:
#                 st.error(f"Semantic search failed: {e}")
#
#     def run_llm_prompt(self, query):
#         if not query:
#             st.warning("Please enter a query before generating.")
#             return
#         with st.spinner("Generating LLM response..."):
#             try:
#                 tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.rag_model.model_id,
#                                                           token=hf_token)
#                 dialogue_template = [{"role": "user", "content": query}]
#                 prompt = tokenizer.apply_chat_template(dialogue_template, tokenize=False, add_generation_prompt=True)
#                 input_ids = tokenizer(prompt, return_tensors="pt").to(self.rag_model.device)
#                 outputs = self.rag_model.llm_model.generate(**input_ids, max_new_tokens=256)
#                 decoded = tokenizer.decode(outputs[0])
#                 response = decoded.replace(prompt, '').replace('<bos>', '').replace('<eos>', '')
#                 st.markdown("### 🤖 LLM Response:")
#                 st.write(response)
#             except Exception as e:
#                 st.error(f"LLM generation failed: {e}")
#
#     def run(self):
#         self.configure_streamlit()
#         uploaded_file = self.upload_file()
#
#         if uploaded_file is not None:
#             self.file_path = self.save_uploaded_file(uploaded_file)
#             if self.file_path:
#                 st.success(f"✅ Uploaded and saved: {self.file_path}")
#                 self.display_pdf(self.file_path)
#
#         if self.file_path:
#             if st.button("🚀 Start RAG Model Training"):
#                 self.process_with_rag_model(self.file_path)
#
#             st.markdown("### 🔍 Semantic Search")
#             semantic_query = st.text_input("Enter your semantic search query:")
#             if semantic_query:
#                 if semantic_query.strip().lower() == "exit":
#                     st.warning("Shutting down the Streamlit app...")
#                     os._exit(0)
#                 result = Search_Semantic(semantic_query)
#                 st.markdown("**🔎 Semantic Search Result:**")
#                 st.text_area("Result", result.result_text, height=400)
#
#             st.markdown("### 💬 LLM Search")
#             llm_query = st.text_input("Enter your LLM query:")
#             if llm_query:
#                 if llm_query.strip().lower() == "exit":
#                     st.warning("Shutting down the Streamlit app...")
#                     os._exit(0)
#                 result = LLmSearch(llm_query)
#                 st.markdown("**🔎 Semantic Search Result:**")
#                 st.text_area("Result", result.result_text, height=400)
#
#         st.markdown("---")
#         # In RAG_Streamlit_driver.py
#         if st.button("🔙 Return to Main Page"):
#             st.session_state.go_home = True
#             st.experimental_rerun()
#
#         if st.button("🛑 Stop App"):
#             st.success("App stopped. You may now close the tab or refresh to restart.")
#             st.stop()
#
#         if st.button("🛑 Force Close App"):
#             st.warning("Shutting down the Streamlit app...")
#             os._exit(0)
#
#         if st.button("🔄 Rerun App"):
#             st.experimental_rerun()
#
