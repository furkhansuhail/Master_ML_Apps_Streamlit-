from dotenv import load_dotenv
import base64
import streamlit as st
import os
import io
from PIL import Image
from pdf2image import convert_from_bytes
import google.generativeai as genai


class ATSResumeApp:
    def __init__(self):
        load_dotenv('../keys.env')
        self.api_key = os.getenv("GOOGLE_API_KEY")
        genai.configure(api_key=self.api_key)

        self.POPPLER_PATH = r"C:\poppler-24.08.0\Library\bin"
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')

        self.input_prompt1 = """
        You are an experienced Technical Human Resource Manager. Your task is to review the provided resume against the job description.
        Please share your professional evaluation on whether the candidate's profile aligns with the role.
        Highlight the strengths and weaknesses of the applicant in relation to the specified job requirements.
        """

        self.input_prompt3 = """
        You are a skilled ATS (Applicant Tracking System) scanner with a deep understanding of data science and ATS functionality.
        Evaluate the resume against the provided job description. Give the percentage match, list missing keywords, and share final thoughts.
        """

    def get_gemini_response(self, prompt, pdf_content, job_description):
        response = self.model.generate_content([prompt, pdf_content[0], job_description])
        return response.text

    def input_pdf_setup(self, uploaded_file):
        if uploaded_file is not None:
            images = convert_from_bytes(uploaded_file.read(), poppler_path=self.POPPLER_PATH)
            first_page = images[0]

            img_byte_arr = io.BytesIO()
            first_page.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            return [{
                "mime_type": "image/jpeg",
                "data": base64.b64encode(img_byte_arr).decode()
            }]
        else:
            raise FileNotFoundError("No file uploaded")

    def run(self):
        st.set_page_config(page_title="ATS Resume Expert")
        st.header("ATS Resume Screening Tool")

        input_text = st.text_area("Paste the Job Description:", key="input")
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])

        if uploaded_file:
            st.success("PDF Uploaded Successfully ✅")

        submit1 = st.button("Tell Me About the Resume")
        submit3 = st.button("Percentage Match")

        col1, col2 = st.columns([1, 1])
        with col1:
            stop_button = st.button("Stop")
        with col2:
            shutdown_button = st.button("Shutdown App")

        if "stop" not in st.session_state:
            st.session_state.stop = False

        if stop_button:
            st.session_state.stop = True
            st.warning("Processing has been stopped by the user.")

        if shutdown_button:
            st.warning("Shutting down the app...")
            st.stop()
            os._exit(0)

        if submit1 and not st.session_state.stop:
            if uploaded_file:
                pdf_content = self.input_pdf_setup(uploaded_file)
                response = self.get_gemini_response(self.input_prompt1, pdf_content, input_text)
                st.subheader("The Response is:")
                st.write(response)
            else:
                st.warning("Please upload a resume.")

        elif submit3 and not st.session_state.stop:
            if uploaded_file:
                pdf_content = self.input_pdf_setup(uploaded_file)
                response = self.get_gemini_response(self.input_prompt3, pdf_content, input_text)
                st.subheader("The Response is:")
                st.write(response)
            else:
                st.warning("Please upload a resume.")

        if submit1 or submit3:
            st.session_state.stop = False


# --- Run App ---
if __name__ == "__main__":
    app = ATSResumeApp()
    app.run()
