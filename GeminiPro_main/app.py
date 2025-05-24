# Q&A Chatbot
from dotenv import load_dotenv
import streamlit as st
import os
import pathlib
import textwrap
import google.generativeai as genai
from IPython.display import Markdown

load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'keys.env'))

def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## Function to load OpenAI model and get response
def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(question)
    return response.text

## Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("Gemini Application")

input = st.text_input("Input: ", key="input")

# Add columns to align Ask and Stop buttons side by side
col1, col2 = st.columns([1, 1])
with col1:
    ask_clicked = st.button("Ask the question")
with col2:
    stop_clicked = st.button("Stop")

# Use session state to handle stop logic
if 'stop' not in st.session_state:
    st.session_state.stop = False

if stop_clicked:
    st.session_state.stop = True
    st.warning("Generation stopped.")

if ask_clicked and not st.session_state.stop:
    response = get_gemini_response(input)
    st.subheader("The Response is")
    st.write(response)

# Reset stop state after Ask is done
if ask_clicked:
    st.session_state.stop = False
