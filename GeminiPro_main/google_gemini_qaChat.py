import os
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from keys.env file
# load_dotenv("../keys.env")
load_dotenv(os.path.join(os.path.dirname(__file__), '..', 'keys.env'))


# Configure the Gemini API with the loaded key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Gemini Pro model and get responses
model = genai.GenerativeModel("gemini-1.5-pro-latest")
chat = model.start_chat(history=[])

def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Initialize Streamlit app
st.set_page_config(page_title="Q&A Demo")
st.header("Gemini LLM Application")

# Initialize session state for chat history if not already present
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Input and button
user_input = st.text_input("Input:", key="input")
submit = st.button("Ask the question")

if submit and user_input:
    response = get_gemini_response(user_input)
    st.session_state['chat_history'].append(("You", user_input))

    st.subheader("The Response is")
    full_response = ""
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Bot", chunk.text))

# Display chat history
st.subheader("The Chat History is")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
